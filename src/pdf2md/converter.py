# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""Conversion orchestration - handles files and directories."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pdf2md.backends.base import OCRBackend
from pdf2md.pdf import ImageExtractor, extract_pdf_text, get_page_count, render_pdf_pages
from pdf2md.postprocess import (
    clean_ocr_output,
    combine_pages,
    merge_cross_page_content,
    restore_missing_image_refs,
    rewrite_image_references,
)

logger = logging.getLogger(__name__)

# Number of trailing lines from the previous page to pass as context
# to the next page.  5 lines is enough to hint at the current
# formatting style (table header, heading level, list style) without
# providing so much content that the model hallucinates a continuation
# instead of reading the actual image.
_CONTEXT_TAIL_LINES = 5


def _tail(text: str, n: int = _CONTEXT_TAIL_LINES) -> str:
    """Return the last *n* non-empty lines of *text*."""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _set_page_context(backend: OCRBackend, cleaned: str) -> None:
    """Pass the tail of *cleaned* to the backend for the next page.

    Only sets the context if the backend supports it (has a
    ``page_context`` attribute).  This keeps the converter decoupled
    from any specific backend implementation.
    """
    if hasattr(backend, "page_context"):
        backend.page_context = _tail(cleaned)  # type: ignore[attr-defined]


def _reset_page_context(backend: OCRBackend) -> None:
    """Clear cross-page context (e.g. at the start of a new file)."""
    if hasattr(backend, "page_context"):
        backend.page_context = None  # type: ignore[attr-defined]


@dataclass
class ProgressEvent:
    """Progress update emitted during conversion.

    Designed to be consumed by CLI progress bars or a future GUI.
    """

    stage: str  # "render", "ocr", "file_start", "file_done"
    message: str
    current: int = 0
    total: int = 0


ProgressCallback = Callable[[ProgressEvent], None] | None


def _supports_batching(backend: OCRBackend) -> bool:
    """Check if a backend implements the batching interface."""
    return hasattr(backend, "preprocess") and hasattr(backend, "generate_from_prepared")


class Converter:
    """Orchestrates PDF-to-Markdown conversion using a pluggable OCR backend.

    Each PDF produces a folder containing the ``.md`` file and an ``images/``
    subdirectory (if the PDF contains extractable images).

    Pages are rendered lazily (one at a time) to keep memory usage bounded
    regardless of document length.

    When the backend supports batching, pages are preprocessed on a
    background CPU thread while the GPU generates the current batch,
    overlapping the two for higher throughput.

    Usage::

        backend = create_backend("deepseek", ...)
        converter = Converter(backend, dpi=144, batch_size=2)
        converter.convert(Path("input.pdf"), Path("output/"), on_progress=cb)
    """

    def __init__(
        self,
        backend: OCRBackend,
        dpi: int = 144,
        batch_size: int = 1,
        merge_pages: bool = False,
        hybrid: bool = False,
        merge_only: bool = False,
    ) -> None:
        self.backend = backend
        self.dpi = dpi
        self.batch_size = max(1, batch_size)
        self.merge_pages = merge_pages
        self.hybrid = hybrid
        self.merge_only = merge_only

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Convert a PDF file or directory of PDFs to Markdown.

        When ``merge_only`` is set, *input_path* should be an output
        directory (or parent directory of output directories) from a
        previous run.  The saved OCR cache is loaded and only the
        post-processing pipeline (hybrid merge, cleanup, cross-page
        merge) is re-run.

        Args:
            input_path: Path to a .pdf file, a directory containing PDFs,
                        or (in merge-only mode) an output directory with
                        an ``ocr_cache/`` subfolder.
            output_path: Output folder (for a single PDF) or root output
                         directory (for a directory of PDFs).
            on_progress: Optional callback for progress updates.
        """
        if self.merge_only:
            self._merge_only(input_path, on_progress)
            return

        self.backend.load()

        try:
            if input_path.is_file():
                self._convert_file(input_path, output_path, on_progress)
            elif input_path.is_dir():
                self._convert_directory(input_path, output_path, on_progress)
            else:
                raise FileNotFoundError(f"Input path does not exist: {input_path}")
        finally:
            self.backend.unload()

    def _convert_file(
        self,
        input_path: Path,
        output_dir: Path,
        on_progress: ProgressCallback = None,
        display_name: str | None = None,
    ) -> None:
        """Convert a single PDF file to Markdown.

        Pages are rendered lazily -- only the current batch is held in
        memory at any time.
        """
        if on_progress:
            label = display_name or str(input_path)
            on_progress(ProgressEvent("file_start", label))

        # Reset cross-page context for each new file.
        # NOTE: Cross-page context is currently disabled (see below) but
        # the reset is kept so re-enabling context later works correctly.
        _reset_page_context(self.backend)

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / f"{input_path.stem}.md"

        # Get total page count without rendering
        total = get_page_count(input_path)
        if total == 0:
            md_path.write_text("", encoding="utf-8")
            return

        # Extract embedded text layer.
        # This is instant — no GPU needed — so we always do it.
        # The raw texts are saved alongside the OCR cache so that
        # --merge-only --hybrid can work even if the original run
        # didn't use --hybrid.
        raw_texts: list[str] = extract_pdf_text(input_path)
        logger.debug("Extracted raw text from %d pages", len(raw_texts))

        use_batching = self.batch_size > 1 and _supports_batching(self.backend)

        # Open the PDF once for image extraction
        with ImageExtractor(output_dir=output_dir) as extractor:
            extractor.open(input_path)

            # Render pages lazily (generator)
            page_iter = render_pdf_pages(input_path, self.dpi)

            if use_batching:
                pages = self._convert_pages_batched(
                    page_iter,
                    input_path,
                    extractor,
                    total,
                    on_progress,
                )
            else:
                pages = self._convert_pages_sequential(
                    page_iter,
                    input_path,
                    extractor,
                    total,
                    on_progress,
                )

        # -- Save OCR cache so that merge-only re-runs are possible --
        self._save_ocr_cache(output_dir, pages, raw_texts, input_path)

        # Combine pages and write the markdown file
        # -- Phase 1.5: hybrid merge (raw text + OCR) --
        cache_pages = list(pages)  # snapshot before hybrid merge for image restoration
        if self.hybrid and raw_texts:
            pages = self._hybrid_merge(pages, raw_texts, total, on_progress)

        # -- Phase 1.75: restore any image references lost during hybrid merge --
        images_dir = output_dir / "images"
        if images_dir.exists():
            pages, _restored = restore_missing_image_refs(pages, images_dir, cache_pages)

        # -- Phase 2: cross-page merge --
        if self.merge_pages:
            pages = merge_cross_page_content(pages)
        markdown = combine_pages(pages)
        md_path.write_text(markdown, encoding="utf-8")

        if on_progress:
            on_progress(ProgressEvent("file_done", str(md_path), current=total, total=total))

    # -- sequential path (batch_size=1 or no batching support) ---------------

    def _convert_pages_sequential(
        self,
        page_iter,
        input_path: Path,
        extractor: ImageExtractor,
        total: int,
        on_progress: ProgressCallback,
    ) -> list[str]:
        """Process pages one at a time.

        Consumes the page iterator lazily -- only one page is in memory
        at a time.  After each page, the tail of the cleaned output is
        passed to the backend as context for the next page so that
        tables, lists, and sections that span page boundaries are
        handled consistently.
        """
        pages: list[str] = []
        for i, image in enumerate(page_iter):
            if on_progress:
                on_progress(ProgressEvent("ocr", f"Page {i + 1}/{total}", current=i, total=total))

            # --- OCR ---
            try:
                raw_text = self.backend.process_image(image)
                cleaned = clean_ocr_output(raw_text)
            except Exception as e:
                cleaned = f"<!-- OCR ERROR on page {i + 1}: {e} -->"

            # Release the image immediately
            del image

            # --- Image extraction ---
            cleaned = self._extract_and_rewrite(cleaned, extractor, i)
            pages.append(cleaned)

            # Feed the tail of this page to the backend as context for
            # the next page.
            # DISABLED: cross-page context caused hallucination / formatting
            # regressions with Qwen-VL.  The backend plumbing is kept for
            # future experimentation.
            # _set_page_context(self.backend, cleaned)

        return pages

    # -- batched path --------------------------------------------------------

    def _convert_pages_batched(
        self,
        page_iter,
        input_path: Path,
        extractor: ImageExtractor,
        total: int,
        on_progress: ProgressCallback,
    ) -> list[str]:
        """Process pages in batches.

        Collects ``batch_size`` pages from the lazy iterator, preprocesses
        them, sends them to the backend in one call, then releases them
        before moving to the next batch.  Peak memory = 1 batch of pages.
        """
        pages: list[str] = []
        bs = self.batch_size
        page_index = 0

        # Consume the lazy iterator in chunks of batch_size
        batch_images: list = []
        for image in page_iter:
            batch_images.append(image)

            if len(batch_images) < bs:
                continue

            # Full batch collected -- process it
            batch_start = page_index
            self._process_batch(
                batch_images,
                batch_start,
                total,
                extractor,
                pages,
                on_progress,
            )
            page_index += len(batch_images)
            batch_images = []

        # Process any remaining pages (final partial batch)
        if batch_images:
            self._process_batch(
                batch_images,
                page_index,
                total,
                extractor,
                pages,
                on_progress,
            )

        return pages

    def _process_batch(
        self,
        batch_images: list,
        batch_start: int,
        total: int,
        extractor: ImageExtractor,
        pages: list[str],
        on_progress: ProgressCallback,
    ) -> None:
        """Preprocess, generate, and post-process a single batch of pages."""
        batch_size = len(batch_images)

        # Report progress for each page in this batch
        if on_progress:
            for offset in range(batch_size):
                pi = batch_start + offset
                on_progress(
                    ProgressEvent(
                        "ocr",
                        f"Page {pi + 1}/{total}",
                        current=pi,
                        total=total,
                    )
                )

        # Preprocess (CPU work -- base64 for Ollama, tensor prep for DeepSeek)
        try:
            prepared_items = [self.backend.preprocess(img) for img in batch_images]
        except Exception as e:
            logger.warning("Batch preprocess failed, falling back to sequential: %s", e)
            for offset, img in enumerate(batch_images):
                pi = batch_start + offset
                try:
                    raw = self.backend.process_image(img)
                    cleaned = clean_ocr_output(raw)
                except Exception as exc:
                    cleaned = f"<!-- OCR ERROR on page {pi + 1}: {exc} -->"
                cleaned = self._extract_and_rewrite(cleaned, extractor, pi)
                pages.append(cleaned)
            return

        # Release the PIL images -- we only need the prepared data now
        batch_images.clear()

        # GPU / Ollama inference for the batch
        try:
            raw_texts = self.backend.generate_from_prepared(prepared_items)
        except Exception:
            # Fall back to one-at-a-time for this batch
            raw_texts = []
            for prep in prepared_items:
                try:
                    results = self.backend.generate_from_prepared([prep])
                    raw_texts.append(results[0])
                except Exception as e:
                    raw_texts.append(f"<!-- OCR ERROR: {e} -->")

        # Post-process each page
        for offset in range(len(prepared_items)):
            pi = batch_start + offset
            try:
                cleaned = clean_ocr_output(raw_texts[offset])
            except Exception as e:
                cleaned = f"<!-- OCR ERROR on page {pi + 1}: {e} -->"

            cleaned = self._extract_and_rewrite(cleaned, extractor, pi)
            pages.append(cleaned)

        # Set context from the last page of this batch for the next batch.
        # DISABLED: see _convert_pages_sequential for rationale.
        # if pages:
        #     _set_page_context(self.backend, pages[-1])

    # -- Phase 1.5: hybrid merge (raw PDF text + OCR markdown) -----------------

    def _hybrid_merge(
        self,
        pages: list[str],
        raw_texts: list[str],
        total: int,
        on_progress: ProgressCallback,
    ) -> list[str]:
        """Merge raw PDF text with OCR markdown using a text LLM.

        For each page, the raw text (ground truth) and OCR markdown
        (structure + images) are sent to the merge model.  Pages where
        the raw text is empty or near-empty (scanned pages) are kept
        as-is.

        The last *_TAIL_LINES* lines of the previous page's merged
        output are forwarded so the model can continue open formatting
        structures (tables, lists, code blocks) across page boundaries.
        """
        merged_pages: list[str] = []
        _MIN_RAW_CHARS = 20  # skip hybrid for pages with very little text
        _TAIL_LINES = 15  # how many lines of context to pass forward

        prev_tail: str | None = None

        for i, page_text in enumerate(pages):
            if on_progress:
                on_progress(
                    ProgressEvent(
                        "hybrid",
                        f"Hybrid {i + 1}/{total}",
                        current=i,
                        total=total,
                    )
                )

            raw = raw_texts[i] if i < len(raw_texts) else ""
            if len(raw.strip()) < _MIN_RAW_CHARS:
                # Scanned page or near-empty — keep OCR output as-is
                merged_pages.append(page_text)
                prev_tail = None  # reset context — scanned page
                continue

            try:
                result = self.backend.hybrid_merge_text(
                    raw,
                    page_text,
                    prev_page_tail=prev_tail,
                )
                result = clean_ocr_output(result)  # strip any outer fences
            except Exception as e:
                logger.warning("Hybrid merge failed on page %d: %s", i + 1, e)
                result = page_text  # keep OCR on failure
            merged_pages.append(result)

            # Extract tail for the next page's context
            lines = result.splitlines()
            prev_tail = "\n".join(lines[-_TAIL_LINES:]) if lines else None

        if on_progress:
            on_progress(ProgressEvent("hybrid", "Hybrid done", current=total, total=total))
        return merged_pages

    # -- shared helpers ------------------------------------------------------

    @staticmethod
    def _extract_and_rewrite(
        cleaned: str,
        extractor: ImageExtractor,
        page_index: int,
    ) -> str:
        """Extract embedded images and rewrite markdown references."""
        try:
            extracted = extractor.extract_page(page_index)
            image_paths = [img.relative_path for img in extracted]
        except Exception:
            image_paths = []

        if image_paths:
            cleaned = rewrite_image_references(cleaned, image_paths)

        return cleaned

    # -- OCR cache (save / load) -----------------------------------------------

    _OCR_CACHE_DIR = "ocr_cache"

    def _save_ocr_cache(
        self,
        output_dir: Path,
        pages: list[str],
        raw_texts: list[str],
        input_path: Path,
    ) -> None:
        """Persist OCR results so that post-processing can be re-run later.

        Layout::

            output_dir/ocr_cache/
                metadata.json
                pages/0001.md  …
                raw/0001.txt   …
        """
        cache_dir = output_dir / self._OCR_CACHE_DIR
        pages_dir = cache_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        for i, page in enumerate(pages):
            (pages_dir / f"{i + 1:04d}.md").write_text(page, encoding="utf-8")

        if raw_texts:
            raw_dir = cache_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for i, text in enumerate(raw_texts):
                (raw_dir / f"{i + 1:04d}.txt").write_text(text, encoding="utf-8")

        metadata = {
            "version": 1,
            "source_pdf": str(input_path.resolve()),
            "source_pdf_name": input_path.name,
            "total_pages": len(pages),
            "dpi": self.dpi,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (cache_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved OCR cache (%d pages) to %s", len(pages), cache_dir)

    @staticmethod
    def _load_ocr_cache(output_dir: Path) -> tuple[list[str], list[str], dict]:
        """Load a previously saved OCR cache.

        Returns:
            (pages, raw_texts, metadata) — *raw_texts* may be empty if
            no raw text was cached (e.g. scanned-only PDF).
        """
        cache_dir = output_dir / Converter._OCR_CACHE_DIR
        meta_path = cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No OCR cache found in {output_dir}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        total = metadata["total_pages"]

        pages_dir = cache_dir / "pages"
        pages: list[str] = []
        for i in range(total):
            p = pages_dir / f"{i + 1:04d}.md"
            pages.append(p.read_text(encoding="utf-8") if p.exists() else "")

        raw_dir = cache_dir / "raw"
        raw_texts: list[str] = []
        if raw_dir.exists():
            for i in range(total):
                r = raw_dir / f"{i + 1:04d}.txt"
                raw_texts.append(r.read_text(encoding="utf-8") if r.exists() else "")

        return pages, raw_texts, metadata

    # -- merge-only path -------------------------------------------------------

    def _merge_only(
        self,
        input_path: Path,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Re-run post-processing on previously cached OCR results.

        *input_path* is either a single output directory containing an
        ``ocr_cache/`` subfolder, or a parent directory whose children
        contain ``ocr_cache/`` subfolders (batch re-merge).
        """
        cache_marker = input_path / self._OCR_CACHE_DIR / "metadata.json"

        if cache_marker.exists():
            # Single document output folder
            dirs = [input_path]
        else:
            # Search for all output folders with a cache
            dirs = sorted(
                p.parent for p in input_path.rglob(f"{self._OCR_CACHE_DIR}/metadata.json")
            )
            if not dirs:
                raise FileNotFoundError(f"No OCR cache found in {input_path} or its subdirectories")

        self.backend.load()
        try:
            for idx, out_dir in enumerate(dirs):
                self._merge_only_one(out_dir, idx, len(dirs), on_progress)
        finally:
            self.backend.unload()

    def _merge_only_one(
        self,
        output_dir: Path,
        file_idx: int,
        file_count: int,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Re-run post-processing on a single cached output directory."""
        pages, raw_texts, metadata = self._load_ocr_cache(output_dir)
        total = len(pages)

        source_name = metadata.get("source_pdf_name", output_dir.name)
        md_stem = Path(source_name).stem
        md_path = output_dir / f"{md_stem}.md"

        if file_count > 1:
            label = f"[{file_idx + 1}/{file_count}] {output_dir.name}"
        else:
            label = str(output_dir)
        if on_progress:
            on_progress(ProgressEvent("file_start", f"(merge-only) {label}"))

        logger.info(
            "Merge-only: loaded %d cached OCR pages from %s",
            total,
            output_dir,
        )

        # -- Phase 1.5: hybrid merge --
        cache_pages = list(pages)  # snapshot for image restoration
        if self.hybrid and raw_texts:
            pages = self._hybrid_merge(pages, raw_texts, total, on_progress)

        # -- Phase 1.75: restore any image references lost during hybrid merge --
        images_dir = output_dir / "images"
        if images_dir.exists():
            pages, _restored = restore_missing_image_refs(pages, images_dir, cache_pages)

        # -- Phase 2: cross-page merge --
        if self.merge_pages:
            pages = merge_cross_page_content(pages)

        markdown = combine_pages(pages)
        md_path.write_text(markdown, encoding="utf-8")

        if on_progress:
            on_progress(ProgressEvent("file_done", str(md_path), current=total, total=total))

    # -- directory conversion --------------------------------------------------

    def _convert_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Recursively convert all PDFs in a directory.

        Each PDF gets its own subfolder.  When ``output_dir`` equals
        ``input_dir`` (the default when no OUTPUT_PATH is given), output
        folders are placed next to each PDF::

            papers/report.pdf  ->  papers/report/report.md
                                   papers/report/images/...
            papers/sub/spec.pdf -> papers/sub/spec/spec.md

        When a separate ``output_dir`` is given, the input directory
        structure is mirrored inside it::

            input/sub/report.pdf  ->  output/sub/report/report.md
        """
        pdf_files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() == ".pdf")

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {input_dir}")

        failed: list[str] = []
        for file_idx, pdf_path in enumerate(pdf_files):
            # Mirror the input directory structure, with a subfolder per PDF
            relative = pdf_path.relative_to(input_dir)
            doc_folder = output_dir / relative.with_suffix("")

            label = f"[{file_idx + 1}/{len(pdf_files)}] {relative}"
            try:
                self._convert_file(pdf_path, doc_folder, on_progress, display_name=label)
            except Exception as e:
                logger.error("Failed to convert %s: %s", relative, e)
                failed.append(str(relative))

        if failed:
            logger.warning(
                "Finished with %d failed file(s): %s",
                len(failed),
                ", ".join(failed),
            )


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    """Determine the output path when the user doesn't specify one.

    Output is always a **folder** (not a bare ``.md`` file).

    Rules:
        - File  ``doc.pdf``       -> ``doc/``    (contains ``doc.md`` + ``images/``)
        - Directory ``papers/``   -> ``papers/`` (each PDF gets a sibling folder)

    If the user explicitly passes a path ending in ``.md``, strip the
    extension — the output is a folder, not a single file.
    """
    if output_path is not None:
        p = Path(output_path)
        # Guard against the common mistake of specifying an .md file path
        # when our output is actually a folder.
        if p.suffix.lower() == ".md":
            p = p.with_suffix("")
        return p

    # Single PDF -> folder named after the PDF stem
    if input_path.suffix.lower() == ".pdf":
        return input_path.parent / input_path.stem

    # Directory -> use the same directory (output folders placed next to PDFs)
    return input_path
