# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""PDF to image rendering and image extraction using PyMuPDF."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def render_pdf_pages(pdf_path: Path | str, dpi: int = 144) -> Iterator[Image.Image]:
    """Render each page of a PDF as an RGB PIL Image.

    This is a **generator** — pages are rendered one at a time so that only
    the current batch needs to be held in memory.

    If a single page fails to render (e.g. a corrupt structure tree), a
    small blank white image is yielded instead so that page indices stay
    in sync and the rest of the document is unaffected.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering resolution. 144 is the DeepSeek-OCR default.
             Higher values produce sharper images but use more memory.

    Yields:
        PIL Images, one per page.
    """
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_num, page in enumerate(doc):
            try:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                yield img
            except Exception as e:
                logger.warning(
                    "Failed to render page %d of %s: %s — yielding blank placeholder",
                    page_num + 1,
                    pdf_path,
                    e,
                )
                yield Image.new("RGB", (100, 100), (255, 255, 255))
    finally:
        doc.close()


def get_page_count(pdf_path: Path | str) -> int:
    """Return the number of pages in a PDF without rendering them."""
    doc = fitz.open(str(pdf_path))
    count: int = doc.page_count
    doc.close()
    return count


def extract_pdf_text(pdf_path: Path | str) -> list[str]:
    """Extract the embedded text layer from each page of a PDF.

    Returns a list of plain-text strings, one per page.  For born-digital
    PDFs this text is character-perfect.  For scanned PDFs the strings
    will be empty or near-empty.

    Individual page failures are logged and replaced with empty strings
    so that the page indices stay in sync with the rendering generator.
    """
    doc = fitz.open(str(pdf_path))
    texts: list[str] = []
    try:
        for page_num, page in enumerate(doc):
            try:
                texts.append(page.get_text())
            except Exception as e:
                logger.warning(
                    "Failed to extract text from page %d of %s: %s",
                    page_num + 1,
                    pdf_path,
                    e,
                )
                texts.append("")
        return texts
    finally:
        doc.close()


@dataclass
class ExtractedImage:
    """Metadata for an image extracted from a PDF page."""

    filename: str  # e.g. "p1_img1.png"
    relative_path: str  # e.g. "images/p1_img1.png" (for markdown refs)
    width: int
    height: int


@dataclass
class ImageExtractor:
    """Stateful image extractor that deduplicates across pages.

    Opens the PDF **once** on first use and keeps it open until
    ``close()`` is called (or used as a context manager).

    Usage::

        with ImageExtractor(output_dir=Path("doc/")) as extractor:
            extractor.open(pdf_path)
            for page_num in range(page_count):
                images = extractor.extract_page(page_num)
    """

    output_dir: Path
    min_size: int = 50
    _seen_xrefs: dict[int, str] = field(default_factory=dict, repr=False)
    _doc: fitz.Document | None = field(default=None, repr=False)

    def open(self, pdf_path: Path | str) -> None:
        """Open the PDF for extraction. Call once before extract_page()."""
        self.close()
        self._doc = fitz.open(str(pdf_path))

    def close(self) -> None:
        """Close the underlying PDF document."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def extract_page(
        self,
        page_index: int,
    ) -> list[ExtractedImage]:
        """Extract embedded images from a single PDF page.

        Args:
            page_index: Zero-based page index.

        Returns:
            List of ExtractedImage entries for images on this page.
            Images smaller than ``min_size`` in either dimension are skipped.
            Images already extracted from a previous page (same xref) are
            returned with the original filename (not saved again).
        """
        if self._doc is None:
            raise RuntimeError("ImageExtractor: call .open(pdf_path) first")

        doc = self._doc
        results: list[ExtractedImage] = []
        images_dir = self.output_dir / "images"

        page = doc[page_index]
        page_images = page.get_images(full=True)

        img_counter = 0
        for img_info in page_images:
            xref = img_info[0]

            # Already extracted from a previous page — reuse the filename
            if xref in self._seen_xrefs:
                filename = self._seen_xrefs[xref]
                rel_path = f"images/{filename}"
                # Read dimensions from the already-saved file
                saved = images_dir / filename
                if saved.exists():
                    with Image.open(saved) as im:
                        w, h = im.size
                else:
                    w, h = 0, 0
                results.append(ExtractedImage(filename, rel_path, w, h))
                continue

            # Extract raw image bytes from the PDF
            try:
                img_data = doc.extract_image(xref)
            except Exception:
                continue  # corrupt or unsupported image — skip silently

            if not img_data or "image" not in img_data:
                continue

            width = img_data.get("width", 0)
            height = img_data.get("height", 0)

            # Filter out tiny images (icons, bullets, spacers)
            if width < self.min_size or height < self.min_size:
                continue

            ext = img_data.get("ext", "png")
            img_counter += 1
            filename = f"p{page_index + 1}_img{img_counter}.{ext}"
            rel_path = f"images/{filename}"

            # Save the image
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / filename).write_bytes(img_data["image"])

            self._seen_xrefs[xref] = filename
            results.append(ExtractedImage(filename, rel_path, width, height))

        return results
