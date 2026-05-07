# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""Post-processing for raw OCR model output."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Separator inserted between pages when combining into a single markdown file.
PAGE_SEPARATOR = "\n\n---\n\n"

# Matches markdown image syntax: ![alt text](url)
# Captures: group(1) = full match, group(2) = alt text, group(3) = url
_IMAGE_REF_RE = re.compile(r"(!\[([^\]]*)\]\(([^)]*)\))")

# Matches an outer code fence that wraps the *entire* output.
# Captures the optional language tag so we can distinguish a full-page wrap
# (```markdown ... ```) from legitimate code blocks embedded in the content.
_OUTER_FENCE_RE = re.compile(
    r"\A\s*```[^\S\n]*\w*[^\S\n]*\n(.*?)\n\s*```\s*\Z",
    re.DOTALL,
)


def clean_ocr_output(raw_text: str) -> str:
    """Clean raw OCR output from DeepSeek-OCR (or similar) models.

    Removes model-specific tokens (grounding refs, EOS markers) and
    normalizes whitespace. The result is clean Markdown.

    Args:
        raw_text: Raw decoded text from the model.

    Returns:
        Cleaned Markdown string.
    """
    if not raw_text:
        return ""

    text = raw_text

    # Strip an outer code fence that wraps the entire output.
    # Some models (e.g. Qwen2.5-VL) tend to wrap everything in
    # ```markdown ... ```.  We only strip if the opening and closing
    # fences are the very first and last things in the output, so
    # legitimate code blocks *within* the content are preserved.
    fence_match = _OUTER_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1)

    # Remove end-of-sentence token used by DeepSeek models
    text = text.replace("\u003c\uff5cend\u2581of\u2581sentence\uff5c\u003e", "")
    # Also handle the literal form in case encoding differs
    text = text.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "")

    # Remove grounding tokens: <|ref|>...<|/ref|><|det|>...<|/det|>
    # These encode bounding-box layout info that we strip for clean markdown.
    grounding_pattern = r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>"
    text = re.sub(grounding_pattern, "", text, flags=re.DOTALL)

    # Remove any remaining orphaned grounding markers
    text = re.sub(r"<\|/?(?:ref|det|grounding)\|>", "", text)

    # Clean up LaTeX symbol aliases
    text = text.replace("\\coloneqq", ":=")
    text = text.replace("\\eqqcolon", "=:")

    # Collapse excessive blank lines (4+ newlines -> 2)
    text = re.sub(r"\n{4,}", "\n\n", text)
    text = re.sub(r"\n{3}", "\n\n", text)

    return text.strip()


def rewrite_image_references(text: str, image_paths: list[str]) -> str:
    """Patch image references in OCR output to point at extracted images.

    Strategy (match-then-append):
      1. Find all ``![alt](url)`` patterns the model produced.
      2. For each one, replace the URL with the next available extracted
         image path (matched by order on the page).
      3. If there are more extracted images than model references, append
         the remaining images at the end of the text.
      4. If the model produced no image references at all, append all
         extracted images at the end.

    Args:
        text: Cleaned OCR markdown for a single page.
        image_paths: Relative paths to extracted images for this page
                     (e.g. ``["images/p1_img1.png", "images/p1_img2.jpg"]``).

    Returns:
        Text with image references rewritten/appended.
    """
    if not image_paths:
        return text

    # Find existing image references in the model output
    matches = list(_IMAGE_REF_RE.finditer(text))

    # Track which extracted images have been consumed
    consumed = 0

    if matches:
        # Rewrite existing refs in order, replacing their URLs
        for match in matches:
            if consumed >= len(image_paths):
                break
            full_match = match.group(1)
            alt_text = match.group(2)
            new_ref = f"![{alt_text}]({image_paths[consumed]})"
            # Replace only the first occurrence of this exact match
            text = text.replace(full_match, new_ref, 1)
            consumed += 1

    # Append any remaining extracted images not matched to existing refs
    remaining = image_paths[consumed:]
    if remaining:
        lines = [f"\n![]({path})" for path in remaining]
        text = text.rstrip() + "\n" + "".join(lines)

    return text


def combine_pages(pages: list[str]) -> str:
    """Join per-page markdown strings with a horizontal-rule separator.

    Args:
        pages: List of cleaned markdown strings, one per page.

    Returns:
        Combined markdown document.
    """
    # Filter out completely empty pages but keep pages with content
    non_empty = [p for p in pages if p.strip()]
    return PAGE_SEPARATOR.join(non_empty)


# ---------------------------------------------------------------------------
# Cross-page merge helpers (v4)
# ---------------------------------------------------------------------------

# Patterns that indicate page-boundary noise (headers, footers, page numbers).
# These are generic — not hardcoded to any specific document.
_PAGE_NUM_RE = re.compile(r"^(?:[ivxlcdm]+|\d{1,4})$", re.IGNORECASE)
_COPYRIGHT_RE = re.compile(r"^©\s", re.IGNORECASE)
_HRULE_RE = re.compile(r"^-{3,}$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*(\|\s*[:*-]+\s*)+\|\s*$")
# Terminal punctuation that signals end of a complete sentence/block.
_TERMINAL_RE = re.compile(r"[.;:?!>\])`\"']\s*$")


def _detect_recurring_header(pages: list[str], threshold: float = 0.4) -> str | None:
    """Find the most common first non-empty line across pages.

    If a line appears as the first content line on more than *threshold*
    fraction of pages, it is considered a recurring page header.
    Returns the line text, or ``None`` if no recurring header is found.
    """
    first_lines: list[str] = []
    for page in pages:
        for line in page.splitlines():
            stripped = line.strip()
            if stripped:
                first_lines.append(stripped)
                break
    if not first_lines:
        return None

    counter = Counter(first_lines)
    most_common_line, count = counter.most_common(1)[0]
    if count / len(pages) >= threshold:
        return most_common_line
    return None


def _is_boundary_noise(line: str, recurring_header: str | None) -> bool:
    """Return True if *line* looks like page-boundary noise."""
    s = line.strip()
    if not s:
        return True
    if _HRULE_RE.match(s):
        return True
    if _PAGE_NUM_RE.match(s):
        return True
    if _COPYRIGHT_RE.match(s):
        return True
    return bool(recurring_header and s == recurring_header)


def _strip_leading_noise(text: str, recurring_header: str | None) -> str:
    """Remove page-boundary noise from the start of a page's text."""
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if _is_boundary_noise(line, recurring_header):
            start = i + 1
        else:
            break
    return "\n".join(lines[start:])


def _strip_trailing_noise(text: str, recurring_header: str | None) -> str:
    """Remove page-boundary noise from the end of a page's text."""
    lines = text.splitlines()
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if _is_boundary_noise(lines[i], recurring_header):
            end = i
        else:
            break
    return "\n".join(lines[:end])


def _last_content_line(text: str) -> str:
    """Return the last non-empty line of *text*."""
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _first_content_line(text: str) -> str:
    """Return the first non-empty line of *text*."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _is_split_table(page_a: str, page_b_clean: str) -> bool:
    """True if page_a ends mid-table and page_b_clean continues it."""
    last = _last_content_line(page_a)
    first = _first_content_line(page_b_clean)
    if not last or not first:
        return False
    # Both lines must look like table rows.
    return bool(_TABLE_ROW_RE.match(last) and _TABLE_ROW_RE.match(first))


def _is_split_paragraph(page_a: str, page_b_clean: str) -> bool:
    """True if page_a ends mid-sentence and page_b_clean continues it."""
    last = _last_content_line(page_a)
    first = _first_content_line(page_b_clean)
    if not last or not first:
        return False
    # Don't merge if the next page starts with a heading, table, or list.
    if first.startswith(("#", "|", "-", "*", ">", "```")):
        return False
    # Don't merge if current page ends with a complete sentence/block.
    if _TERMINAL_RE.search(last):
        return False
    # Next page should start with a lowercase letter (continuation).
    return first[0].islower()


def merge_cross_page_content(pages: list[str]) -> list[str]:
    """Merge content that spans page boundaries.

    Detects markdown tables and paragraphs that are split across pages
    and joins them, removing page-header/footer noise at the join
    points.  Pages that do not need merging are left untouched.

    Args:
        pages: List of cleaned per-page markdown strings.

    Returns:
        A (possibly shorter) list where split content has been merged.
    """
    if len(pages) <= 1:
        return list(pages)

    pages = list(pages)  # work on a mutable copy

    recurring_header = _detect_recurring_header(pages)
    if recurring_header:
        logger.debug("Detected recurring page header: %r", recurring_header)

    # ------------------------------------------------------------------
    # Phase 1: Pre-pass – strip boundary noise from every page so that
    # copyright lines, recurring headers, and standalone page numbers
    # are removed regardless of whether pages merge.  Skip the leading-
    # noise strip on page 0 to preserve the real title / cover content.
    # ------------------------------------------------------------------
    for idx in range(len(pages)):
        if idx > 0:
            pages[idx] = _strip_leading_noise(pages[idx], recurring_header)
        pages[idx] = _strip_trailing_noise(pages[idx], recurring_header)

    merged: list[str] = []
    i = 0
    while i < len(pages):
        current = pages[i]
        # Try to merge with subsequent pages.
        while i + 1 < len(pages):
            next_raw = pages[i + 1]
            next_clean = _strip_leading_noise(next_raw, recurring_header)

            # Phase 2 safety net: strip any trailing noise that crept
            # into *current* through a prior merge iteration so the
            # split-detection heuristics and joins stay clean.
            current = _strip_trailing_noise(current, recurring_header)

            if _is_split_table(current, next_clean):
                # If the continuation re-emits a table header + separator
                # row, strip the orphaned separator.  The typical pattern
                # at the top of the continuation page is:
                #   | First data row | ... |      ← keep
                #   | --- | --- |                 ← strip (orphaned separator)
                #   | Next data row | ... |       ← keep
                # Or just:
                #   | --- | --- |                 ← strip
                #   | Data row | ... |            ← keep
                next_lines = next_clean.splitlines()
                cleaned_lines: list[str] = []
                stripped_sep = False
                for nl in next_lines:
                    if not stripped_sep and _TABLE_SEP_RE.match(nl.strip()):
                        stripped_sep = True
                        continue  # skip the first separator row
                    cleaned_lines.append(nl)
                remainder = "\n".join(cleaned_lines)
                current = current.rstrip() + "\n" + remainder.lstrip("\n")
                logger.debug("Merged split table at page boundary %d→%d", i, i + 1)
                i += 1
            elif _is_split_paragraph(current, next_clean):
                # Join with a space so the sentence flows.
                current = current.rstrip() + " " + next_clean.lstrip()
                logger.debug("Merged split paragraph at page boundary %d→%d", i, i + 1)
                i += 1
            else:
                break

        merged.append(current)
        i += 1

    if len(merged) < len(pages):
        logger.info(
            "Cross-page merge: %d pages → %d entries (%d merges)",
            len(pages),
            len(merged),
            len(pages) - len(merged),
        )

    return merged


# ---------------------------------------------------------------------------
# Image reference restoration (post-processing)
# ---------------------------------------------------------------------------

# Matches image filenames like p36_img1.png, p7_img2.jpg
_IMG_FILENAME_RE = re.compile(r"p(\d+)_img(\d+)\.\w+")


def _get_images_for_page(images_dir: Path, page_num: int) -> list[str]:
    """Return sorted list of image filenames belonging to *page_num* (1-indexed)."""
    results = []
    for img_path in sorted(images_dir.iterdir()):
        m = _IMG_FILENAME_RE.match(img_path.name)
        if m and int(m.group(1)) == page_num:
            results.append(img_path.name)
    return results


def _find_referenced_images(text: str) -> set[str]:
    """Return set of image filenames referenced in *text* via markdown syntax."""
    # Match both ![alt](images/filename) and ![alt](./images/filename)
    refs = re.findall(r"!\[[^\]]*\]\((?:\./)?images/([^)]+)\)", text)
    return set(refs)


def _is_common_line(line: str) -> bool:
    """Return True if *line* is likely a repeating page header/footer/noise."""
    s = line.strip()
    if not s:
        return True
    # Very short lines are often page numbers or noise
    if len(s) < 10:
        return True
    # Common patterns: ISO doc numbers, copyright, page numbers, watermarks
    if re.match(r"^ISO\s*\d", s):
        return True
    if s.startswith("©"):
        return True
    if re.match(r"^\d{1,4}$", s):
        return True
    if "Uncontrolled copy" in s:
        return True
    return "ITS/T:" in s


def _find_insertion_point_from_cache(
    page_text: str,
    cache_page_text: str,
    image_filename: str,
) -> int | None:
    """Find where to insert *image_filename* in *page_text* using cache context.

    Looks for the image reference in the cache page, extracts surrounding
    context lines, and searches for those lines in the current page text
    to determine the correct insertion point.

    Strategy:
      1. Prefer context_after (figure captions) — insert before it.
      2. Fall back to context_before — insert after it.
      3. Skip common/short lines (page headers, copyright) as anchors.
      4. Require the anchor line to be sufficiently unique (>20 chars).

    Returns:
        Character offset in *page_text* where the image ref should be
        inserted, or None if placement cannot be determined.
    """
    # Find the image reference line in the cache
    img_ref_pattern = re.compile(
        r"!\[[^\]]*\]\((?:\./)?images/" + re.escape(image_filename) + r"\)"
    )
    cache_lines = cache_page_text.splitlines()

    img_line_idx = None
    for idx, line in enumerate(cache_lines):
        if img_ref_pattern.search(line):
            img_line_idx = idx
            break

    if img_line_idx is None:
        return None

    # Extract context: lines before and after the image reference in the cache.
    # Skip empty lines, image refs, horizontal rules, and common noise lines.
    def _is_good_anchor(line: str) -> bool:
        s = line.strip()
        if not s or s.startswith("![") or s.startswith("---"):
            return False
        if _is_common_line(s):
            return False
        # Must be long enough to be a useful anchor
        return len(s) > 15

    context_before: str | None = None
    for i in range(img_line_idx - 1, -1, -1):
        if _is_good_anchor(cache_lines[i]):
            context_before = cache_lines[i].strip()
            break

    context_after: str | None = None
    for i in range(img_line_idx + 1, len(cache_lines)):
        if _is_good_anchor(cache_lines[i]):
            context_after = cache_lines[i].strip()
            break

    page_lines = page_text.splitlines()

    def _find_line_in_page(target: str) -> int | None:
        """Find *target* in page_lines, return line index or None.

        Uses exact match first, then substring match for longer strings.
        Returns the LAST match if there are multiple (more likely to be
        the unique content rather than a repeated header).
        """
        best: int | None = None
        for line_idx, line in enumerate(page_lines):
            stripped = line.strip()
            if stripped == target or (len(target) > 25 and target[:40] in stripped):
                best = line_idx
        return best

    # Strategy 1: Find context_after line (e.g. figure caption), insert BEFORE it
    if context_after:
        line_idx = _find_line_in_page(context_after)
        if line_idx is not None:
            offset = sum(len(ln) + 1 for ln in page_lines[:line_idx])
            return offset

    # Strategy 2: Find context_before line, insert AFTER it
    if context_before:
        line_idx = _find_line_in_page(context_before)
        if line_idx is not None:
            offset = sum(len(ln) + 1 for ln in page_lines[: line_idx + 1])
            return offset

    return None


def restore_missing_image_refs(
    pages: list[str],
    images_dir: Path,
    cache_pages: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Restore image references that were lost during post-processing.

    Compares images on disk with references in each page's markdown.
    Missing references are re-inserted at their correct position (using
    the OCR cache as a guide) or appended to the end of the page.

    This function operates on the per-page list BEFORE ``combine_pages()``
    so that page-to-image mapping is unambiguous.

    Args:
        pages: Per-page markdown strings (after hybrid merge).
        images_dir: Path to the ``images/`` directory containing extracted
                    image files.
        cache_pages: Optional list of original OCR cache page strings
                     (pre-hybrid-merge). Used to determine correct
                     placement of restored references.

    Returns:
        Tuple of (updated_pages, list_of_restored_image_filenames).
    """
    if not images_dir.exists():
        return pages, []

    restored: list[str] = []
    updated_pages = list(pages)

    for page_idx in range(len(updated_pages)):
        page_num = page_idx + 1  # images use 1-indexed page numbers
        expected_images = _get_images_for_page(images_dir, page_num)

        if not expected_images:
            continue

        # Find which images are already referenced in this page
        existing_refs = _find_referenced_images(updated_pages[page_idx])
        missing = [img for img in expected_images if img not in existing_refs]

        if not missing:
            continue

        # Get the cache page for placement guidance
        cache_text = cache_pages[page_idx] if cache_pages and page_idx < len(cache_pages) else None

        for img_filename in missing:
            ref_str = f"![](images/{img_filename})"
            insertion_offset: int | None = None

            # Try to find the correct insertion point from cache
            if cache_text:
                insertion_offset = _find_insertion_point_from_cache(
                    updated_pages[page_idx],
                    cache_text,
                    img_filename,
                )

            if insertion_offset is not None:
                text = updated_pages[page_idx]
                updated_pages[page_idx] = (
                    text[:insertion_offset] + ref_str + "\n" + text[insertion_offset:]
                )
            else:
                # Fallback: append at the end of the page
                updated_pages[page_idx] = updated_pages[page_idx].rstrip() + "\n\n" + ref_str

            restored.append(img_filename)
            logger.debug(
                "Restored image ref: %s (page %d, %s)",
                img_filename,
                page_num,
                "positioned" if insertion_offset is not None else "appended",
            )

    if restored:
        logger.info(
            "Image ref restoration: restored %d missing reference(s): %s",
            len(restored),
            ", ".join(restored),
        )

    return updated_pages, restored


def restore_image_refs_in_file(
    output_dir: Path,
    cache_dir: Path | None = None,
) -> list[str]:
    """Standalone image reference restoration for an existing output folder.

    Reads the final ``.md`` file, checks for missing image references,
    restores them, and writes the updated file back.

    Args:
        output_dir: Path to the output folder (contains ``.md``, ``images/``,
                    and optionally ``ocr_cache/``).
        cache_dir: Optional explicit cache directory. If None, looks for
                   ``ocr_cache/`` within *output_dir*.

    Returns:
        List of restored image filenames.
    """
    import json

    images_dir = output_dir / "images"
    if not images_dir.exists():
        logger.info("No images/ directory found in %s — nothing to restore.", output_dir)
        return []

    # Find the markdown file
    md_files = list(output_dir.glob("*.md"))
    if not md_files:
        logger.warning("No .md file found in %s", output_dir)
        return []
    md_path = md_files[0]

    # Determine cache directory
    if cache_dir is None:
        cache_dir = output_dir / "ocr_cache"

    # Load cache pages if available
    cache_pages: list[str] | None = None
    meta_path = cache_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        total = metadata["total_pages"]
        pages_dir = cache_dir / "pages"
        cache_pages = []
        for i in range(total):
            p = pages_dir / f"{i + 1:04d}.md"
            cache_pages.append(p.read_text(encoding="utf-8") if p.exists() else "")

    # Read current markdown and split into pages
    markdown = md_path.read_text(encoding="utf-8")
    page_texts = markdown.split(PAGE_SEPARATOR)

    # If we have cache pages, use them; otherwise work with split pages
    # Note: cross-page merge may have reduced page count, so we can't
    # assume 1:1 mapping. Fall back to whole-file approach if mismatch.
    if cache_pages and len(page_texts) != len(cache_pages):
        # Page counts don't match (cross-page merge happened).
        # Use whole-file approach: check all images against the full markdown.
        all_images = sorted(images_dir.iterdir())
        existing_refs = _find_referenced_images(markdown)
        missing_images = [
            img
            for img in all_images
            if _IMG_FILENAME_RE.match(img.name) and img.name not in existing_refs
        ]

        if not missing_images:
            logger.info("All images are properly referenced — nothing to restore.")
            return []

        restored: list[str] = []
        for img_path in missing_images:
            m = _IMG_FILENAME_RE.match(img_path.name)
            if not m:
                continue
            page_num = int(m.group(1))
            page_idx = page_num - 1
            img_filename = img_path.name

            cache_text = (
                cache_pages[page_idx] if cache_pages and page_idx < len(cache_pages) else None
            )

            ref_str = f"![](images/{img_filename})"
            insertion_offset: int | None = None

            if cache_text:
                insertion_offset = _find_insertion_point_from_cache(
                    markdown, cache_text, img_filename
                )

            if insertion_offset is not None:
                markdown = (
                    markdown[:insertion_offset] + ref_str + "\n" + markdown[insertion_offset:]
                )
            else:
                # Fallback: find a unique, meaningful line from the cache page
                # in the markdown. Use it as an anchor for placement.
                appended = False
                if cache_text:
                    # Collect candidate anchor lines near the image reference
                    # in the cache page (prioritize lines AFTER the image, e.g. captions)
                    cache_lines_list = cache_text.splitlines()
                    # Find image line in cache
                    img_pat = re.compile(
                        r"!\[[^\]]*\]\((?:\./)?images/" + re.escape(img_filename) + r"\)"
                    )
                    img_idx_in_cache = None
                    for ci, cl in enumerate(cache_lines_list):
                        if img_pat.search(cl):
                            img_idx_in_cache = ci
                            break

                    # Build ordered list of candidate anchors: after image first,
                    # then before image
                    candidates: list[tuple[str, str]] = []  # (line, "before"|"after")
                    if img_idx_in_cache is not None:
                        for ci in range(img_idx_in_cache + 1, len(cache_lines_list)):
                            s = cache_lines_list[ci].strip()
                            if s and not _is_common_line(s) and not s.startswith("!["):
                                candidates.append((s, "before"))  # insert before this
                                break
                        for ci in range(img_idx_in_cache - 1, -1, -1):
                            s = cache_lines_list[ci].strip()
                            if s and not _is_common_line(s) and not s.startswith("!["):
                                candidates.append((s, "after"))  # insert after this
                                break

                    for anchor_text, position in candidates:
                        # Find the LAST occurrence of this anchor in the markdown
                        # (more likely to be the unique content instance)
                        pos = markdown.rfind(anchor_text)
                        if pos == -1 and len(anchor_text) > 30:
                            # Try prefix match for long lines
                            pos = markdown.rfind(anchor_text[:40])
                        if pos == -1:
                            continue

                        if position == "before":
                            # Insert before this line
                            # Find the start of the line containing pos
                            line_start = markdown.rfind("\n", 0, pos)
                            insert_at = line_start + 1 if line_start != -1 else 0
                        else:
                            # Insert after this line
                            line_end = markdown.find("\n", pos)
                            insert_at = (line_end + 1) if line_end != -1 else len(markdown)

                        markdown = markdown[:insert_at] + ref_str + "\n" + markdown[insert_at:]
                        appended = True
                        break

                if not appended:
                    # Last resort: append at end of document
                    markdown = markdown.rstrip() + "\n\n" + ref_str + "\n"

            restored.append(img_filename)
            logger.debug(
                "Restored image ref: %s (page %d, %s)",
                img_filename,
                page_num,
                "positioned" if insertion_offset is not None else "heuristic",
            )

        if restored:
            md_path.write_text(markdown, encoding="utf-8")
            logger.info(
                "Image ref restoration: restored %d missing reference(s) in %s",
                len(restored),
                md_path,
            )

        return restored

    else:
        # Page counts match — use the per-page approach
        updated_pages, restored = restore_missing_image_refs(page_texts, images_dir, cache_pages)

        if restored:
            new_markdown = PAGE_SEPARATOR.join(updated_pages)
            md_path.write_text(new_markdown, encoding="utf-8")
            logger.info(
                "Image ref restoration: restored %d missing reference(s) in %s",
                len(restored),
                md_path,
            )

        return restored
