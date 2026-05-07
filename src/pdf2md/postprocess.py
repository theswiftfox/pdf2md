# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""Post-processing for raw OCR model output."""

from __future__ import annotations

import logging
import re
from collections import Counter

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
