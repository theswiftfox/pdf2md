# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""Abstract interface for OCR backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class OCRBackend(Protocol):
    """Protocol that all OCR backends must satisfy.

    Lifecycle:
        1. __init__(**config)  - configure, no heavy work
        2. load()              - download/load model into memory
        3. process_image(...)  - called once per page (may be called many times)
        4. unload()            - free GPU/memory

    Optional high-throughput path (for backends that support batching):
        - preprocess(image) -> PreparedInput   (CPU, thread-safe)
        - generate_from_prepared([...])        (GPU)
        - process_batch([images])              (convenience wrapper)
    """

    def load(self) -> None:
        """Load the model into memory. Idempotent - safe to call multiple times."""
        ...

    def process_image(self, image: Image.Image) -> str:
        """Run OCR on a single page image and return raw text/markdown.

        Args:
            image: RGB PIL Image of a single document page.

        Returns:
            Raw OCR output string (may contain model-specific tokens that
            need post-processing).
        """
        ...

    def unload(self) -> None:
        """Release model resources and free GPU memory."""
        ...

    # -- optional batching interface ----------------------------------------
    # Backends that do not implement these will fall back to single-page
    # processing in the converter.

    def preprocess(self, image: Image.Image) -> Any:
        """Preprocess an image into backend-specific tensors (CPU only).

        Returns an opaque object that ``generate_from_prepared()`` accepts.
        """
        ...

    def generate_from_prepared(self, batch: list[Any]) -> list[str]:
        """Run inference on a list of preprocessed inputs.

        Args:
            batch: List of objects returned by ``preprocess()``.

        Returns:
            List of raw OCR text strings, one per input.
        """
        ...

    def process_batch(self, images: list[Image.Image]) -> list[str]:
        """Convenience: preprocess + generate in one call.

        Args:
            images: List of RGB PIL Images.

        Returns:
            List of raw OCR text strings, one per input image.
        """
        ...

    # -- hybrid merge interface ---------------------------------------------

    def hybrid_merge_text(
        self,
        raw_text: str,
        ocr_markdown: str,
        prev_page_tail: str | None = None,
    ) -> str:
        """Merge raw PDF text with OCR markdown using a text LLM.

        Args:
            raw_text: Plain text extracted directly from the PDF (ground truth).
            ocr_markdown: OCR output with markdown formatting.
            prev_page_tail: Last few lines of the previous page's output
                            for formatting continuity.

        Returns:
            Corrected markdown combining accuracy from raw_text with
            structure from ocr_markdown.
        """
        ...
