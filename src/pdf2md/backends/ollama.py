# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""Ollama backend for OCR inference via a local Ollama server.

Uses the Ollama REST API to send images to a locally-running vision model
(e.g. ``deepseek-ocr``).  This avoids loading the model in-process and
lets Ollama handle quantization, memory management, and GPU scheduling.

All inference uses ``stream: true`` so that Ollama can free the KV cache
for each request as soon as generation completes, rather than buffering
the full response.  This keeps server-side RAM bounded.

Requires:
    - Ollama installed and running (https://ollama.com)
    - The target model pulled: ``ollama pull deepseek-ocr``
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from pdf2md.backends.llama_cpp import HYBRID_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "deepseek-ocr"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Prompt that mirrors the DeepSeek-OCR "document" style.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise OCR engine. Output only the markdown content of the "
    "document shown in the image. Do not add any commentary. "
    "Do only wrap output in code fences if you are absolutely certain it is a block of code. "
    "NEVER place code fences (```) inside markdown pipe table cells — if a table cell "
    "needs to contain code, use an HTML <table> with <pre><code> inside <td> instead. "
    "Do NOT output page headers, footers, standalone page numbers, or repeated "
    "document identifiers. Do NOT add commentary like 'no visible content'."
)
DEFAULT_USER_PROMPT = "Convert the document to markdown."

# Maximum dimension (pixels) for the longest side of a page image sent to
# Ollama.  Larger images are downscaled proportionally.  1536 px is large
# enough for good OCR quality on A4/Letter pages while keeping the vision
# encoder's memory footprint manageable across many sequential requests.
_MAX_IMAGE_DIM = 1536

# Per-request timeout for the initial connection (seconds).
_CONNECT_TIMEOUT = 30

# Per-chunk read timeout -- if no new data arrives within this many seconds
# we consider the request hung.  Ollama streams tokens, so data should
# arrive continuously.
_READ_TIMEOUT = 120

# Number of retries on transient failures (timeout, 5xx, connection reset).
_MAX_RETRIES = 2

# Seconds to wait between retries.
_RETRY_DELAY = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap_image_size(image: Image.Image, max_dim: int = _MAX_IMAGE_DIM) -> Image.Image:
    """Downscale *image* so its longest side is at most *max_dim* pixels.

    Returns the original image unchanged if it's already within bounds.
    Uses LANCZOS resampling for quality.
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.debug("Capping image from %dx%d to %dx%d", w, h, new_w, new_h)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _is_retryable(exc: Exception) -> bool:
    """Return True if the error is transient and worth retrying."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, HTTPError) and exc.code >= 500:
        return True
    if isinstance(exc, URLError):
        reason = str(getattr(exc, "reason", exc))
        if any(s in reason.lower() for s in ("reset", "broken", "timed out", "timeout")):
            return True
    return False


def _ollama_post(
    host: str,
    endpoint: str,
    payload: dict,
    timeout: float = _CONNECT_TIMEOUT,
) -> dict:
    """Send a JSON POST and return the full response (for non-streaming calls)."""
    url = f"{host.rstrip('/')}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        result: dict = json.loads(resp.read().decode("utf-8"))
        return result


def _ollama_generate_stream(
    host: str,
    payload: dict,
    timeout: float = _READ_TIMEOUT,
) -> str:
    """Send a streaming generate request, consume tokens, return full text.

    Uses ``stream: true`` so Ollama sends one JSON line per token.  We
    read each line as it arrives, which:
      - keeps server-side memory bounded (KV cache freed on completion)
      - lets us detect hangs quickly (no data = timeout)
      - avoids buffering a huge response in Ollama before sending
    """
    payload = {**payload, "stream": True}
    url = f"{host.rstrip('/')}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    chunks: list[str] = []
    with urlopen(req, timeout=timeout) as resp:
        # Read line by line -- each line is a JSON object with a "response" field
        while True:
            line = resp.readline()
            if not line:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = obj.get("response", "")
            if token:
                chunks.append(token)
            # Ollama sets "done": true on the final chunk
            if obj.get("done", False):
                break

    return "".join(chunks).strip()


def _generate_with_retry(
    host: str,
    payload: dict,
    timeout: float = _READ_TIMEOUT,
    max_retries: int = _MAX_RETRIES,
    label: str = "",
) -> str:
    """Streaming generate with automatic retry on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            t0 = time.monotonic()
            result = _ollama_generate_stream(host, payload, timeout=timeout)
            elapsed = time.monotonic() - t0
            if label:
                logger.debug("%s completed in %.1fs (%d chars)", label, elapsed, len(result))
            return result
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt <= max_retries and _is_retryable(exc):
                logger.warning(
                    "%s attempt %d/%d failed (%s), retrying in %.0fs...",
                    label or "generate",
                    attempt,
                    max_retries + 1,
                    exc,
                    _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
            else:
                break

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class OllamaOCRBackend:
    """OCR backend that delegates to a local Ollama server.

    All inference uses streaming responses so Ollama can free server-side
    memory (KV cache) immediately after each request completes.

    When ``batch_size > 1`` in the converter, this backend fires
    concurrent HTTP requests.  Set ``OLLAMA_NUM_PARALLEL`` on the
    Ollama server to match, otherwise requests just queue up.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        num_predict: int = 8192,
        num_ctx: int = 8192,
        cleanup_model: str | None = None,
        use_system_prompt: bool = True,
        **_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.host = host
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.cleanup_model = cleanup_model
        self.use_system_prompt = use_system_prompt
        self._available = False
        self._interrupted = False
        # Cross-page context: set by the converter after each page so the
        # next request can see how the previous page ended (helps with
        # tables, lists, and sections that span page boundaries).
        self.page_context: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._available

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Verify that Ollama is reachable and the model is pulled."""
        if self._available:
            return

        # Check server is running (GET endpoint)
        try:
            url = f"{self.host.rstrip('/')}/api/tags"
            urlopen(Request(url), timeout=5)
        except URLError as exc:
            raise ConnectionError(
                f"Ollama server not reachable at {self.host}. Start it with: ollama serve"
            ) from exc

        # Check model is available
        try:
            _ollama_post(self.host, "/api/show", {"name": self.model_name}, timeout=10)
        except Exception as exc:
            raise RuntimeError(
                f"Model {self.model_name!r} not found in Ollama. "
                f"Pull it with: ollama pull {self.model_name}"
            ) from exc

        # Warm up: preload model onto GPU with the correct KV-cache size.
        # IMPORTANT: include num_ctx here so Ollama allocates the KV cache
        # at our intended size from the start.  Without this, Ollama uses
        # its VRAM-based default (often 32K+), wasting several GB of VRAM.
        logger.info("Warming up Ollama model %s (num_ctx=%d)...", self.model_name, self.num_ctx)
        _ollama_post(
            self.host,
            "/api/generate",
            {
                "model": self.model_name,
                "keep_alive": "60m",
                "prompt": "",
                "stream": False,
                "options": {"num_ctx": self.num_ctx},
            },
            timeout=120,
        )

        self._available = True
        self._interrupted = False
        logger.info(
            "Ollama backend ready (model=%s, num_ctx=%d, num_predict=%d)",
            self.model_name,
            self.num_ctx,
            self.num_predict,
        )

    def unload(self) -> None:
        """No-op -- Ollama manages model lifetime."""
        self._available = False

    # -- single-page inference -----------------------------------------------

    def process_image(self, image: Image.Image) -> str:
        """Run OCR on a single page via Ollama (streaming)."""
        if not self._available:
            raise RuntimeError("Backend not loaded. Call .load() first.")
        img = _cap_image_size(image.convert("RGB"))
        b64 = _image_to_base64(img)
        return self._call_ollama(b64, label="page")

    # -- hybrid merge (raw PDF text + OCR markdown) --------------------------

    def hybrid_merge_text(
        self,
        raw_text: str,
        ocr_markdown: str,
        prev_page_tail: str | None = None,
    ) -> str:
        """Merge raw PDF text (ground truth) with OCR markdown (structure).

        Sends both representations to the merge LLM which combines
        perfect text accuracy from the PDF with markdown formatting
        from the OCR output.

        When *prev_page_tail* is provided (last ~15 lines of the previous
        page's merged output), it is included so the model can continue
        open formatting structures (tables, lists, code blocks) across
        page boundaries.

        Uses ``cleanup_model`` for routing if set.
        """
        if not self._available:
            raise RuntimeError("Backend not loaded. Call .load() first.")

        parts: list[str] = []
        if prev_page_tail:
            tail_stripped = prev_page_tail.rstrip()
            ends_with_table = (
                tail_stripped.endswith("</table>")
                or tail_stripped.endswith("|")
                or any(
                    line.strip().startswith("|") or "<td>" in line
                    for line in tail_stripped.splitlines()[-3:]
                )
            )

            toc_line_re = re.compile(r"^\s*\d+(?:\.\d+)*\s+\S.+\s+\d{1,3}\s*$")
            raw_lines = raw_text.strip().splitlines()
            toc_like_lines = sum(1 for line in raw_lines if toc_line_re.match(line))
            looks_like_toc = toc_like_lines >= 5

            if ends_with_table and looks_like_toc:
                parts.append(
                    "PREVIOUS PAGE TAIL (for formatting continuity only — "
                    "do NOT repeat this content):\n"
                    '"""\n'
                    f"{prev_page_tail}\n"
                    '"""\n'
                )
                parts.append(
                    "⚠️ IMPORTANT: The previous page ended with a TABLE and "
                    "this page continues with table-of-contents entries. "
                    "You MUST output them as pipe table rows "
                    "(| section | title | page |), NOT as markdown headings. "
                    "The OCR model wrongly converted these table rows into "
                    "headings — fix them back into table rows.\n"
                )
        parts.append(
            "PLAIN TEXT (ground truth):\n"
            '"""\n'
            f"{raw_text}\n"
            '"""\n\n'
            "OCR MARKDOWN (has formatting + images):\n"
            '"""\n'
            f"{ocr_markdown}\n"
            '"""\n'
        )
        user_content = "\n".join(parts)

        # Use the cleanup_model for the merge if specified, otherwise
        # fall back to the primary OCR model.
        model = self.cleanup_model or self.model_name

        payload = {
            "model": model,
            "prompt": user_content,
            "system": HYBRID_SYSTEM_PROMPT,
            "options": {
                "temperature": 0,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        return _generate_with_retry(
            self.host,
            payload,
            timeout=_READ_TIMEOUT,
            label="hybrid",
        )

    # -- batching interface --------------------------------------------------

    def preprocess(self, image: Image.Image) -> str:
        """Preprocess: cap dimensions + base64-encode (CPU work, thread-safe)."""
        img = _cap_image_size(image.convert("RGB"))
        return _image_to_base64(img)

    def generate_from_prepared(self, batch: list[str]) -> list[str]:
        """Fire Ollama requests for a batch of base64 images.

        For batch_size=1 (the recommended default), this is a simple
        sequential streaming call.  For batch_size>1, requests run
        concurrently in threads.
        """
        if len(batch) == 1:
            return [self._call_ollama(batch[0], label="page")]
        return self._call_ollama_concurrent(batch)

    def process_batch(self, images: list[Image.Image]) -> list[str]:
        """Convenience: preprocess + generate in one call."""
        if not images:
            return []
        prepared = [self.preprocess(img) for img in images]
        return self.generate_from_prepared(prepared)

    # -- internal helpers ----------------------------------------------------

    def _make_payload(self, image_b64: str) -> dict:
        """Build the Ollama /api/generate payload (without stream key).

        ``num_ctx`` caps the KV-cache allocation so Ollama keeps the
        model fully on GPU instead of spilling layers to system RAM.
        8192 is plenty for single-page OCR (≈600 input + up to 7500
        output tokens).

        When ``use_system_prompt`` is False (bare-image mode), the system
        prompt is omitted — suitable for models like LightOnOCR that
        expect a bare image without text instructions.

        When ``page_context`` is set, the previous page's tail is
        appended to the prompt so the model can maintain formatting
        continuity across pages.
        """
        if self.use_system_prompt:
            if self.page_context:
                prompt = (
                    f"{self.user_prompt}\n\n"
                    "IMPORTANT: Output ONLY what is visible in the image. "
                    "Do NOT repeat or continue the text below. "
                    "Use it ONLY as a formatting reference for style consistency "
                    "(e.g. table format, heading levels, list style):\n"
                    f"{self.page_context}"
                )
            else:
                prompt = self.user_prompt
        else:
            # Bare-image mode: minimal or no text prompt.
            prompt = ""

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "options": {
                "temperature": 0,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
                "repeat_penalty": 1.15,
                "repeat_last_n": 128,
            },
        }

        if self.use_system_prompt:
            payload["system"] = self.system_prompt

        return payload

    def _call_ollama(self, image_b64: str, label: str = "") -> str:
        """Send a single image to Ollama (streaming) and return the text."""
        if self._interrupted:
            raise KeyboardInterrupt
        payload = self._make_payload(image_b64)
        return _generate_with_retry(
            self.host,
            payload,
            timeout=_READ_TIMEOUT,
            label=label,
        )

    def _call_ollama_concurrent(self, batch: list[str]) -> list[str]:
        """Send multiple images concurrently using daemon threads.

        Uses short-polling on futures so KeyboardInterrupt is handled
        promptly in the main thread.
        """
        logger.debug("Sending %d concurrent requests to Ollama", len(batch))
        t0 = time.monotonic()

        results: list[str | None] = [None] * len(batch)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(batch))
        try:
            futures = {
                pool.submit(self._call_ollama, b64, label=f"batch[{i}]"): i
                for i, b64 in enumerate(batch)
            }

            # Poll with short timeout so main thread can handle signals
            done: set = set()
            while len(done) < len(futures):
                newly_done, _not_done = concurrent.futures.wait(
                    futures.keys() - done,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                done |= newly_done

                for future in newly_done:
                    idx = futures[future]
                    try:
                        results[idx] = future.result(timeout=0)
                    except Exception as exc:
                        logger.error("Ollama batch[%d] failed: %s", idx, exc)
                        results[idx] = f"<!-- OCR ERROR: {exc} -->"

        except KeyboardInterrupt:
            self._interrupted = True
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=False)

        elapsed = time.monotonic() - t0
        logger.debug(
            "Batch of %d completed in %.1fs (%.1fs/page)",
            len(batch),
            elapsed,
            elapsed / len(batch),
        )
        return results  # type: ignore[return-value]

    def __repr__(self) -> str:
        status = "connected" if self._available else "not connected"
        return f"OllamaOCRBackend(model={self.model_name!r}, host={self.host!r}, {status})"
