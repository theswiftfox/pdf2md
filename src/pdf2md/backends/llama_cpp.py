# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""llama.cpp server backend for OCR inference.

Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint exposed by
``llama-server`` (or ``llama-cpp-python[server]``).

Requires only the Python stdlib + Pillow (no extra deps).
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_CPP_HOST = "http://localhost:8080"

# Reuse the same OCR prompts as the Ollama backend.
DEFAULT_SYSTEM_PROMPT_GENERIC = (
    "You are a precise OCR engine. Output only the markdown content of the "
    "document shown in the image. Do not add any commentary. "
    "Do only wrap output in code fences if you are absolutely certain it is a block of code. "
    "NEVER place code fences (```) inside markdown pipe table cells — if a table cell "
    "needs to contain code, use an HTML <table> with <pre><code> inside <td> instead. "
    "Do NOT output page headers, footers, standalone page numbers, or repeated "
    "document identifiers. Do NOT add commentary like 'no visible content'."
)

DEFAULT_SYSTEM_PROMPT_SPEC = (
    "You are a precise OCR engine specialized in technical specification documents. "
    "The document is containing formal normative "
    "language, numbered sections, cross-references, definition tables, JSON schemas, "
    "HTTP request/response examples, and technical figures.\n\n"
    "Output the content as clean markdown, preserving:\n"
    "- Section numbering and heading hierarchy exactly as shown\n"
    "- Tables as markdown tables with proper column alignment\n"
    "- Code examples (JSON, HTTP, JavaScript) in fenced code blocks with correct "
    "language tags (```json, ```http, ```javascript — never ```plaintext for these)\n"
    "- Formal terms (shall, may, normative, informative) exactly as written\n\n"
    "CRITICAL TABLE RULES:\n"
    "- NEVER place code fences (```) inside markdown pipe table cells. "
    "Pipe tables cannot contain block-level elements.\n"
    "- If a table cell needs to contain code or JSON, use an HTML <table> with "
    "<pre><code> blocks inside <td> elements instead of a pipe table.\n"
    "- Ensure every data row has the same number of | delimiters as the header.\n\n"
    "IGNORE page artifacts:\n"
    "- Do NOT output page headers, footers, or standalone page numbers.\n"
    "- Do NOT output repeated document identifiers like 'ABC XXXXX:YYYY(en)' "
    "that appear as running headers.\n"
    "- Do NOT output lines like 'DRAFT International Standard' that are page headers.\n"
    "- Do NOT add commentary like 'no visible content' or 'blank page'.\n\n"
    "For figures and diagrams, output a brief description like [Figure X: description] "
    "rather than attempting to transcribe graphical content as text.\n\n"
    "Do not add any commentary or explanation beyond what appears in the document."
)

DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_SPEC
DEFAULT_USER_PROMPT = "Convert the document to markdown."


# -- Hybrid merge prompt (raw PDF text + OCR) --------------------------------

HYBRID_SYSTEM_PROMPT = (
    "You are a document reconstruction tool. You receive two representations "
    "of the same document page:\n\n"
    "1. PLAIN TEXT — extracted directly from the PDF. It is character-perfect "
    "but has no formatting.\n"
    "2. OCR MARKDOWN — produced by a vision model. It has markdown formatting "
    "(headings, tables, code blocks, bold, etc.) and image references, but "
    "may contain text errors or noise.\n"
    "3. PREVIOUS PAGE TAIL (optional) — the last few lines of the already-"
    "merged output from the previous page. Use this to decide whether "
    "to continue an open structure. Do NOT repeat any text from it.\n\n"
    "Your job is to produce corrected markdown that:\n"
    "- Uses the PLAIN TEXT for word-level accuracy (it is the ground truth)\n"
    "- Uses the OCR MARKDOWN for structure: headings (#), bold (**), italic (*), "
    "tables (| or <table>), code blocks (```), lists, etc.\n"
    "- FORMATTING CONTINUITY: If PREVIOUS PAGE TAIL ends with a table "
    "(even a closed one), and the PLAIN TEXT on THIS page continues the "
    "same pattern (e.g. numbered section entries with page numbers like "
    "a table of contents), then output this page's content as table rows "
    "continuing the same format — NOT as headings. The OCR model does not "
    "know the table continues across pages, so it wrongly converts table-"
    "of-contents entries into headings. You must fix this.\n"
    "- HEADING LEVELS: When content is actual headings (not ToC entries), "
    "copy the exact number of # characters from the OCR MARKDOWN. "
    "Do NOT change heading depth.\n"
    "- CODE BLOCKS: Wrap every HTTP request/response example and every "
    "code snippet (JavaScript, JSON, etc.) in ``` fences with correct "
    "language tags (```json, ```http, ```javascript).\n"
    "- CODE IN TABLE CELLS: If a table cell needs to contain code or JSON, "
    "use an HTML <table> with <pre><code> inside <td> elements. "
    "NEVER place code fences (```) inside markdown pipe table cells.\n"
    "- Preserves ALL image references ![](images/...) from the OCR MARKDOWN\n"
    "- PAGE ARTIFACTS: Remove page headers, footers, copyright notices, "
    "standalone page numbers, and repeated document identifiers like "
    "'ISO/DIS ...(en)'\n"
    "- Does NOT add content that is not in the original text\n"
    "- Does NOT add commentary such as 'no visible content' or 'blank page'\n"
    "- Does NOT wrap the entire output in a code fence\n\n"
    "Output ONLY the corrected markdown. No commentary."
)


# Maximum dimension (pixels) for the longest side of a page image.
_MAX_IMAGE_DIM = 1536

# Timeouts (seconds).
_CONNECT_TIMEOUT = 30
_READ_TIMEOUT = 120

# Retry settings.
_MAX_RETRIES = 2
_RETRY_DELAY = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap_image_size(image: Image.Image, max_dim: int = _MAX_IMAGE_DIM) -> Image.Image:
    """Downscale *image* so its longest side is at most *max_dim* pixels."""
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
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    if isinstance(exc, HTTPError) and exc.code >= 500:
        return True
    if isinstance(exc, URLError):
        reason = str(getattr(exc, "reason", exc))
        if any(s in reason.lower() for s in ("reset", "broken", "timed out", "timeout")):
            return True
    return False


def _post_json(host: str, endpoint: str, payload: dict, timeout: float = _CONNECT_TIMEOUT) -> dict:
    """Send a JSON POST and return the parsed response."""
    url = f"{host.rstrip('/')}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        result: dict = json.loads(resp.read().decode("utf-8"))
        return result


def _chat_completions_stream(
    host: str,
    payload: dict,
    timeout: float = _READ_TIMEOUT,
) -> str:
    """Send a streaming chat completions request, return the full text.

    The ``/v1/chat/completions`` endpoint with ``stream: true`` returns
    Server-Sent Events (SSE).  Each line looks like::

        data: {"choices":[{"delta":{"content":"tok"}}]}

    The final line is ``data: [DONE]``.
    """
    payload = {**payload, "stream": True}
    url = f"{host.rstrip('/')}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    chunks: list[str] = []
    with urlopen(req, timeout=timeout) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            # SSE format: lines starting with "data: "
            if line.startswith("data: "):
                body = line[6:]
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        chunks.append(token)
            # Some llama.cpp builds omit the "data: " prefix
            elif line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        chunks.append(token)

    return "".join(chunks).strip()


def _generate_with_retry(
    host: str,
    payload: dict,
    timeout: float = _READ_TIMEOUT,
    max_retries: int = _MAX_RETRIES,
    label: str = "",
) -> str:
    """Streaming chat completion with automatic retry on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            t0 = time.monotonic()
            result = _chat_completions_stream(host, payload, timeout=timeout)
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


class LlamaCppOCRBackend:
    """OCR backend that delegates to a llama.cpp server.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint with
    multimodal message format (inline base64 images).

    llama-server must already be running with the desired model loaded.
    Context length is controlled by the server's ``-c`` flag, not
    per-request, so there is no ``num_ctx`` parameter here.
    """

    def __init__(
        self,
        host: str = DEFAULT_LLAMA_CPP_HOST,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        num_predict: int = 8192,
        ocr_model: str | None = None,
        cleanup_model: str | None = None,
        use_system_prompt: bool = True,
        **_kwargs: Any,
    ) -> None:
        self.host = host
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.num_predict = num_predict
        self.ocr_model = ocr_model
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
        """Verify that llama-server is reachable and healthy."""
        if self._available:
            return

        # llama-server exposes GET /health
        try:
            url = f"{self.host.rstrip('/')}/health"
            with urlopen(Request(url), timeout=_CONNECT_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                status = body.get("status", "unknown")
                if status not in ("ok", "no slot available"):
                    raise ConnectionError(f"llama-server reports unhealthy status: {status}")
        except (URLError, ConnectionError) as exc:
            raise ConnectionError(
                f"llama-server not reachable at {self.host}. "
                f"Start it with: llama-server -m <model.gguf> --host 0.0.0.0 --port 8080\n"
                f"  ({exc})"
            ) from exc

        self._available = True
        self._interrupted = False
        logger.info(
            "llama.cpp backend ready (host=%s, max_tokens=%d)",
            self.host,
            self.num_predict,
        )

    def unload(self) -> None:
        """No-op -- llama-server manages model lifetime."""
        self._available = False

    # -- single-page inference -----------------------------------------------

    def process_image(self, image: Image.Image) -> str:
        """Run OCR on a single page via llama-server (streaming)."""
        if not self._available:
            raise RuntimeError("Backend not loaded. Call .load() first.")
        img = _cap_image_size(image.convert("RGB"))
        b64 = _image_to_base64(img)
        return self._call_server(b64, label="page")

    # -- hybrid merge (raw PDF text + OCR markdown) ----------------------------

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

        Uses ``cleanup_model`` for routing (same text-only model slot).
        """
        if not self._available:
            raise RuntimeError("Backend not loaded. Call .load() first.")

        parts: list[str] = []
        if prev_page_tail:
            # Detect if previous page ended with a table AND the current
            # page looks like a ToC continuation (section numbers + page
            # numbers pattern in the raw text).
            import re

            tail_stripped = prev_page_tail.rstrip()
            ends_with_table = (
                tail_stripped.endswith("</table>")
                or tail_stripped.endswith("|")
                or any(
                    line.strip().startswith("|") or "<td>" in line
                    for line in tail_stripped.splitlines()[-3:]
                )
            )

            # Check if the current page's raw text looks like ToC entries:
            # lines matching "X.Y.Z  Title  NNN" pattern (section + page no).
            toc_line_re = re.compile(r"^\s*\d+(?:\.\d+)*\s+\S.+\s+\d{1,3}\s*$")
            raw_lines = raw_text.strip().splitlines()
            toc_like_lines = sum(1 for line in raw_lines if toc_line_re.match(line))
            looks_like_toc = toc_like_lines >= 5

            # Only include context when it's a ToC continuation — avoids
            # phi-4 repeating tail content on normal pages.
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

        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": HYBRID_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.num_predict,
            "temperature": 0,
        }
        if self.cleanup_model:
            payload["model"] = self.cleanup_model
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
        """Fire requests for a batch of base64 images.

        For batch_size=1 this is a simple sequential call.
        For batch_size>1, requests run concurrently.
        """
        if len(batch) == 1:
            return [self._call_server(batch[0], label="page")]
        return self._call_server_concurrent(batch)

    def process_batch(self, images: list[Image.Image]) -> list[str]:
        """Convenience: preprocess + generate in one call."""
        if not images:
            return []
        prepared = [self.preprocess(img) for img in images]
        return self.generate_from_prepared(prepared)

    # -- internal helpers ----------------------------------------------------

    def _make_payload(self, image_b64: str) -> dict:
        """Build the /v1/chat/completions payload with an inline image.

        Uses the OpenAI multimodal message format.  When ``use_system_prompt``
        is False, the payload contains only the image (no system message, no
        user text) — required for models like LightOnOCR that expect a bare
        image input.

        When ``page_context`` is set and system prompts are enabled, the
        previous page's tail is included in the user message so the model
        can maintain formatting continuity across pages.
        """

        if not self.use_system_prompt:
            # Bare-image mode: no system message.
            # If a custom user_prompt is set (e.g. "OCR:" for PaddleOCR-VL),
            # include it as text alongside the image.
            content: list[dict[str, Any]] = []
            if self.user_prompt != DEFAULT_USER_PROMPT:
                content.append({"type": "text", "text": self.user_prompt})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            )
            payload: dict[str, Any] = {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                "max_tokens": self.num_predict,
                "temperature": 0.2,
                "top_p": 0.9,
            }
            if self.ocr_model:
                payload["model"] = self.ocr_model
            return payload

        # Full prompt mode (default): system message + user text + image.
        # Build the user prompt, optionally with previous-page context.
        if self.page_context:
            prompt_text = (
                f"{self.user_prompt}\n\n"
                "IMPORTANT: Output ONLY what is visible in the image. "
                "Do NOT repeat or continue the text below. "
                "Use it ONLY as a formatting reference for style consistency "
                "(e.g. table format, heading levels, list style):\n"
                f"{self.page_context}"
            )
        else:
            prompt_text = self.user_prompt

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": self.num_predict,
            "temperature": 0,
        }
        if self.ocr_model:
            payload["model"] = self.ocr_model
        return payload

    def _call_server(self, image_b64: str, label: str = "") -> str:
        """Send a single image to llama-server (streaming) and return the text."""
        if self._interrupted:
            raise KeyboardInterrupt
        payload = self._make_payload(image_b64)
        return _generate_with_retry(
            self.host,
            payload,
            timeout=_READ_TIMEOUT,
            label=label,
        )

    def _call_server_concurrent(self, batch: list[str]) -> list[str]:
        """Send multiple images concurrently using daemon threads."""
        logger.debug("Sending %d concurrent requests to llama-server", len(batch))
        t0 = time.monotonic()

        results: list[str | None] = [None] * len(batch)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(batch))
        try:
            futures = {
                pool.submit(self._call_server, b64, label=f"batch[{i}]"): i
                for i, b64 in enumerate(batch)
            }

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
                        logger.error("llama-server batch[%d] failed: %s", idx, exc)
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
        return f"LlamaCppOCRBackend(host={self.host!r}, {status})"
