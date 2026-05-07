# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""Backend registry for OCR engines."""

from __future__ import annotations

import importlib
from typing import Any

from pdf2md.backends.base import OCRBackend

# Lazy-loaded backend registry: name -> "module.path:ClassName"
_BACKENDS: dict[str, str] = {
    "ollama": "pdf2md.backends.ollama:OllamaOCRBackend",
    "llama-cpp": "pdf2md.backends.llama_cpp:LlamaCppOCRBackend",
}


def create_backend(name: str, **kwargs: Any) -> OCRBackend:
    """Create an OCR backend instance by name.

    Args:
        name: Registered backend name (e.g. "ollama").
        **kwargs: Passed to the backend constructor.

    Returns:
        An initialized (but not yet loaded) backend instance.

    Raises:
        ValueError: If the backend name is not registered.
        ImportError: If the backend's dependencies are missing.
    """
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(f"Unknown backend: {name!r}. Available: {available}")

    module_path, class_name = _BACKENDS[name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance: OCRBackend = cls(**kwargs)
    return instance


def available_backends() -> list[str]:
    """Return list of registered backend names."""
    return sorted(_BACKENDS.keys())


__all__ = ["OCRBackend", "create_backend", "available_backends"]
