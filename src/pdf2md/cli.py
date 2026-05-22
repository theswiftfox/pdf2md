# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Elena Gantner

"""CLI entry point for pdf2md."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pdf2md import __version__
from pdf2md.backends import create_backend
from pdf2md.backends.llama_cpp import DEFAULT_LLAMA_CPP_HOST
from pdf2md.backends.ollama import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL
from pdf2md.converter import Converter, ProgressEvent, resolve_output_path
from pdf2md.postprocess import restore_image_refs_in_file

console = Console()

# Default hosts per backend (used when --host is not given).
_DEFAULT_HOSTS = {
    "ollama": DEFAULT_OLLAMA_HOST,
    "llama-cpp": DEFAULT_LLAMA_CPP_HOST,
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False, default=None)
@click.option(
    "-b",
    "--backend",
    type=click.Choice(["ollama", "llama-cpp"], case_sensitive=False),
    default="ollama",
    show_default=True,
    help="Inference backend.",
)
@click.option(
    "-m",
    "--model",
    default=DEFAULT_OLLAMA_MODEL,
    show_default=True,
    help="Model name/alias. Ollama: model to pull. "
    "llama-cpp: routes OCR to this model in the dynamic router.",
)
@click.option(
    "--host",
    "host",
    default=None,
    help="Server URL. Defaults to localhost:11434 (Ollama) or localhost:8080 (llama-cpp).",
)
@click.option(
    "--dpi",
    type=int,
    default=144,
    show_default=True,
    help="PDF rendering DPI.",
)
@click.option(
    "--num-ctx",
    type=int,
    default=8192,
    show_default=True,
    help="Context window size (Ollama backend only).",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Concurrent requests per batch. Try 2-4.",
)
@click.option(
    "--merge-pages/--no-merge-pages",
    default=False,
    show_default=True,
    help="Merge tables/paragraphs split across page boundaries.",
)
@click.option(
    "--cleanup-model",
    default=None,
    help="Model alias for the hybrid merge pass (e.g. 'phi-4'). Implies --hybrid.",
)
@click.option(
    "--system-prompt/--no-system-prompt",
    default=True,
    show_default=True,
    help="Send system+user text prompts with the image. "
    "Disable for models like LightOnOCR that expect a bare image.",
)
@click.option(
    "--hybrid/--no-hybrid",
    default=False,
    show_default=True,
    help="Correct OCR text using raw PDF text extraction + merge LLM. Requires --cleanup-model.",
)
@click.option(
    "--merge-only",
    is_flag=True,
    default=False,
    show_default=True,
    help="Skip OCR; re-run post-processing (hybrid/merge) from a cached OCR output folder.",
)
@click.option(
    "--fix-image-refs",
    is_flag=True,
    default=False,
    show_default=True,
    help="Restore missing image references in an existing output folder. "
    "Scans images/ for unreferenced files and inserts them at correct positions.",
)
@click.version_option(__version__, prog_name="pdf2md")
def main(
    input_path: str,
    output_path: str | None,
    backend: str,
    model: str,
    host: str | None,
    dpi: int,
    num_ctx: int,
    batch_size: int,
    merge_pages: bool,
    cleanup_model: str | None,
    system_prompt: bool,
    hybrid: bool,
    merge_only: bool,
    fix_image_refs: bool,
) -> None:
    """Convert PDF files to Markdown using vision-language OCR.

    Supports two inference backends:

    \b
      ollama     - Ollama server (default). Requires `ollama serve` + model pulled.
      llama-cpp  - llama.cpp server. Requires `llama-server -m <model.gguf>`.

    INPUT_PATH is a PDF file or a directory of PDFs (processed recursively).

    OUTPUT_PATH is optional. For a single PDF it defaults to a folder named
    after the file (e.g. doc.pdf -> doc/doc.md). For a directory, output
    folders are placed next to each PDF (e.g. papers/report.pdf ->
    papers/report/report.md). Pass an explicit OUTPUT_PATH to collect all
    results into a separate tree instead.

    Each PDF produces a folder containing the .md file and an images/
    subdirectory with any figures extracted from the PDF.

    \b
    With --merge-only, INPUT_PATH is a previously generated output folder
    (or a parent directory of output folders) containing ocr_cache/. OCR
    is skipped and only post-processing (hybrid merge, cross-page merge)
    is re-run. This is useful for trying different merge models without
    re-running the expensive OCR step.

    \b
    Examples:
        pdf2md document.pdf                                         # Ollama default
        pdf2md document.pdf -b llama-cpp --host http://host:8080    # llama.cpp
        pdf2md document.pdf --host http://host:11434       # remote Ollama
        pdf2md ./papers/                                            # batch, output next to each PDF
        pdf2md ./papers/ ./papers_md/                               # batch, separate output tree
        pdf2md --merge-only ./doc/ --hybrid --cleanup-model phi-4   # re-merge with different model
        pdf2md --fix-image-refs ./doc/                               # restore missing image refs
    """
    # -- Quick-exit mode: --fix-image-refs -----------------------------------
    if fix_image_refs:
        in_path = Path(input_path)
        console.print(f"[bold]pdf2md[/bold] v{__version__}")
        console.print("  Mode:       [bold cyan]fix-image-refs[/bold cyan]")
        console.print(f"  Input:      {in_path}")
        console.print()

        # Determine which directories to process
        dirs_to_process: list[Path] = []
        cache_marker = in_path / "ocr_cache" / "metadata.json"
        if cache_marker.exists():
            dirs_to_process = [in_path]
        else:
            # Search for output folders with a cache
            dirs_to_process = sorted(p.parent for p in in_path.rglob("ocr_cache/metadata.json"))
            if not dirs_to_process:
                # No cache — try as a single output folder anyway
                if (in_path / "images").exists():
                    dirs_to_process = [in_path]
                else:
                    console.print("[red]Error:[/red] No output folders with images/ found.")
                    sys.exit(1)

        total_restored: list[str] = []
        for out_dir in dirs_to_process:
            console.print(f"  Processing: {out_dir.name}/")
            restored = restore_image_refs_in_file(out_dir)
            total_restored.extend(restored)

        if total_restored:
            console.print(
                f"\n[green]Done[/green] — restored {len(total_restored)} image reference(s):"
            )
            for name in total_restored:
                console.print(f"    + {name}")
        else:
            console.print("\n[green]Done[/green] — all images already referenced, nothing to fix.")
        return

    # -- Normal mode ---------------------------------------------------------
    # Resolve the server host: --host takes priority, then default per backend.
    effective_host = host or _DEFAULT_HOSTS.get(backend, DEFAULT_OLLAMA_HOST)

    # --cleanup-model implies --hybrid
    if cleanup_model and not hybrid:
        hybrid = True

    # --hybrid requires a cleanup model for the merge LLM
    if hybrid and not cleanup_model:
        console.print(
            "[yellow]Warning:[/yellow] --hybrid works best with"
            " --cleanup-model to specify the merge LLM."
        )

    in_path = Path(input_path)

    # In merge-only mode the input IS the output directory (or parent
    # of output directories).  Skip the normal PDF-based resolution.
    if merge_only:
        out_path = in_path
    else:
        out_path = resolve_output_path(in_path, Path(output_path) if output_path else None)

    # Header
    console.print(f"[bold]pdf2md[/bold] v{__version__}")
    if merge_only:
        console.print(
            "  Mode:       [bold cyan]merge-only[/bold cyan]"
            " (re-running post-processing from OCR cache)"
        )
    console.print(f"  Backend:    {backend}")
    if backend == "ollama":
        console.print(f"  Model:      {model}")
    console.print(f"  Host:       {effective_host}")
    console.print(f"  DPI:        {dpi}")
    if backend == "ollama":
        console.print(f"  Context:    {num_ctx}")
    console.print(f"  Batch size: {batch_size}")
    # Show enabled enhancements
    enhancements = []
    if merge_pages:
        enhancements.append("merge-pages")
    if hybrid:
        label = f"hybrid ({cleanup_model})" if cleanup_model else "hybrid"
        enhancements.append(label)
    if not system_prompt:
        enhancements.append("no-system-prompt")
    if enhancements:
        console.print(f"  Enhance:    {', '.join(enhancements)}")
    console.print(f"  Input:      {in_path}")
    console.print(f"  Output:     {out_path}/")
    console.print()

    # Create backend
    try:
        if backend == "ollama":
            ocr_backend = create_backend(
                "ollama",
                model_name=model,
                host=effective_host,
                num_ctx=num_ctx,
                cleanup_model=cleanup_model,
                use_system_prompt=system_prompt,
            )
        elif backend == "llama-cpp":
            backend_kwargs: dict[str, Any] = {
                "host": effective_host,
                "num_predict": num_ctx,
                "ocr_model": model if model != DEFAULT_OLLAMA_MODEL else None,
                "cleanup_model": cleanup_model,
                "use_system_prompt": system_prompt,
            }
            ocr_backend = create_backend("llama-cpp", **backend_kwargs)
        else:
            console.print(f"[red]Unknown backend:[/red] {backend}")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]Failed to create backend:[/red] {e}")
        sys.exit(1)

    converter = Converter(
        ocr_backend,
        dpi=dpi,
        batch_size=batch_size,
        merge_pages=merge_pages,
        hybrid=hybrid,
        merge_only=merge_only,
    )

    # Progress tracking
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    current_task_id: TaskID | None = None
    current_task_desc: str = ""
    current_task_total: int = 0
    previous_task_ids: list[TaskID] = []

    def on_progress(event: ProgressEvent) -> None:
        nonlocal current_task_id, current_task_desc, current_task_total

        if event.stage == "file_start":
            # Mark previous task complete if any
            if current_task_id is not None:
                progress.update(current_task_id, completed=current_task_total)
                previous_task_ids.append(current_task_id)
            # Remove all progress bars from the previous file
            for tid in previous_task_ids:
                progress.remove_task(tid)
            previous_task_ids.clear()
            console.print(f"\n[bold cyan]Converting:[/bold cyan] {event.message}")
            current_task_id = None

        elif event.stage == "ocr":
            if current_task_id is None:
                current_task_id = progress.add_task("OCR pages", total=event.total)
                current_task_desc = "OCR pages"
                current_task_total = event.total
            progress.update(current_task_id, completed=event.current + 1)

        elif event.stage == "hybrid":
            # Finish the OCR progress bar if still open
            if current_task_id is not None:
                if "Hybrid" not in current_task_desc:
                    progress.update(
                        current_task_id,
                        completed=current_task_total,
                    )
                    previous_task_ids.append(current_task_id)
                    current_task_id = None
            if current_task_id is None:
                current_task_id = progress.add_task("Hybrid merge", total=event.total)
                current_task_desc = "Hybrid merge"
                current_task_total = event.total
            progress.update(current_task_id, completed=event.current + 1)

        elif event.stage == "file_done":
            if current_task_id is not None:
                progress.update(current_task_id, completed=event.total)
                previous_task_ids.append(current_task_id)
                current_task_id = None

    # Make Ctrl+C immediately raise KeyboardInterrupt even if we're
    # inside a blocking urllib call or thread-pool future.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # Run conversion
    t0 = time.monotonic()

    backend_label = "Ollama" if backend == "ollama" else "llama-server"

    try:
        with progress:
            console.print(f"[dim]Connecting to {backend_label}...[/dim]")
            converter.convert(in_path, out_path, on_progress=on_progress)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    console.print(f"\n[green]Done[/green] in {elapsed:.1f}s")
    console.print(f"  Output: {out_path}/")
