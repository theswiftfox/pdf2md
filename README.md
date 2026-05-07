# pdf2md

Convert PDF documents to Markdown using vision-language OCR models, with optional hybrid correction using raw PDF text extraction.

## Features

- **Two inference backends**: [llama.cpp](https://github.com/ggerganov/llama.cpp) server or [Ollama](https://ollama.com)
- **Hybrid merge**: Combines character-perfect PDF text extraction with OCR-derived markdown structure for best-of-both-worlds accuracy
- **Cross-page merge**: Joins tables, paragraphs, and lists split across page boundaries
- **Batch processing**: Convert entire directories of PDFs recursively
- **OCR caching**: Re-run post-processing with different models without repeating expensive OCR

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

## Usage

### Basic OCR (Ollama)

```bash
pdf2md document.pdf
```

### llama.cpp with hybrid merge

```bash
pdf2md input_data/ output_data/ \
  -b llama-cpp --host http://localhost:8080 \
  -m LightOnOCR-2-1B-Q8_0 --no-system-prompt \
  --hybrid --cleanup-model Qwen3.6-35B-A3B-UD-Q4_K_M \
  --merge-pages --batch-size 1 --dpi 200
```

### Re-run merge with a different model (no OCR re-run)

```bash
pdf2md --merge-only output_data/ \
  -b llama-cpp --host http://localhost:8080 \
  --hybrid --cleanup-model Qwen3.6-35B-A3B-UD-Q4_K_M \
  --merge-pages
```

## Options

| Option | Description |
|--------|-------------|
| `-b, --backend` | `ollama` (default) or `llama-cpp` |
| `-m, --model` | Model name/alias for OCR |
| `--host` | Server URL |
| `--dpi` | PDF rendering resolution (default: 144) |
| `--batch-size` | Concurrent requests per batch |
| `--hybrid` | Correct OCR using raw PDF text + merge LLM |
| `--cleanup-model` | Model for hybrid merge (implies `--hybrid`) |
| `--merge-pages` | Join content split across page boundaries |
| `--merge-only` | Skip OCR, re-run post-processing from cache |
| `--no-system-prompt` | Bare image mode (for models like LightOnOCR) |
| `--num-ctx` | Context window size (Ollama only) |

## Pipeline

1. **OCR** — Each page is rendered and sent to a vision-language model
2. **Hybrid merge** (optional) — Raw PDF text (ground truth) is combined with OCR markdown (structure) via a text LLM
3. **Cross-page merge** (optional) — Split tables and paragraphs are rejoined

Results are cached in `ocr_cache/` so step 1 can be skipped on subsequent runs.

## Development

```bash
pip install -e ".[dev]"
ruff check src/          # lint
ruff format src/         # format
mypy src/pdf2md/         # type check
```

## License

Apache-2.0
