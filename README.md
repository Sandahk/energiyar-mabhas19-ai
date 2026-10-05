# energiyar-mabhas19-ai

Persian AI assistant for the **5th edition (1404) of Mabhas 19** (Iranian National Building
Regulations — energy management in buildings) and its 9 official appendices.

The system uses retrieval-augmented generation (RAG). It answers only from the official
source documents and cites the document, section or table, and page (both the PDF page and
the printed page). It must never fabricate regulatory values or citations.

> **Status:** Phase 2A (extraction benchmark). No full-corpus extraction, retrieval, model or API
> code yet. Findings: [`docs/source-audit.md`](docs/source-audit.md),
> [`docs/extraction-benchmark.md`](docs/extraction-benchmark.md).

## Repository layout

| Path | Purpose | In Git |
|---|---|---|
| `sources/` | Immutable official PDFs plus `manifest.yaml` (SHA-256, page count, original filename, versions) | yes |
| `curated/` | Human-verified knowledge: page maps, section structure, structured tables, text corrections | yes |
| `data/` | Derived, regenerable outputs (renders, OCR, chunks, indexes), keyed by source hash | no |
| `eval/golden/` | Golden evaluation questions with expected answers and citations | yes |
| `eval/runs/` | Evaluation run reports | no |
| `src/mabhas19/` | Application package (`core`, `text`, `ingest`, `tables`, `indexing`, `retrieval`, `generation`, `providers`, `api`, `cli`) | yes |
| `tests/` | `unit`, `integration`, `regression` tests and small `fixtures` | yes |
| `configs/` | Project and provider configuration (no secrets) | yes |
| `docs/` | Architecture notes, decision records (`adr/`), source audit | yes |
| `scripts/` | One-off maintenance scripts | yes |

## Source documents

Original PDFs are stored under `sources/mabhas19/ed5-1404/` with ASCII canonical names
(`m19-ed5-1404-main-v1.pdf`, `m19-ed5-1404-app01-v1.pdf` … `app09-v1.pdf`). Original
filenames, hashes and provenance are kept in `sources/manifest.yaml`.

- Source bytes are never modified. The test suite verifies every file against its SHA-256.
- A better original is added as a new version (`-v2`, `supersedes: v1`). Nothing is overwritten.
- See [`docs/source-audit.md`](docs/source-audit.md) for known text-layer issues.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```sh
uv sync                 # create .venv and install the project + dev tools
uv run pytest           # tests (includes source integrity check)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type check (strict)

# Page-by-page audit of all sources -> data/interim/<doc_id>/<sha8>/
uv run python -m mabhas19.cli.audit_sources

# Extraction benchmark (needs Tesseract 5 + fas/eng traineddata in data/models/tessdata)
uv run python -m mabhas19.cli.extraction_benchmark
```

Copy `.env.example` to `.env` for local settings. Never commit `.env`.
