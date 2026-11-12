# energiyar-mabhas19-ai

> **Work in progress.** This is a research codebase, not a finished product. Answer
> generation is **disabled**: retrieval has not yet passed the safety bar that would allow it.
> Nothing here is a usable assistant yet, and its output must not be relied on for compliance.

Persian AI assistant for the **5th edition (1404) of Mabhas 19** (Iranian National Building
Regulations — energy management in buildings) and its 9 official appendices.

The goal is a retrieval-augmented assistant that answers only from the official documents and
cites the document, section or table, and page (both the PDF page and the printed page). Its
one non-negotiable rule: **it must never state a regulatory value or citation that the source
does not support.** Every design decision below follows from that rule.

## Status (2026-11-12)

| Area | State |
|---|---|
| Source registry and audit | Done. 10 PDFs pinned by SHA-256; text-layer issues recorded |
| Curated knowledge | Main text structure and page maps; structured tables from the main text and appendices 2, 3 and 7, reviewed |
| Retrieval | Built and frozen: deterministic router, hybrid prose search, fail-closed structured lookup |
| Safety evaluation | Ten frozen evaluation sets (2,394 questions); last decision run (Final-8) **not promoted** |
| LLM semantic shadow (SH0) | **Closed, not promoted.** The semantic veto (S1-V) stayed safe but caught too few misreads: on its final holdout, 0 harmful vetoes and 100% precision, but recall 15/67 = 22.4% against a 40% threshold. No result-affecting LLM layer is approved |
| Answer generation, API, UI | **M1 closed-pilot chatbot built**: deterministic Persian answers from certified rows and quoted sections, with an interpretation echo, a local API and a chat page. No LLM; not deployed |

### Where the project stands

The deterministic retrieval engine is **frozen** at the Final-8 candidate. Final-8 failed
promotion on two questions whose meaning, not their wording, decides the answer: a colloquial
possessive «مال» read as the occupancy name "mall", and a background remark read as a value to
verify. Fixing such cases with more rules was judged the wrong direction, so deterministic
safety work stopped there.

The next phase tested whether a language model could help **only by proposing** an
interpretation of the question, which deterministic code then verified field by field, with no
power to add a row, a value or a citation. See [`src/mabhas19/shadow/`](src/mabhas19/shadow/README.md).
That line is now closed:

1. **Pilot 1** (80 questions, Gemini): the original veto policy was too aggressive and harmed
   correct answers.
2. **V1** (`s1v-grammatical-veto-1`) narrowed the model's authority to one lever: an explicit,
   verified `grammatical` label on a grounded word.
3. **`veto1`** (168 cases): Gemini 3.8 Flash was safe but caught 10/40 = 25% of opportunities,
   below the 40% usefulness threshold. Gemini 4 Argon passed every threshold (19/40 = 47.5%).
   `veto1` was used to choose the model, so it is model-selection evidence only.
4. **`s1v_holdout1`** (223 cases, a fresh holdout with a documented limited isolation exception):
   Argon with V1 made 0 harmful vetoes at 100% precision, but caught only 15/67 = 22.4% of
   opportunities. Recall failed, so S1-V is **not promoted** and the line is closed.

### Stopping decision (2026-11-12)

Semantic veto with the tested LLM configurations did not reach the required recall at the
required precision. The deterministic safety engine remains the production boundary. No
result-affecting LLM semantic layer is approved.

- Nested-span abstention was necessary: on the final holdout it stopped two vetoes that would
  have removed correct answers.
- All 52 missed opportunities were semantic-label misses: the model called the word
  `contextual`, not `grammatical`. No miss came from the verifier, the guards or invalid output.
- The limit is therefore semantic usefulness, not verifier safety.

Recovery of failed-closed questions (S1-R) was never started and is not planned.

### Closed-pilot chatbot (M1)

For a **closed pilot only** (not public production), the rule "no product before retrieval is
promoted" is replaced by a product-level safety contract:

- a table answer shows only rows the frozen engine has certified, and every number shown comes
  from the question or a certified row;
- every answer shows its source (table or section, page); a table answer also shows how the
  question was read («برداشت سیستم از سؤال شما»), so a misreading is visible;
- a failed or ambiguous lookup stays a refusal, explained in Persian, with what is missing;
- no language model generates, rewrites or interprets anything; the retrieval engine is
  unchanged and verified against its Final-8 pins at startup;
- every request and feedback is logged locally for the pilot review, under a pseudonymous
  participant ID; testers are told so on the page.

See [`src/mabhas19/api/`](src/mabhas19/api/README.md).

## How retrieval works

```
question
  └─ normalization (Persian letters, digits, spacing, ZWNJ)
      └─ answerability gate ── out of scope? → typed refusal
          └─ router v4: structured | prose | mixed | unsupported
              ├─ prose: BM25 + multilingual-e5-small, reciprocal rank fusion
              └─ structured: plan → admission → coverage → execution → certification
                    any step unsure? → fail closed with a typed outcome (never a guess)
```

- **No LLM in the answer path.** Every routing and lookup decision is deterministic and records
  the rule that fired.
- **Fail closed.** A structured row is returned only when every part of the question has been
  interpreted, proven against the question text, executed and independently re-checked.
- **Source-exact identity.** A question names a table row only through normalization,
  reversible morphology or a reviewed alias; fuzzy matching is never evidence.

Details: [`src/mabhas19/retrieval/`](src/mabhas19/retrieval/README.md).

## How it is evaluated

Each development cycle ends with one run on a **new, unseen** question set authored by an
isolated agent and reviewed independently. That run is recorded once and never re-scored;
afterwards the set becomes development data. Safety is reported before quality, and any wrong
regulatory value blocks promotion. See [`eval/retrieval/`](eval/retrieval/README.md).

## Repository layout

| Path | Purpose | In Git |
|---|---|---|
| `sources/` | Immutable official PDFs plus `manifest.yaml` (SHA-256, page count, original filename, versions) | yes |
| `curated/` | Human-verified knowledge: page maps, section structure, structured tables, text ([details](docs/curated.md)) | yes |
| `data/` | Derived, regenerable outputs (renders, OCR, indexes, run records, replay stores) | no |
| `eval/retrieval/` | Frozen retrieval evaluation sets, candidate pins and the final-test procedure | yes |
| `eval/shadow/` | Frozen SH0 pilot and veto-benchmark manifests | yes |
| `eval/extraction/` | Extraction benchmark references | yes |
| `src/mabhas19/` | Application package (see below) | yes |
| `tests/` | `unit`, `integration`, `regression` tests and small `fixtures` | yes |
| `configs/` | Project and provider configuration (no secrets) | yes |
| `docs/` | Source audit, curated-data notes, benchmark findings, recorded technical debt | yes |

Package modules:

| Module | Purpose | State |
|---|---|---|
| `core`, `text`, `ingest` | Paths, configuration, Persian text handling, PDF audit and extraction | in use |
| `tables` | Build, review and promotion of the curated structured tables | in use |
| `indexing` | Chunking of the curated main text | in use |
| `retrieval` | Routing, prose search, structured planning and safety checks | frozen |
| `shadow` | SH0 semantic proposer, verifier and pilot tooling | offline only |
| `providers` | Embedding, OCR and vision provider adapters | in use |
| `benchmark` | Extraction benchmark scoring | in use |
| `cli` | Command-line entry points | in use |
| `generation`, `api` | Deterministic answer composer with provenance checks; pilot server and Persian chat page | closed pilot (M1) |

## Source documents

Original PDFs are stored under `sources/mabhas19/ed5-1404/` with ASCII canonical names
(`m19-ed5-1404-main-v1.pdf`, `m19-ed5-1404-app01-v1.pdf` … `app09-v1.pdf`). Original
filenames, hashes and provenance are kept in `sources/manifest.yaml`.

- Source bytes are never modified. The test suite verifies every file against its SHA-256.
- A better original is added as a new version (`-v2`, `supersedes: v1`). Nothing is overwritten.
- See [`docs/source-audit.md`](docs/source-audit.md) for known text-layer issues.
- The PDFs are official publications of Iran's National Building Regulations (Mabhas 19, 5th edition, 1404),
  included unmodified for research and reproducibility; the curated text and tables are derived from them.
  All rights in the source documents remain with the publisher. The code in this repository is the
  author's own work.

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

# Phase 2B: score cached Gemini appendix 7 results offline (no API calls)
uv run python -m mabhas19.cli.score_app07 --with-tesseract

# Offline retrieval benchmark (BM25 baseline; no model)
uv run python -m mabhas19.cli.eval_retrieval --set dev

# Hybrid retrieval on development sets (BM25 + local multilingual-e5-small, CPU PyTorch)
uv run --group embeddings python -m mabhas19.cli.embedding_experiment --set dev --set heldout

# Closed-pilot chatbot on http://127.0.0.1:8019/ (frozen hybrid engine; logs to data/chat/)
uv run --group embeddings python -m mabhas19.api.server

# Closed-pilot review and report from the logs (no engine call)
uv run python -m mabhas19.cli.pilot_report pending
uv run python -m mabhas19.cli.pilot_report report

# Answer snapshots over the development sets -> data/answers/
uv run --group embeddings python -m mabhas19.cli.answer_snapshots

# SH0 pilot, local checks only (no network)
uv run python -m mabhas19.shadow.pilot verify
uv run python -m mabhas19.shadow.pilot preflight
```

The `embeddings` group installs CPU-only PyTorch from the PyTorch CPU index; `ocr-easyocr` keeps
PyPI PyTorch, so the two groups are declared conflicting and cannot share one environment.
`uv run --group …` and `uv sync --inexact --group …` add a group without removing others; a
plain `uv sync --group …` is exact and removes groups not named on the command line.

Copy `.env.example` to `.env` for local settings. Never commit `.env`.

### Ground rules for contributors

- Never add a per-question rule to make a test pass; extend schemas and generic rules instead.
- Wrong regulatory values must stay at 0 on every evaluation set after every change.
- A frozen final run is never re-run or re-scored. Frozen manifests are never edited; a change
  that alters a pinned module means a new freeze.
- No external model call without `--allow-external`, credentials, a frozen manifest and a budget.
