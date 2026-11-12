# `mabhas19.api` and `mabhas19.generation` (M1 closed pilot)

A usable Persian Mabhas 19 chatbot on top of the **frozen** retrieval engine. No language model
is involved: every answer is composed deterministically from one retrieval result.

```
question → frozen engine (router, BM25 + E5 + RRF, structured plan → admission → coverage
           → execution → certification)
         → answer composer (generation/composer.py) → provenance checks (generation/provenance.py)
         → typed answer (generation/contract.py) → API → chat page
```

## Safety contract (closed pilot only)

- A table answer uses only rows the engine certified; each value is read from its own row's
  certified output cell, exactly as printed. No arithmetic.
- Every number shown comes from the question or a certified row (values and conditions), or
  from source metadata (captions, pages, section numbers) and the engine's own outcomes.
- Every answer cites its source. A table answer and a table refusal show the interpretation
  echo, rendered from the engine's plan.
- A typed refusal stays a refusal; the composer cannot turn it into a value.
- Prose answers are verbatim excerpts of retrieved sections (1–3), never paraphrased.
- Every answer passes the provenance checks before it is returned. An answer that fails any
  check is replaced by a refusal that shows nothing, and the violation is logged.
- The server starts only when the live engine matches the Final-8 candidate pins (retrieval
  module hashes, curated tree, schemas, scope map, chunks, BM25/RRF/E5).

## Response types

| Type | Shows |
|---|---|
| `structured_value` | Value, unit, row label, printed conditions, table and page, linked source issues or conflicts, interpretation echo |
| `prose` | 1–3 verbatim excerpts with section number, title and page |
| `refusal` | Why the engine cannot answer safely, what is missing, the printed options when a name is ambiguous, interpretation echo |
| `out_of_scope` | Why the question is outside the loaded corpus, and the covering regulation when the scope map names one |

## Running

```sh
uv run --group embeddings python -m mabhas19.api.server            # http://127.0.0.1:8019/
uv run python -m mabhas19.api.server --mode lexical --allow-unpinned  # development only
```

| Endpoint | |
|---|---|
| `GET /` | Persian RTL chat page |
| `POST /api/ask` `{"question"}` | `{"answer": <Answer>, "latency_ms"}` |
| `GET /api/health` | Status, pilot warning, engine mode and pins |
| `POST /api/feedback` `{"request_id", "rating": "up"\|"down", "note"}` | Records feedback |

## Logs (`data/chat/`, git-ignored)

`requests.jsonl`, one line per request, holds:

- the question, both raw and normalized;
- the route and scope decision;
- the structured plans (entities, constraints, missing fields);
- the interpretation echo;
- certified rows, typed outcomes and retrieved chunks;
- the response type, headline, refusals, sources and provenance violations;
- latency and the engine description;
- `auto_hint`, a deterministic pointer such as `refused:ambiguous_entity`.

`triage` is left empty for the pilot review, which sets one of `correct_answer`, `routing`,
`deterministic_recall`, `semantic_interpretation`, `prose`, `out_of_scope` or `presentation`.

`feedback.jsonl` holds one line per rating: the request ID, `up` or `down`, and an optional note.
No credential, environment value or client address is logged.

## Modules

| Module | Role |
|---|---|
| `generation/contract.py` | The typed answer contract |
| `generation/labels.py` | Persian field names, unit readings and templates (no regulatory numbers) |
| `generation/composer.py` | Retrieval result → answer, read-only over the engine |
| `generation/provenance.py` | Numeric, row, citation, refusal, excerpt and interpretation checks |
| `api/engine.py` | Frozen engine loading and the Final-8 pin check |
| `api/chatlog.py` | Request and feedback logs |
| `api/server.py` | Standard-library HTTP server |
| `api/static/index.html` | Chat page (vanilla JS; all content written as text, never as HTML) |
| `cli/answer_snapshots.py` | Answer snapshots over the development sets for manual review |
