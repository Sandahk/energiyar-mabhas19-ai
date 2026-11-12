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

The server listens on the loopback interface only. It has no authentication, so remote
testers reach it through a private tunnel or VPN (see below); `--allow-network` exists only for a
trusted private network.

| Endpoint | |
|---|---|
| `GET /` | Persian RTL chat page |
| `POST /api/ask` `{"question", "participant"}` | `{"answer": <Answer>, "latency_ms"}` |
| `GET /api/health` | Status, pilot warning, privacy notice, engine mode and pins |
| `POST /api/feedback` `{"request_id", "rating": "up"\|"down", "note", "participant"}` | Records feedback |

`participant` is optional: a pseudonymous pilot ID of 1–40 ASCII letters, digits, `-` or `_`
(`engineer_01`, `student_03`). The page generates `pilot-xxxxxx` once, keeps it in the browser and
lets the tester change it; anything else (a name with spaces, an email) is refused. No account,
password or email exists.

## Logs (`data/chat/`, git-ignored, append-only)

| File | One line per | Holds |
|---|---|---|
| `requests.jsonl` | question (immutable) | timestamp, request ID, participant, question (raw and normalized), route and scope, plans, interpretation echo, certified rows, typed outcomes, prose chunks, response type, `structured_answer`, refusals, sources, provenance violations, latency, engine, `auto_hint` |
| `feedback.jsonl` | rating | timestamp, request ID, participant, `up`/`down`, note |
| `reviews.jsonl` | manual review | timestamp, request ID, reviewer, `reviewed`, `answer_correct`, `answerable`, `triage`, `wrong_regulatory_value`, `unsupported_numeric_claim`, note |

A review never edits the request log. A later review of the same request supersedes the earlier
one. `answer_correct` and `answerable` take `yes`, `no`, `uncertain` or `not_applicable`;
`triage` is one of `correct_answer`, `routing`, `deterministic_recall`,
`semantic_interpretation`, `prose`, `out_of_scope` or `presentation`, set by a person, never by a
model. No credential, header, environment value or client address is logged.

## Closed pilot (M2)

```sh
uv run python -m mabhas19.cli.pilot_report pending                 # questions not yet reviewed
uv run python -m mabhas19.cli.pilot_report review --request <id> \
    --correct yes|no|uncertain|not_applicable --answerable yes|no|uncertain|not_applicable \
    --triage <class> [--wrong-value] [--unsupported-number] [--note "..."] [--reviewer name]
uv run python -m mabhas19.cli.pilot_report report                  # data/chat/reports/
```

The report reads the logs only. It lists, in this order:

1. **Safety:** wrong regulatory values and unsupported numeric claims found by review, and
   provenance violations caught at answer time.
2. **Usage:** participants, questions, questions per participant, errors.
3. **Answer rate:** structured, prose, refusal and out-of-scope answers.
4. **Feedback:** rate, up, down, negative rate, notes.
5. **Reviewed quality:** correct, incorrect, uncertain; unanswered but answerable; genuinely
   unsupported.
6. **Failure taxonomy** and **latency**.

- **Hard safety criterion:** no unflagged wrong regulatory value in reviewed answers; any wrong
  value is investigated immediately.
- **Product measure:** the real answered and useful rate, with no target set in advance.
- **M3 choice:** the largest meaningful real-user failure bucket, not Final-N counts.

During the pilot the retrieval engine stays frozen (the Final-8 pin guard stays on). A
retrieval problem found by a real question is logged and reviewed, not fixed.

### Known M1 issues (hypotheses for the pilot, not fixed)

- Table-name questions can misroute («جدول ۱۹-۳-۲ را نشان بده» reads «بده» as a city).
- Whole-table requests are refused (only single rows are served).
- A WMO station-code request is refused as an unattested quantity.
- «دیوار بین دو واحد گرمایش‌شده» is refused as ambiguous between three printed walls.
- Prose ranking is occasionally noisy (a loosely related definition in the top three).
- Source-issue titles are English (`SI-0009: …`).
- Some echo labels are the engine's normalized forms (`t19-5-2`).
- Some Appendix 7 names repeat their subgroup («گچ گچ و خاک»).
- Table 19-2-1 shows ✓/✗ without a legend.
- An Appendix-2-only city also shows a «جدول ۱۹-۳-۲: no row» note.

## Modules

| Module | Role |
|---|---|
| `generation/contract.py` | The typed answer contract |
| `generation/labels.py` | Persian field names, unit readings and templates (no regulatory numbers) |
| `generation/composer.py` | Retrieval result → answer, read-only over the engine |
| `generation/provenance.py` | Numeric, row, citation, refusal, excerpt and interpretation checks |
| `api/engine.py` | Frozen engine loading and the Final-8 pin check |
| `api/chatlog.py` | Request, feedback and review logs (append-only) |
| `api/pilot_metrics.py` | M2 metrics from the logs |
| `api/server.py` | Standard-library HTTP server |
| `api/static/index.html` | Chat page (vanilla JS; all content written as text, never as HTML) |
| `cli/answer_snapshots.py` | Answer snapshots over the development sets for manual review |
| `cli/pilot_report.py` | M2 review entry, pending list and report |
