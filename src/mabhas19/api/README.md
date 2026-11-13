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
| `clarification` | A Persian question for the one missing or ambiguous item, with the printed choices as buttons (or a free-text field), interpretation echo |
| `refusal` | Why the engine cannot answer safely, what is missing, the printed options when a name is ambiguous, interpretation echo |
| `out_of_scope` | Why the question is outside the loaded corpus, and the covering regulation when the scope map names one |

### Clarifications (pre-pilot usability pass)

A refusal that the engine typed as one missing or ambiguous item becomes a question for that
item. The retrieval engine is unchanged; the composer only reads its typed outcome.

| Engine outcome | Question | Choices |
|---|---|---|
| `missing_required_dimension` (category) | «برای پاسخ دقیق، لطفاً اقلیم را مشخص کنید.» etc. | The printed domain: climate classes, energy ratings, building groups, years |
| `missing_required_dimension` (row of a main table) | «کدام جزء…/محل مصرف… منظور است؟» | The table's printed row labels (up to 16) |
| `missing_required_dimension` (interval) | «مساحت زیربنای ساختمان چقدر است؟» | Free text, prefixed with the dimension word; no example number |
| `ambiguous_entity` | «کدام جزء…/نوع کاربری دقیق‌تر کدام است؟» | The printed candidates the engine could not choose between |
| `unresolved_constraint` / `no_structured_anchor`, one resolved entity and its printed quantity | «منظور شما «λ» برای «…» است؟» | That printed quantity only; for R asked of an Appendix 7 material, a note that the source prints λ, that R depends on thickness and that nothing is calculated |

- **Choices are printed values only.** `generation/options.py` reads them from the curated
  store, and the provenance check re-derives them independently: an option that is not printed
  in its own table is a violation and the answer becomes a refusal.
- **A choice never answers anything itself.** The page sends the original question and the
  choice; the server adds the choice in parentheses (`question (اقلیم 3B)`) and the frozen
  engine runs again with every gate. The next answer may be a value, another clarification or
  a refusal.
- **Matched only by the quantity's name:** when the candidate rows share no word with the
  question beyond the requested quantity (an external wall's «ضریب انتقال حرارت» matched only
  the window rows of table 19-5-2), no choice is offered; the refusal says the item was not
  found in that table.
- A clarification is never shown with a value; the refusal behind it stays in the answer
  (`refusals`) and in the log.

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
| `POST /api/ask` `{"question", "participant", "clarify"?}` | `{"answer": <Answer>, "latency_ms"}`; `clarify` = `{"parent", "field", "kind": "option"\|"free", "text"}` continues a clarification |
| `GET /api/health` | Status, pilot warning, privacy notice, engine mode and pins |
| `POST /api/feedback` `{"request_id", "rating": "up"\|"down", "note", "participant"}` | Records feedback |

`participant` is optional: a pseudonymous pilot ID of 1–40 ASCII letters, digits, `-` or `_`
(`engineer_01`, `student_03`). The page generates `pilot-xxxxxx` once, keeps it in the browser and
lets the tester change it; anything else (a name with spaces, an email) is refused. No account,
password or email exists.

## Logs (`data/chat/`, git-ignored, append-only)

| File | One line per | Holds |
|---|---|---|
| `requests.jsonl` | question (immutable) | timestamp, request ID, participant, `follow_up` (parent request, field, option/free, added text), question (raw and normalized), route and scope, plans, interpretation echo, certified rows, typed outcomes, prose chunks, response type, `structured_answer`, refusals, clarifications (field, choices), sources, provenance violations, latency, engine, `auto_hint` |
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

### Pre-pilot routing patch (2026-11-13)

One approved patch before M2, in the router's intent wording only (two families):

- **Energy limit:** «حد مجاز / حداکثر / سقف / بیشینه مصرف انرژی», «مصرف مجاز انرژی» name the
  Appendix 3 output; missing occupancy, area band or rating are still asked, never assumed.
- **U of an opaque element:** «ضریب انتقال حرارت / U / U-value» of an element table 19-5-1
  prints (walls, roofs, floors, opaque doors) is about 19-5-1, never 19-5-2. The engine still
  refuses the U slot; the answer explains that the requirement is a minimum R and offers the
  printed R as an explicit choice. A shown R carries a notice that it is R, not U. No U = 1/R.

The pin guard compares with Final-8 after applying `eval/retrieval/m2_prepilot_pins.yaml`
(only `retrieval/router.py`'s hash, from its Final-8 value). Sanity check: the frozen
paraphrase set `eval/prepilot/paraphrases.yaml` (development only, not a benchmark).

### Known issues (hypotheses for the pilot, not fixed)

Routing issues found by the clarification pass and the pre-pilot patch (not fixed):

- Opaque elements under unprinted names («بام», «دیوار بیرونی») are not matched to the printed
  19-5-1 rows; a conversational «دیوار خارجی من چه ضریب … داشته باشه» is still refused.
- Window U questions («ضریب انتقال حرارت مجاز پنجره») are refused as matched only by the
  quantity's name in 19-5-2 (the table names transparent parts differently).
- A clarification can offer rows matched by a generic word only (a roof U question with «حد
  مجاز» was offered 19-5-2 SHGC rows before the patch; no longer for that question).
- A city does not stand in for a climate class in tables 19-5-1/19-5-2 («… (در تبریز)» is not
  applied); the climate clarification therefore offers classes and a hint to look the city up.
- An ambiguous label the engine matched through its own normalization («چهارلوله» →
  «4 لوله») stays a plain refusal: adding the printed label does not resolve it.

M1 issues:

- Table-name questions can misroute («جدول ۱۹-۳-۲ را نشان بده» reads «بده» as a city).
- Whole-table requests are refused (only single rows are served).
- A WMO station-code request is refused as an unattested quantity.
- Prose ranking is occasionally noisy (a loosely related definition in the top three).
- Source-issue titles are English (`SI-0009: …`).
- Some echo labels outside the clarified label kinds are the engine's normalized forms.
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
