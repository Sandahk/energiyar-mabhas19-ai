# Retrieval answerability: findings and proposal (routing v3)

Status: analysis and proposal only. Nothing below is implemented; v3 routes to `unsupported`
only on positive evidence (answerability gate category, unresolved required entity after the
v3 resolver, unknown section/table reference). Low lexical corpus coverage is recorded as
diagnostic metadata and never decides `unsupported`.

## 1. Experiment: lexical evidence coverage (2026-10-18)
For every query routed to prose on the dev (76), diagnostic (117) and final-as-development
(152) sets, the *focus terms* (query words that are not function, question, request, unit or
explanation words and occur in at most 8% of chunks) were compared with the best of the top-3
retrieved chunks.

| Expected route | n | best-chunk focus coverage |
|---|---|---|
| prose | 105 | spread over 0.0-1.0; 25 cases at or below 0.4 |
| unsupported | 9 | 0.1-0.7 |

Findings:

- Coverage does **not** separate answerable from unanswerable questions: valid colloquial or
  conceptual questions (e.g. «… تو مدل بیاد», «… حساب میشه؟») score as low as clearly external
  ones, because their distinctive words are conversational, not regulatory.
- Every threshold that catches the external questions also rejects valid conceptual questions;
  the strictest rule tried (coverage 0 with at least 3 focus terms) would have flagged only one
  case, a valid material question.
- Removing the old low-coverage fallback loses three external questions it used to catch
  (traffic rules, architectural history, LEED points) and fixes six valid questions it used to
  reject. No embedding-similarity threshold is used; none was chosen or tuned.

## 2. Proposal: source-domain-aware scope classification
Classify a question by *what the sources cover*, using a maintained scope map derived from the
Mabhas 19 structure and appendices rather than a growing negative keyword list.

Scope map (curated, reviewed, versioned with the corpus):

- **In-scope domains**: one node per chapter/section of the main document (19-1 … 19-8) and per
  appendix, with the section title, the defined terms of 19-1 (term + English equivalent) and
  the printed table dimensions (occupancies, climate classes, materials, equipment, spaces).
- **Explicit boundaries printed in the sources**: places where Mabhas 19 itself defers to other
  regulations (e.g. «مبحث سیزدهم», «مبحث سوم») or to placeholders (SI-0004); each boundary names
  the external domain it hands off to.
- **Adjacent external domains**: a short, reviewed list of neighbouring regulatory domains
  (structural/seismic design, municipal fees and zoning, tariffs, parking provision, acoustics,
  lift sizing, plumbing sizing), each linked to the Mabhas or authority that covers it. Added
  only when an adjudicated question shows the need, never from test wording alone.

Decision classes:

1. **In-scope concept or requirement** - focus terms map to a scope node (title, defined term,
   clause vocabulary) -> prose/structured as routed.
2. **Structured question with an unresolved entity** - a table's dimensions are asked for but
   an entity is not printed -> typed no-match (unsupported or structured, both acceptable).
3. **External domain** - the question's focus maps to an adjacent external domain or a printed
   hand-off boundary and to no in-scope node -> unsupported, naming the domain that covers it.
4. **Nonexistent reference** - section/table/appendix number not in the curated index ->
   unsupported (implemented in v3).
5. **Valid but colloquial question** - low lexical overlap but the focus maps to a scope node
   (directly or through the hybrid prose evidence) -> answered; low coverage stays diagnostic.

Evaluation: develop on dev + diagnostic + final-as-development; judge only on a new untouched
final-2 set. Report false positives on valid conceptual questions separately from recall on
external ones.
