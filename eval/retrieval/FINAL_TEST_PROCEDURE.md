# Final retrieval test set: authoring and execution procedure (not yet executed)

The development set (`dev.yaml`, 76 cases) was used to fix v1, and the diagnostic set
(`heldout.yaml`, 117 cases) was used to diagnose v2. Neither is an unbiased measure any more.
This procedure creates a new, untouched final test set **after v2 is frozen**.

## 1. Freeze v2 first
- Commit the retrieval code; record the commit hash in the final set's header.
- From then on, no change to router rules, answerability categories, normalization, chunking,
  BM25 parameters, RRF, structured lookup or thresholds until the final run is reported.

## 2. Who authors and what they may see
- Preferably the project owner or another person who has **not** seen v2 failures.
- Allowed inputs: the source PDFs, `curated/` (text, sections, tables), and the anchor list
  from step 3.
- Not allowed: retrieval code, indexes, any `data/retrieval/runs/*`, `dev.yaml`,
  `heldout.yaml`, failure reports, or running the retriever while authoring.
- An assistant may author only from the anchor list and curated evidence, under the same
  restrictions, and must not consult earlier failure analyses.

## 3. Draw anchors (seeded, recorded)
```
from mabhas19.retrieval.final_test import sample_anchors
sample_anchors(CURATED_DIR, seed=<new seed>, exclude=<IDs used in dev.yaml + heldout.yaml>)
```
- Choose the seed once (e.g. a date-based integer), record it, and store the anchor list under
  `data/retrieval/final_anchors_<seed>.yaml`.
- Quotas: 30 sections, 10 city climate records, 10 EUI records, 10 materials, 20 main-table
  records, 5 table references. Exclude every section/record already used as evidence in the
  development or diagnostic sets.

## 4. Write the cases
- One natural-language question per anchor (do not copy the regulation's wording verbatim for
  conceptual questions), plus:
  - at least 10 mixed questions combining two anchors (one structured, one prose);
  - at least 8 spelling/normalization variants (Arabic letters, ASCII digits, ZWNJ/space,
    kashida);
  - at least 20 unsupported or adversarial questions written from the scope of the regulation
    (what Mabhas 19 does not cover, values it does not print, unknown entities, conflicts,
    interpolation requests), without looking at the router's categories.
- Evidence only (no expected answers), using the `dev.yaml` schema (`blocks`, `sections`,
  `records`, `acceptable_records`, `tables`, `no_records`, `source_issues`, `conflict`,
  `acceptable_routes`). Every expected route must follow from the curated sources.
- A second person checks each case's evidence against the source before freezing.

## 5. Freeze and run once
1. `uv run python -m mabhas19.cli.eval_retrieval --set final --freeze` (validates every evidence
   ID against curated/ and writes `final.sha256`).
2. Commit `final.yaml` + `final.sha256` before running.
3. `uv run python -m mabhas19.cli.eval_retrieval --set final` exactly once on the frozen v2
   commit; report all metrics and every failure without fixing anything.
4. Any later change to the system is evaluated on a *new* final set, never by re-tuning on this
   one.

## 6. Route policy for future sets (from the first final adjudication)
The first final set (`final.yaml`) stays immutable. New sets (final-2 onward) state the
following ambiguities explicitly with `acceptable_routes` instead of forcing one reading:

- **Unknown entity in a structured question** (a city, occupancy or material not printed in the
  curated tables): `unsupported` or `structured` with a typed no-match are both correct, provided
  no record is returned. Example: an unlisted city's climate class.
- **Interpretation questions about a printed table row** (e.g. what an operator in an Appendix 3
  label means): `structured`, `mixed` or `prose` are acceptable when the case's evidence is
  satisfied (table evidence and the relevant source issue).
- **Record evidence is scored at record level.** List `tables` only when no concrete record is
  expected; a typed no-match never satisfies expected records.
