# Final retrieval test sets: authoring and execution procedure

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

## 7. Final-2 (after routing v3 / answerability v3.1)
The original final set is development data from now on, like dev and diagnostic.

1. Freeze the candidate: `python -m mabhas19.cli.freeze_candidate --write` writes
   `final2_candidate.yaml` (code tree, curated tree, scope map, taxonomy, chunks, BM25, E5 pins,
   RRF, structured schemas, evaluator). Commit it. The Final-2 run refuses on any difference.
2. Anchors: a new seed; exclude every evidence ID of dev, diagnostic and final, plus records and
   sections directly adjacent to them where the pool allows; report every category whose unused
   pool is too small instead of reusing development evidence.
3. Authoring by a fresh agent with access only to the anchors, the curated sources and the
   authoring brief (no code, router rules, scope taxonomy, earlier queries, failures or runs);
   review by a separate fresh agent (routes, records, no-match, issues, prose evidence,
   acceptable routes under the §6 policy). Unsupported questions are written from the
   underlying domains, not from any taxonomy.
4. Validate every evidence ID, `--freeze` (SHA-256), commit `final2.yaml` + `final2.sha256`.
5. Run once: `python -m mabhas19.cli.embedding_experiment --set final2` (corrected evaluator,
   scope-class confusion recorded). Report everything; change nothing afterwards.

## 8. Final-3 (after the structured safety cycle)
Dev, diagnostic, the original final set and Final-2 are all development data from now on.

1. Freeze the candidate: `python -m mabhas19.cli.freeze_candidate --set final3 --write` writes
   `final3_candidate.yaml` (§7 pins plus the constraint-parser, entity-resolver and evaluator
   versions and the safety invariant `wrong_regulatory_values_expected: 0`). Commit it. The
   Final-3 run (`embedding_experiment --set final3`) refuses on any difference;
   `final2_candidate.yaml` stays as the historical record of the Final-2 candidate.
2. Anchors: a new seed; exclude every evidence ID of all four earlier sets, plus adjacent
   sections, blocks and records where the pool allows; report each shortfall explicitly.
3. Authoring by a fresh agent that sees only the anchors, the curated sources and a
   self-contained authoring brief (no code, router or scope rules, external-domain taxonomy,
   earlier queries, failures or metrics); review by a separate fresh agent. Stress coverage is
   written from independent source/domain anchors, never as variants of earlier failures.
4. Validate every evidence ID, record expectation, no-match expectation, acceptable route and
   issue/conflict ID; `--freeze` (SHA-256); commit `final3.yaml` + `final3.sha256`; verify the
   candidate manifest. Only then run once.
5. Safety is reported first: wrong regulatory values (hard requirement 0 — any nonzero value
   blocks generation), wrong extra rows, silently dropped constraints. Failures are reviewed by
   severity; nothing is fixed after the run.

## 9. Final-4 (after routing v4)
Dev, diagnostic, the original final set, Final-2 and Final-3 are all development data from now on.

1. Freeze the candidate: `python -m mabhas19.cli.freeze_candidate --set final4 --write` writes
   `final4_candidate.yaml`: the candidate commit (an ancestor of every later run; after it only
   `final4.yaml` and `final4.sha256` may change), the §8 pins plus the query-plan (with its outcome
   codes), numeric-parser and router versions, the structured-schema, scope-map, taxonomy and
   policy-ambiguity fingerprints, evaluator `corrected-3`, and the safety invariants
   (wrong regulatory values, wrong extra rows and silently dropped constraints all 0). Commit it.
   `final2_candidate.yaml` and `final3_candidate.yaml` stay historical.
2. Anchors: a new seed; exclude every evidence ID of all five earlier sets, plus adjacent
   sections, blocks and records where the pool allows; report each shortfall explicitly.
3. Authoring by a fresh agent that sees only the anchors, the curated sources (PDFs where needed)
   and a self-contained brief (no code, router or planner logic, taxonomy, policy-ambiguity
   record, earlier queries, failures or metrics); review by a separate fresh agent that never
   tunes a case toward current behaviour. Cases may state, besides the §4 evidence keys,
   `no_match_outcomes` (acceptable plan outcome codes for a `no_records` case) and
   `enumeration: {dimension}` (an explicitly requested enumeration of printed alternatives).
4. `--freeze` validates every evidence ID, record, typed no-match outcome, enumeration
   dimension, calculation refusal, issue/conflict ID and acceptable route, then writes the
   SHA-256; commit `final4.yaml` + `final4.sha256`; check the candidate manifest. Only then run
   once: `python -m mabhas19.cli.embedding_experiment --set final4`.
5. Report safety first (hard requirement 0 / 0 / 0; a nonzero wrong regulatory value blocks
   generation), then structured, routing/scope and prose (BM25, vector, hybrid; intrinsic and
   end-to-end). Failures are classified by severity; nothing is changed after the run.

## 10. Final-5 (after M1 source-structure identity and M2 source-attested constraints)
Dev, diagnostic, the original final set and Final-2 to Final-4 are all development data from now
on; Final-4 and its run stay immutable (its L338/L342 adjudication is recorded in
`final4_adjudication.yaml`).

1. Freeze the candidate: `python -m mabhas19.cli.freeze_candidate --set final5 --write` writes
   `final5_candidate.yaml`: the §9 pins plus admission, certification, identity (with the
   identity-snapshot fixture fingerprint), polarity, operations, rating vocabulary and
   final-authoring validation modules; router `v4.1`, query plan `plan-2`, numeric parser
   `numeric-2`, constraint parser `attested-1`, entity resolver `source-structure-1`, evaluator
   `corrected-4` with `final-validation-1`; and the safety invariants (wrong regulatory values,
   wrong extra rows, silently dropped constraints and failed certifications all 0). Commit it.
2. Anchors: a new seed; exclude every evidence ID of all six earlier sets (and adjacent IDs
   where the pool allows); report each shortfall.
3. **Exclusions are for authoring independence, not for the expected answer.** An excluded ID
   never seeds a case. When an independently written question's true answer is an excluded
   record, the case lists that record and sets `uses_excluded_records: true`, or it is dropped
   before freeze. It is never written as a structured case with no expected records.
4. Authoring by a fresh agent and review by a separate fresh agent, as in §9 (no code, router,
   planner or candidate output; the candidate is never run while authoring or reviewing).
5. `--freeze` runs the §9 validation plus the Final-5 authoring validation
   (`retrieval/final_validation.py`), which rejects a structured/mixed case with no expected
   records that names a curated record, cites a source issue or conflict located on records,
   or states neither a typed no-match (`no_records` + `no_match_outcomes`) nor a table-level
   question; and an excluded record used without `uses_excluded_records`. The checks read only
   the case and the curated source, never the candidate under test. Then commit `final5.yaml` +
   `final5.sha256`, check the manifest, and run once:
   `python -m mabhas19.cli.embedding_experiment --set final5`.
6. Report safety first (hard requirement 0 / 0 / 0 / 0), then structured, routing/scope and
   prose. Nothing is changed after the run.
