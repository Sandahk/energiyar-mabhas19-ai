# `mabhas19.normalizer`: LLM suggest-and-confirm shadow experiment (closed, NO-GO)

**Status: closed 2026-11-17, NO-GO under the frozen decision rule.** Nothing here is imported by
the chatbot, the API or the retrieval engine (enforced by `tests/unit/test_shadow.py`). It is
kept as the record of one finite experiment and must not be re-run.

## What was tested

When the frozen engine fails closed (prose, refusal or out of scope), could a language model
map the user's everyday Persian onto a closed catalogue of printed table rows, so the chatbot
could ask "did you mean «row» …?" and, after confirmation, run the unchanged engine on a
question built only from printed labels?

- **Model:** Gemini 4 Argon (`gemini-4-argon`), thinking `medium`, 16,000 output tokens,
  300 s timeout, one retry; no temperature.
- **Catalogue** (`catalog.py`): 10 families (Appendix 3, tables 19-5-1 to 19-5-7, Appendix 7,
  Appendix 2 cities), 217 printed row labels.
- **Verifier** (`verify.py`): a row label is kept only when it equals a printed row of the
  proposed family exactly; outputs, climate codes and ratings must be printed for that family
  and written in the question; a city must pass the exact certified city lookup; a number must
  be a verbatim span of the question at the stated offset. A failing field is discarded and
  logged, never repaired or mapped to a near label. The model never supplies a value.
- **Cases** (`eval/normalizer/`): 70 authored questions; 29 engine failures with gold, 4
  abstain-expected, 14 controls (engine answers that must stay untouched), 23 excluded.

## Decision rule (frozen before any model call)

useful ≥ 18/29 · wrong ≤ 3/33 · 0 model-supplied numbers · p90 latency ≤ 20 s · mean cost
≤ $0.05 per call · failed calls ≤ 3 · controls never triggered and unchanged. Any failure:
stop the normalizer line (no prompt tuning, no second model, no new benchmark).

## Result (run once, 2026-11-17)

| Check | Threshold | Result | |
|---|---|---|---|
| Useful | ≥ 18/29 | **24/29 (82.8%)** | pass |
| Wrong | ≤ 3/33 | **4/33 (12.1%)** | **FAIL** |
| Model-supplied numbers | 0 | 0 | pass |
| Latency p90 | ≤ 20 s | 14.2 s | pass |
| Mean cost per call | ≤ $0.05 | $0.0124 | pass |
| Failed calls | ≤ 3 | 0 | pass |
| Controls | never triggered, unchanged | 14/14 not triggered, 14/14 unchanged | pass |

**Decision: NO-GO** (wrong 4/33 against a limit of 3). The line is closed.

- **Abstain-expected cases (4):** the model correctly abstained on N31 (shading-overhang
  ratio), N33 (brick, no printed row) and N60 (shop, no printed space); it proposed on N58.
- **Wrong (4):** N17, N34 and N35 offered the correct row plus extra printed sibling rows
  (floor over pilotis; concrete; gypsum). **N58 is the key risk:** asked for a window-to-floor
  area ratio, the model offered table 19-5-6's daylight-sufficiency area ratio, a different
  quantity with a similar name, and both offered rows would have returned certified values.
  This is semantic substitution, which a field-by-field verifier cannot detect.
- **Other misses:** the model abstained on N29 and N30 (SHGC questions whose gold spans four
  printed bands).
- The verifier discarded nothing: no invented or altered label was proposed.

## Safety facts

- The deterministic engine remained the only source of regulatory values; every value in the
  analysis comes from the unchanged engine running a canonical question.
- No live answer changed (every trigger question was re-asked after its call and matched).
- Nothing was deployed; the live chatbot never loaded this package.

## Spend and provenance

- **Total spend ≈ $0.42:** run $0.4101 (33 calls), one provider-acceptance call $0.0126, one
  1-token diagnostic < $0.001. Two earlier attempts were void at $0 (HTTP 429: a Windows
  user environment variable held a key of an unbilled project and overrode `.env`), and one
  stopped at $0 on HTTP 400 (the original schema enumerated all 217 labels: "too many
  states for serving").
- **Commits:** `6c4ef01` (experiment frozen), `6d83df3` (line-ending-independent hashes),
  `aa43b1d` (flaky unknown-route test fixed before the run), `f140402` (row labels as plain
  strings in the response schema, verifier unchanged).
- **Manifest:** `eval/normalizer/normalizer1.yaml`, sha256
  `803b9364a5ccffcf1ca5f66e5f2164532da6b6a6c9cadb0fa4952a2bb5758958`.
- **Outputs (not committed, by policy):** `data/normalizer/run1/`,
  `data/normalizer/run1_analysis.json`; void attempts in `data/normalizer/run1_void_429/` and
  `run1_void_429b/`.
