# `mabhas19.shadow` (SH0)

A language model in **shadow mode**: it proposes how to read a question, deterministic code
verifies the proposal, and the result is only measured. Production output is never changed.

**Status: closed, not promoted.** Built and tested offline; shadow only; nothing here affects a
production result.

- The 80-question Gemini pilot ran once: safety held, but the original veto policy was too
  aggressive and harmed correct controls.
- V1 (`s1v-grammatical-veto-1`) narrowed authority to an explicit, verified `grammatical` label.
- On `veto1`, Gemini 3.8 Flash was safe (0 harmful, 100% precision) but recall was 10/40 = 25%,
  below the 40% threshold. `gemini-4-argon` passed every threshold (recall 19/40 = 47.5%), which
  made `veto1` model-selection evidence.
- On `s1v_holdout1`, a fresh holdout with a documented limited isolation exception, Argon with
  V1 made 0 harmful vetoes at 100% precision (15/15), but recall was 15/67 = 22.4%. S1-V is not
  promoted and the line is closed. S1-R was never started.

**Stopping decision:** semantic veto with the tested LLM configurations did not reach the
required recall at the required precision. The deterministic safety engine remains the
production boundary. No result-affecting LLM semantic layer is approved. On the final holdout,
nested-span abstention was necessary to avoid two harmful vetoes, and all 52 missed
opportunities were semantic-label misses (`contextual`), so the limit is semantic usefulness,
not verifier safety.

Both providers receive the same instructions and the same rendered candidate context; only the
request envelope and the schema projection differ. The canonical parser and the verifier
decide, whatever the provider.

## Flow

```
question → production retrieval (unchanged, returned as is)
         → candidate context (clauses, entity candidates, stated values, tables; redacted)
         → model proposal (labels and choices among context items only)
         → verifier (field by field, against the deterministic facts)
         → restriction-only shadow plan → counterfactual run through the real pipeline
         → metrics (would it remove a wrong row? would it remove a right one?)
```

A proposal can only **restrict**: mask a false entity reading, drop an unasked claim, remove a
table, or force a fail-closed answer. A counterfactual that would return a new row is refused
and logged as a blocked recovery.

## Modules

| Module | Role |
|---|---|
| `context.py` | Candidate context sent to the model, and the facts kept back for verification; same-length redaction |
| `proposal.py` | Canonical proposal schema `sh0-proposal-1` and its strict parser |
| `schema_projection.py` | Deterministic projections of the canonical schema onto Anthropic's and Gemini's structured-output subsets |
| `verifier.py` | Checks every proposal field; a proposal is never evidence |
| `translate.py` | Verified proposal → restriction-only plan → counterfactual |
| `run.py` | One shadow run beside production |
| `replay.py` | Record/replay keys (provider and exact model included) and records; request rendering and prompt hash |
| `provider.py` | Text-only `GeminiProposer` (REST) and `AnthropicProposer`, and the external-call guards |
| `metrics.py` | Veto (S1-V), recovery (S1-R) and preservation metrics; S1-V criteria check |
| `veto.py` | Candidate S1-V policy `s1v-grammatical-veto-1` (one lever: a verifier-accepted `grammatical` label masks a grounded span) and the veto-benchmark metrics |
| `pilot.py` | Pilot CLI: `verify`, `preflight`, `count-tokens`, `run`, `report` |

## External-call guards

A provider call happens only when **all** of these hold: `--allow-external`, the provider's
credential in the environment (`GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN`), a frozen provider manifest that verifies against the live engine, a
positive dollar and token budget, and an exact approved model for that provider. There is no
fallback to another model or provider. A Gemini call is also refused once its pinned price has
expired (Flash 2026-12-31, Argon 2026-11-30). Each recorded case is called once and replayed
afterwards; invalid outputs are recorded, never regenerated, and a token-capped response is never
resumed. One transient retry is allowed and recorded in the
case's usage metadata. The run stops on an HTTP 400 (the request shape itself is rejected), on
a response from another model, at the budget, or when invalid outputs exceed 10% after 20
calls.

```sh
uv run python -m mabhas19.shadow.pilot verify      # local
uv run python -m mabhas19.shadow.pilot preflight   # local
uv run python -m mabhas19.shadow.pilot report      # replay only
# count-tokens and run make network calls and need explicit approval
```
