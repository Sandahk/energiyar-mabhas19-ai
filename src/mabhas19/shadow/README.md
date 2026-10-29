# `mabhas19.shadow` (SH0)

A language model in **shadow mode**: it proposes how to read a question, deterministic code
verifies the proposal, and the result is only measured. Production output is never changed.

**Status:** built and tested offline. The 80-question pilot is frozen; the first provider is
Gemini (`gemini-3.8-flash`), with Claude (`claude-opus-5-5`) kept as an optional later
comparison. No model has been called yet. S1-V and S1-R are not implemented.

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
| `pilot.py` | Pilot CLI: `verify`, `preflight`, `count-tokens`, `run`, `report` |

## External-call guards

A provider call happens only when **all** of these hold: `--allow-external`, the provider's
credential in the environment (`GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN`), a frozen provider manifest that verifies against the live engine, a
positive dollar and token budget, and the exact approved model for that provider. There is no
fallback to another model or provider. A Gemini call is also refused once its pinned price has
expired (2026-12-31). Each recorded case is called once and replayed afterwards; invalid
outputs are recorded, never regenerated. One transient retry is allowed and recorded in the
case's usage metadata. The run stops on an HTTP 400 (the request shape itself is rejected), on
a response from another model, at the budget, or when invalid outputs exceed 10% after 20
calls.

```sh
uv run python -m mabhas19.shadow.pilot verify      # local
uv run python -m mabhas19.shadow.pilot preflight   # local
uv run python -m mabhas19.shadow.pilot report      # replay only
# count-tokens and run make network calls and need explicit approval
```
