# `mabhas19.shadow` (SH0)

A language model in **shadow mode**: it proposes how to read a question, deterministic code
verifies the proposal, and the result is only measured. Production output is never changed.

**Status:** built and tested offline. The 80-question pilot manifest is frozen; no model has
been called yet. S1-V and S1-R are not implemented.

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
| `schema_projection.py` | Deterministic projection of the canonical schema onto Anthropic's structured-output subset |
| `verifier.py` | Checks every proposal field; a proposal is never evidence |
| `translate.py` | Verified proposal → restriction-only plan → counterfactual |
| `run.py` | One shadow run beside production |
| `replay.py` | Record/replay keys and records; the request rendering and prompt hash |
| `provider.py` | Text-only `AnthropicProposer` (`claude-opus-5-5`) and the external-call guards |
| `metrics.py` | Veto (S1-V), recovery (S1-R) and preservation metrics; S1-V criteria check |
| `pilot.py` | Pilot CLI: `verify`, `preflight`, `count-tokens`, `run`, `report` |

## External-call guards

A provider call happens only when **all** of these hold: `--allow-external`, an
`ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in the environment, a frozen manifest that
verifies against the live engine, a positive dollar and token budget, and the exact model
`claude-opus-5-5`. Each recorded case is called once and replayed afterwards; invalid outputs
are recorded, never regenerated.

```sh
uv run python -m mabhas19.shadow.pilot verify      # local
uv run python -m mabhas19.shadow.pilot preflight   # local
uv run python -m mabhas19.shadow.pilot report      # replay only
# count-tokens and run make network calls and need explicit approval
```
