# SH0 pilot manifests

| File | Holds |
|---|---|
| `pilot1.yaml` (+ `.sha256`) | The 80 cases, queries, gold labels, expected results and context hashes. Its own pins describe the Anthropic request at freeze time |
| `pilot1_gemini.yaml` (+ `.sha256`) | The Gemini request contract (`gemini-3.8-flash`) and the live engine; refers to `pilot1.yaml` by its SHA-256 and reads the cases from it |

Both were frozen before any model output existed. **Do not edit them.** A provider manifest never
copies the cases, so a provider change cannot alter the sample or the gold. Any change to a
pinned retrieval or shadow module requires a new provider manifest; a later Claude comparison
would get its own `pilot1_anthropic.yaml` over the same base.

## Contents

- **Pins:** engine fingerprint, prompt hash, canonical and provider schema hashes, projection
  and context versions, proposer settings.
- **Per question:** source set and ID, query, normalized query, candidate-context hash,
  stratum, expected rows and outcomes, whether production is currently correct, and gold
  labels for clauses, entities and values.

| Stratum | Questions |
|---|---|
| Homograph / possessive (incl. Final-8 Q099, Final-7 P179) | 12 |
| Background / claim / request (incl. Final-8 Q264, Final-7 P011) | 12 |
| Meaningful modifier | 10 |
| Semantic or colloquial, fail-closed | 14 |
| Routing / intent | 12 |
| Controls: true identity, verification claim, background constraint, two results, colloquial entity (4 each) | 20 |

Questions were drawn by a seeded deterministic selection from the ten existing development
sets. Gold labels come from an isolated author and an independent reviewer over two rounds;
nine poor-fit cases were replaced in round two.

Known gap: P269's immutable development record lacks an out-of-scope outcome code, so its
expected result is incomplete.

Check it with `uv run python -m mabhas19.shadow.pilot verify` (the Gemini manifest is the
default).
