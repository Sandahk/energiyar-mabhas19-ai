# SH0 pilot manifest

`pilot1.yaml` (checksum in `pilot1.sha256`) freezes the 80 questions of the first SH0 pilot.
It was frozen before any model output existed. **Do not edit it**; a change to the pinned
retrieval or shadow modules requires a new manifest.

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

Check it with `uv run python -m mabhas19.shadow.pilot verify`.
