# SH0 pilot manifests

| File | Holds |
|---|---|
| `pilot1.yaml` (+ `.sha256`) | The 80 cases, queries, gold labels, expected results and context hashes. Its own pins describe the Anthropic request at freeze time |
| `pilot1_gemini.yaml` (+ `.sha256`) | The Gemini request contract of the pilot, which ran once. Closed: it no longer verifies against the live engine, because the shadow code has changed since |
| `veto1.yaml` (+ `.sha256`) | The unseen benchmark for the candidate S1-V policy `s1v-grammatical-veto-1`: cases, gold labels, each case's production rows and expected V1 rows. Not a Final-N set |
| `veto1_gemini.yaml` (+ `.sha256`) | The `gemini-3.8-flash` request contract of that benchmark, which ran once: safe (0 harmful vetoes, 100% precision) but recall 10/40 = 25%, below the 40% threshold. Closed: it no longer verifies against the live engine, because the shadow code has changed since |
| `veto1_argon.yaml` (+ `.sha256`) | The `gemini-4-argon` request contract for one model comparison on the same cases: only the model and the client timeout differ from Flash |

Each benchmark and its first manifest were frozen before any model output existed. **Do not edit
them.** A provider manifest never copies the cases, so a provider change cannot alter the sample
or the gold. Any change to a pinned retrieval or shadow module requires a new provider manifest;
a later Claude comparison would get its own `pilot1_anthropic.yaml` over the same base.

## `veto1`

| Group | Cases | Purpose |
|---|---|---|
| A grammatical misread | 40 | Production returns a wrong row because a word used grammatically, possessively or relationally matched a printed label |
| B identity lookalike | 69 | Hard controls: a genuine entity whose name holds a word that could be grammatical elsewhere |
| C plain identity | 49 | Ordinary controls |
| D ambiguous | 10 | No safe decision: a veto here earns no credit and lowers precision, but is not counted as harmful |

Two isolated authors wrote the cases with the deterministic engine as a probe; two independent
reviewers labelled them blind, with one adjudication round. A label stands only where author and
reviewer agree, and a surviving disagreement drops the case. Gold labels were frozen before any
provider call.

## `pilot1` contents

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

Check the current benchmark with `uv run python -m mabhas19.shadow.pilot verify`
(`veto1_argon.yaml` is the default).
