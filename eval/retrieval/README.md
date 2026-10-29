# Retrieval evaluation sets

Ten frozen question sets (2,394 questions). Each `final*` set was authored by an isolated
agent, reviewed independently, frozen with a `.sha256`, and run **once** against a pinned
candidate. After its run a set becomes development data; its recorded result is never re-run
or re-scored. The full procedure is in [`FINAL_TEST_PROCEDURE.md`](FINAL_TEST_PROCEDURE.md).

| Set | Questions | Run after |
|---|---|---|
| `dev` | 76 | development set |
| `heldout` | 117 | development set |
| `final` | 152 | first hybrid baseline (recorded in `final_baseline.yaml`) |
| `final2` | 205 | routing v3 / answerability v3.1 |
| `final3` | 275 | structured safety cycle |
| `final4` | 344 | routing v4 |
| `final5` | 302 | source-structure identity and source-attested constraints |
| `final6` | 343 | plan coverage and evidence certification |
| `final7` | 295 | Safety Closure 2 |
| `final8` | 285 | Safety Closure 3, the last planned deterministic final |

No final run has been promoted yet; Final-8 failed on two meaning-dependent questions.

## Other files

| File | Purpose |
|---|---|
| `final*_candidate.yaml` | Pins of the code, data and evaluator a final ran against |
| `final4_adjudication.yaml`, `adjudications.yaml` | Reviewed decisions on disputed cases |
| `dev_baselines.yaml`, `final_baseline.yaml`, `hybrid_candidate.yaml` | Recorded baselines |
| `scope_map.yaml`, `structured_schemas.yaml` | Generated scope map and table schemas |
| `external_domains.yaml`, `policy_ambiguities.yaml` | Out-of-scope domains and policy questions |

## Safety metrics

Reported before any quality metric. Promotion needs all of them at 0: wrong regulatory
values, wrong extra rows, dropped constraints and certification failures.
