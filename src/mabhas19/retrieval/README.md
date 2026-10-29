# `mabhas19.retrieval`

Deterministic retrieval over the curated Mabhas 19 knowledge. **Frozen** at the Final-8
candidate (`eval/retrieval/final8_candidate.yaml`): changes here alter pinned module hashes and
invalidate the frozen candidate and the SH0 pilot manifest.

No language model is used anywhere in this package.

## Pipeline

| Stage | Module | Role |
|---|---|---|
| Normalization | `normalization.py` | Persian letters, digits, spacing and ZWNJ |
| Answerability | `answerability.py` | Recognises kinds of request Mabhas 19 does not answer |
| Scope | `scope.py`, `table_scope.py` | Which source domain and which tables a question can touch |
| Routing | `router.py` | `structured` / `prose` / `mixed` / `unsupported`, decided by dry-running the planner |
| Prose search | `lexical.py`, `vector.py`, `hybrid.py` | BM25 + multilingual-e5-small, reciprocal rank fusion |
| Identity | `identity.py`, `entities.py` | Which printed rows the question really names (no fuzzy matching) |
| Planning | `planner.py`, `plan.py`, `schema.py` | Question → typed `StructuredQueryPlan` per table; validation gate |
| Constraints | `constraints.py`, `numeric.py`, `units.py`, `dimensions.py`, `polarity.py`, `ratings.py`, `operations.py` | Values, intervals, units, negation, energy ratings, calculations |
| Admission | `admission.py` | Proves every plan component against an exact span of the question |
| Coverage | `coverage.py` | Cross-plan checks (invariants I1–I11): nothing read twice, nothing unasked |
| Execution | `executor.py`, `structured.py` | Selects rows only through admitted constraints; never broadens |
| Certification | `certification.py` | Independently re-checks every returned row before it leaves |

If any stage cannot fully account for the question, the result is a typed
`StructuredNoMatch` outcome rather than a best guess.

## Evaluation helpers

`eval.py`, `safety_metrics.py`, `final_test.py`, `final_validation.py`, `candidate.py` and
`identity_snapshot.py` score runs, compute the safety metrics (wrong values, extra rows,
dropped constraints, certification failures) and pin candidates before a final run.
See [`eval/retrieval/`](../../../eval/retrieval/README.md).

## Known limits

- Meaning-dependent readings (colloquial possessives, background remarks, claims to verify)
  cannot be told apart reliably by rules; this is what the SH0 shadow phase investigates.
- Runtime is about 0.9 s per question; the cost and a likely fix are recorded in
  [`docs/retrieval-performance.md`](../../../docs/retrieval-performance.md).
