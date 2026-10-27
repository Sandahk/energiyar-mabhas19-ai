# Retrieval performance: technical debt after Safety Closure 2

Status: recorded debt only. Nothing below is implemented and runtime behaviour is unchanged.

## 1. Measurement (2026-10-27)
- Eight-set development gate: about 0.9 s/query and about 45 minutes per run, against about
  10 minutes before Safety Closure 2.
- Profile of 58 Final-6 queries (`cProfile`, `PYTHONHASHSEED=4711`): 29.6 s in total, 22.1 s
  of it under `planner.identify` → `identity.named` → `identity.query_forms`, which runs once
  per query and catalogue row.

| Caller | Cumulative time | Note |
|---|---|---|
| `router.decompose` (identity via the planner dry run) | 13.9 s | existed before Closure 2 |
| `coverage.named_rows_words` (I3 named-row accounting) | 6.0 s | new in Closure 2 |
| `coverage.coordinated_members` (I8 generic coordination) | 5.5 s | new in Closure 2 |
| `coverage.evaluate` (rest of the coverage stage) | about 0.7 s | |

The dominant new cost is repeated identity-index work: coordination re-identifies each member
segment, and named-row accounting re-identifies the query against every row of every table it
might name. Both work over the same query forms that the router and planner have already
computed.

## 2. Constraints on any future optimisation
- No heuristic prefilters, approximate caches or reduced candidate searches. A shortcut that
  can skip a row can skip an identity, a family or a scope violation, which is the safety
  surface.
- An optimisation is admissible only if tests prove it result-identical:
  - cache vs no cache (or optimised vs reference execution) yields identical
    `RetrievalResult`s, ledgers and certification outcomes;
  - the comparison runs on all eight development sets and the coverage property suites
    (P1-P19).
- Any change to a pinned module invalidates the frozen candidate, so optimisation happens
  between benchmarks and never against a frozen manifest.

## 3. Likely direction (not designed)
Compute the per-query identity forms once (`query_forms` and the named positions per row) and
share that exact, query-keyed result across the router, the planner, coordination and
named-row accounting. Prove it identical against the current uncached path before adopting it.
