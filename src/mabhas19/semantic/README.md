# `mabhas19.semantic`: offline semantic row retrieval (closed, NO-GO)

**STATUS: CLOSED — NOT FOR PRODUCTION.** Historical experimental code, kept for the record.
Nothing in the chatbot, the API or the retrieval engine has ever imported this package, and the
shadow isolation test (`tests/unit/test_shadow.py`) forbids it. Do not tune it further, do not try
another embedding model, do not add a ranker, and do not author another benchmark for it.

## What was tested

Could a free, local semantic layer map everyday Persian row names («آشپزخونه», «درمانگاه»,
«XPS») onto printed table rows, once the regulatory quantity and table were fixed
deterministically?

```text
engine failed closed -> deterministic quantity (api/quantities.py, no embeddings)
  -> table from the quantity -> E5-small + character-trigram ranking of that table's
  printed rows only -> confirm one row / choose among 2-3 / abstain
  -> the user picks -> canonical printed question -> unchanged engine -> certification
```

- Model: the frozen `intfloat/multilingual-e5-small` already used by the engine ("query:" /
  "passage:"); no provider call and no spending in either run.
- Catalogue: the 217 printed rows of Appendix 3, tables 19-5-1 to 19-5-7 and Appendix 7.
- Retrieval never acted on a question stating a number, never chose a quantity, and every row it
  proposed had to be confirmed by the user; the value always came from the unchanged engine.
- Thresholds were tuned on development questions only and frozen, with the decision rule and the
  fresh set's hash, before each fresh run. Each fresh set was written by an isolated author from
  the printed catalogue, frozen unread, and run once.

## Results

| | B (first run) | B2 (final iteration) |
|---|---|---|
| Frozen at | `7c6386f` (fresh set `7532890`) | `e7ebf42` (fresh set `c887388`) |
| Usable row questions | **4/22** (A: 0/22) | **10/22 = 45%** (A: 1/22) |
| Wrong row offerings | **2** (limit 1) | **5** (limit 1) |
| Traps (similar-sounding unprinted quantity) blocked | 8/8 | 8/8 |
| Unsupported things declined | 10/10 | **8/10** (needed 9) |
| Wrong quantities | 0 | 0 |
| Wrong regulatory values | 0 | 0 |
| Right row in the top 3 (engaged) | 10/20 | 13/20 |
| Retrieval latency | ≤ 23 ms | ≤ 28 ms |
| Decision | **NO-GO** | **NO-GO** |

B2 differed from B by explicit extra quantity phrasings (`quantity.py`), a top-3 display instead
of abstaining when the quantity was certain, a sibling guard (no confirmation when a same-head row
is not told apart by the user's words) and a cross-table guard (a word printed in both tables of a
quantity brings each table's row). Of B2's 9 gains over A, 2 came only from the quantity
phrasings; the top-3 display gained nothing under the frozen attribution test and produced two of
the wrong offerings.

## Failure modes

- **XPS** → glass wool, stone wool, asphalt (the printed row is extruded polystyrene).
- **بلوک AAC** → earth/adobe and foam materials (the printed row is autoclaved aerated concrete).
- **کناف** → a single, high-confidence confirmation of the wrong gypsum row.
- **ماشین لباسشویی** (not a printed point of use) → three printed hot-water places offered.
- **ورق کامپوزیت آلومینیومی** (not printed) → printed aluminium and cement-board rows offered.
- First run: «کولر سلولزی» confirmed as «کولر گازی»; a skylight offered roof rows.

Safety held in both runs (0 wrong quantities, 0 wrong values, every trap blocked). The limit was
precision, not the verifier.

## Conclusion

> Semantic similarity is insufficiently precise for regulatory row identification even when
> quantity and table are deterministically constrained.

The production direction stays deterministic: wording normalization, quantity recognition, the
certified city/climate lookup, printed-row clarifications, side-by-side certified results, the
structured rescue, and the unchanged engine → certification → provenance.

Run outputs (`data/semantic/run1*.json`, `run2*.json`) are git-ignored and not committed. The
quantity phrasings are not part of semantic retrieval; whether any becomes a production rule in
`api/quantities.py` is a separate, gated decision.
