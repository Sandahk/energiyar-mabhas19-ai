"""Answer provenance over the existing development questions (frozen engine, BM25 prose
ranking; the structured path is identical in every prose mode).

For every question of ``dev`` and ``heldout``: no provenance violation, no value without a
certified row, every certified row accounted for, and a structured answer always carries its
interpretation echo and its citation.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import pytest
import yaml

from mabhas19.core.paths import CURATED_DIR, EVAL_DIR
from mabhas19.generation.composer import compose
from mabhas19.generation.provenance import numbers
from mabhas19.indexing.chunking import build_chunks
from mabhas19.retrieval.hybrid import Retriever
from mabhas19.retrieval.structured import StructuredStore


@cache
def _store() -> StructuredStore:
    return StructuredStore(CURATED_DIR)


@cache
def _retriever() -> Retriever:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    blocks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return Retriever(build_chunks(blocks, "m19-ed5-1404-main"), _store())


def _cases() -> list[dict[str, Any]]:
    out = []
    for name in ("dev", "heldout"):
        doc = yaml.safe_load((EVAL_DIR / "retrieval" / f"{name}.yaml").read_text(encoding="utf-8"))
        out += doc["cases"]
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: str(c["id"]))
def test_every_development_answer_is_provenance_clean(case: dict[str, Any]) -> None:
    result = _retriever().retrieve(case["query"])
    answer = compose(result, _store(), "p")
    assert answer.safety["provenance_violations"] == []
    certified = [h.record_id for h in result.structured]
    if not certified:
        assert not answer.values
        assert answer.response_type != "structured_value"
        return
    assert answer.response_type == "structured_value"
    assert [v.record_id for v in answer.values] + [] == certified[: len(answer.values)]
    assert len(answer.values) + answer.hidden_values == len(certified)
    assert answer.interpretation, "a structured answer always shows its interpretation"
    for v in answer.values:
        assert v.source is not None
        assert v.source.printed_pages
        asked_or_printed = numbers(result.query) | {
            n
            for h in result.structured
            if h.record_id == v.record_id
            for n in numbers(json.dumps(h.values, ensure_ascii=False) + str(h.unit))
        }
        assert numbers(v.value) <= asked_or_printed
