"""SI-0004 provenance links: guarded application and the curated result (read-only)."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest
import yaml

from mabhas19.core.paths import CURATED_DIR
from mabhas19.ingest.main_decisions import (
    PROVENANCE_MEANING,
    DecisionError,
    _apply_provenance,
    load_provenance,
)

APPROVED = {
    "main-p077-b004": "m19-ed5-1404-app05",
    "main-p078-b014": "m19-ed5-1404-app06",
    "main-p079-b006": "m19-ed5-1404-app06",
    "main-p082-b007": "m19-ed5-1404-app06",
    "main-p118-b008": "m19-ed5-1404-app08",
    "main-p146-b009": "m19-ed5-1404-app05",
    "main-p146-b010": "m19-ed5-1404-app06",
    "main-p146-b012": "m19-ed5-1404-app08",
}


def _blocks() -> list[dict[str, Any]]:
    path = CURATED_DIR / "text" / "m19-ed5-1404-main" / "blocks.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def _entry(text: str, page: int = 118, printed: str = "99") -> dict[str, Any]:
    return {
        "id": "PV-x",
        "block": "b",
        "expect_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "expect_pdf_page": page,
        "expect_printed_page": printed,
        "issue": "SI-0004",
        "references_doc": "m19-ed5-1404-app08",
        "references_appendix": 8,
        "reference_kind": "substantive",
        "basis": "assistant_reviewed_provenance",
    }


def test_provenance_adds_only_the_link_fields() -> None:
    block: dict[str, Any] = {
        "block_id": "b",
        "text": "ر) …",
        "pdf_page": 118,
        "printed_page": "99",
        "kind": "x",
    }
    before = copy.deepcopy(block)
    _apply_provenance(block, _entry("ر) …"))
    assert {k: block[k] for k in before} == before
    assert block["source_issue_refs"] == ["SI-0004"]
    (link,) = block["provenance_links"]
    assert link["references_doc"] == "m19-ed5-1404-app08"
    assert link["meaning"] == PROVENANCE_MEANING


def test_provenance_is_guarded() -> None:
    with pytest.raises(DecisionError):
        _apply_provenance({"block_id": "b", "text": "y", "pdf_page": 118, "printed_page": "99"},
                          _entry("x"))  # fmt: skip
    with pytest.raises(DecisionError):
        _apply_provenance({"block_id": "b", "text": "x", "pdf_page": 17, "printed_page": "ظ"},
                          _entry("x"))  # fmt: skip


def test_decision_file_pins_exactly_the_approved_anchors() -> None:
    decisions = load_provenance()["decisions"]
    assert {d["block"]: d["references_doc"] for d in decisions} == APPROVED
    assert {d["issue"] for d in decisions} == {"SI-0004"}


def test_curated_corpus_carries_si0004_on_exactly_the_approved_blocks() -> None:
    blocks = _blocks()
    carrying = {b["block_id"] for b in blocks if "SI-0004" in b.get("source_issue_refs", [])}
    assert carrying == set(APPROVED)
    by_id = {b["block_id"]: b for b in blocks}
    for bid, doc in APPROVED.items():
        (link,) = by_id[bid]["provenance_links"]
        assert (link["issue"], link["references_doc"]) == ("SI-0004", doc)
        assert by_id[bid]["retrieval"]["include"]  # eligibility unchanged
    for dup in ("main-p017-b024", "main-p017-b025", "main-p017-b027"):
        assert "source_issue_refs" not in by_id[dup]


def test_issue_registry_is_unchanged_by_the_links() -> None:
    issues = yaml.safe_load((CURATED_DIR / "source_issues.yaml").read_text(encoding="utf-8"))
    (si4,) = [i for i in issues["issues"] if i["id"] == "SI-0004"]
    assert {loc["doc_id"] for loc in si4["locations"]} == {
        "m19-ed5-1404-app05",
        "m19-ed5-1404-app06",
        "m19-ed5-1404-app08",
    }
