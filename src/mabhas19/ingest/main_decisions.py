"""Apply the main-text review decisions and assign review status and retrieval eligibility.

Decisions (``eval/extraction/review/main-assistant-review.yaml``):

* ``replace``: exact substring replacements in one block's repaired text, guarded by the
  sha256 of the text they expect and by exact occurrence counts (stale -> refused);
* ``exclude_from_retrieval``: the block stays in the corpus as provenance only;
* ``merge_formula``: fragments of one formula become one structured formula block; the
  fragments are kept (excluded from retrieval) with ``merged_into``;
* ``classify``: structural class of unnumbered bold lines.

Raw text and the repair_v1 text are never changed. Every block gets ``verification``
(status, basis, rules, decisions), ``retrieval`` (included or not, with a reason) and the
list of review flags it still carries.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import EVAL_DIR
from mabhas19.ingest.main_blocks import quality_flags
from mabhas19.text.normalize import normalize_for_search

DECISIONS_PATH = EVAL_DIR / "extraction" / "review" / "main-assistant-review.yaml"
ERRATA_PATH = EVAL_DIR / "extraction" / "review" / "main-errata-review.yaml"
# B2/L1/T1 are applied only to the reviewed cases listed in the errata file, never generically
APPROVED_RULES = frozenset({"M1", "M1-exception", "G1", "G2", "B1", "C1", "B2", "L1", "T1"})
EXTRACTION_FIXES = frozenset({"K1"})  # classification fixes of the extractor itself
NOT_RETRIEVED_KINDS = {
    "running_header": "page furniture",
    "page_number": "page furniture",
    "toc_entry": "table of contents (duplicates the headings)",
    "table": "pending structured extraction (cells are data, not prose)",
    "figure": "image region, not transcribed",
    "formula_fragment": "merged into one formula",
}
STATUS_BY_KIND = {
    "table": "pending_structured_extraction",
    "figure": "not_transcribed",
    "running_header": "page_furniture",
    "page_number": "page_furniture",
}


class DecisionError(ValueError):
    """A decision targets an unknown block or no longer matches the corpus (stale)."""


def load_decisions(path: Path = DECISIONS_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {"decisions": [], "basis_default": "assistant_reviewed"}
    doc: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return doc


def load_errata(path: Path = ERRATA_PATH) -> dict[str, Any]:
    return load_decisions(path)


def _apply_errata(block: dict[str, Any], d: dict[str, Any]) -> None:
    """One guarded errata entry: expected text, exact counts, expected result."""
    text = block["text"]
    if _sha(text) != d["expect_text_sha256"]:
        raise DecisionError(f"{d['id']}: {block['block_id']} text changed since the review (stale)")
    for r in d["replacements"]:
        if text.count(r["from"]) != r["count"]:
            raise DecisionError(f"{d['id']}: «{r['from']}» found {text.count(r['from'])} times, "
                                f"expected {r['count']}")  # fmt: skip
        text = text.replace(r["from"], r["to"])
        if r["rule"]:
            block.setdefault("applied_rules", []).append(
                {"rule": r["rule"], "count": r["count"], "layer": "errata", "decision": d["id"]}
            )
    if _sha(text) != d["result_text_sha256"]:
        raise DecisionError(f"{d['id']}: result differs from the reviewed text")
    if text != block["text"]:
        block["text_before_errata"] = block["text"]
        block["text"] = text
        block["text_normalized"] = normalize_for_search(text)
    reviewed = [r for r in d["replacements"] if not r["rule"]]
    if reviewed or "reclassify" in d:
        _note(block, {"id": d["id"], "decision": "errata"}, "assistant_reviewed")
    if "reclassify" in d:
        block["kind"] = d["reclassify"]["kind"]
        block["caption_block"] = d["reclassify"]["caption_block"]
    block["errata"] = {"id": d["id"], "candidates": sorted(
        {c for r in d["replacements"] for c in r["candidates"]}
        | set(d.get("reclassify", {}).get("candidates", [])))}  # fmt: skip


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _note(block: dict[str, Any], decision: dict[str, Any], basis: str) -> None:
    block.setdefault("decisions", []).append({"id": decision["id"], "basis": basis,
                                              "type": decision["decision"]})  # fmt: skip


def _replace(block: dict[str, Any], d: dict[str, Any], basis: str) -> None:
    text = block["text"]
    if _sha(text) != d["expect_text_sha256"]:
        raise DecisionError(f"{d['id']}: {block['block_id']} text changed since the review (stale)")
    for r in d["replacements"]:
        if text.count(r["from"]) != r["count"]:
            raise DecisionError(f"{d['id']}: «{r['from']}» found {text.count(r['from'])} times, "
                                f"expected {r['count']}")  # fmt: skip
        text = text.replace(r["from"], r["to"])
    block["text_before_review"] = block["text"]
    block["text"] = text
    block["text_normalized"] = normalize_for_search(text)
    _note(block, d, basis)


def _merge_formula(blocks: list[dict[str, Any]], d: dict[str, Any], basis: str) -> None:
    by_id = {b["block_id"]: b for b in blocks}
    parts = [by_id[i] for i in d["blocks"]]
    for p in parts:
        if p["text"] != d["expect_texts"][p["block_id"]]:
            raise DecisionError(f"{d['id']}: fragment {p['block_id']} changed (stale)")
    first = parts[0]
    x0 = min(p["bbox"][0] for p in parts)
    y0 = min(p["bbox"][1] for p in parts)
    x1 = max(p["bbox"][2] for p in parts)
    y1 = max(p["bbox"][3] for p in parts)
    merged = {**{k: first[k] for k in ("doc_id", "pdf_page", "printed_page",
                                       "printed_page_confidence", "page_role", "section_path")},
              "block_id": d["new_block"], "kind": "formula", "section_number": None,
              "extraction_method": "merged_formula_fragments",
              "text_raw": "\n".join(p["text_raw"] for p in parts),
              "text_repair_v1": "\n".join(p["text_repair_v1"] for p in parts),
              "text": d["formula"]["plain"], "text_normalized": normalize_for_search(
                  d["formula"]["plain"]), "formula": d["formula"],
              "fragments": [p["block_id"] for p in parts], "bbox": [x0, y0, x1, y1],
              "flags": [], "applied_rules": [], "confidence": "medium"}  # fmt: skip
    _note(merged, d, basis)
    for p in parts:
        p["kind"] = "formula_fragment"
        p["merged_into"] = d["new_block"]
        _note(p, d, basis)
    blocks.insert(blocks.index(parts[-1]) + 1, merged)


def apply_decisions(blocks: list[dict[str, Any]], doc: dict[str, Any],
                    errata: dict[str, Any] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    out = [copy.deepcopy(b) for b in blocks]
    by_id = {b["block_id"]: b for b in out}
    basis = doc.get("basis_default", "assistant_reviewed")
    for d in doc.get("decisions", []):
        kind = d["decision"]
        targets = [d["block"]] if "block" in d else list(d.get("classes", {})) + d.get("blocks", [])
        missing = [t for t in targets if t not in by_id]
        if missing:
            raise DecisionError(f"{d['id']}: unknown blocks {missing}")
        if kind == "replace":
            _replace(by_id[d["block"]], d, basis)
        elif kind == "exclude_from_retrieval":
            b = by_id[d["block"]]
            b["retrieval_exclusion"] = d["reason"]
            _note(b, d, basis)
        elif kind == "merge_formula":
            _merge_formula(out, d, basis)
        elif kind == "classify":
            for bid, cls in d["classes"].items():
                by_id[bid]["structural_class"] = cls
                _note(by_id[bid], d, basis)
        else:
            raise DecisionError(f"{d['id']}: unknown decision type {kind!r}")
    for d in (errata or {}).get("decisions", []):
        if d["block"] not in by_id:
            raise DecisionError(f"{d['id']}: unknown block {d['block']}")
        _apply_errata(by_id[d["block"]], d)
    for b in out:
        finalize(b)
    return out


RESOLVED_BY = {"unnumbered_bold_line": "structural_class", "ocr_scanned_region":
               "retrieval_exclusion"}  # fmt: skip
STATUS_FLAGS = frozenset({"table_pending_structured_extraction", "figure_not_transcribed"})


def finalize(b: dict[str, Any]) -> None:
    """Review status, retrieval eligibility and the flags still open on a block."""
    if b.get("text") is not None and b["kind"] not in ("ocr_text", "formula"):
        keep = [f for f in b["flags"] if f == "unnumbered_bold_line"]
        flags = quality_flags(b["text"], b.get("text_raw") or "")
        if any(r["rule"] == "M1-exception" for r in b.get("applied_rules", [])):
            flags = [f for f in flags if f != "list_marker_paren_mirrored"]  # approved exception
        b["flags"] = flags + keep
    resolved = [f for f in b["flags"] if f in RESOLVED_BY and b.get(RESOLVED_BY[f])]
    b["resolved_flags"] = resolved
    b["open_flags"] = [f for f in b["flags"] if f not in resolved and f not in STATUS_FLAGS]
    rules = sorted({r["rule"] for r in b.get("applied_rules", [])})
    decided = b.get("decisions", [])
    if b["kind"] in STATUS_BY_KIND:
        status = STATUS_BY_KIND[b["kind"]]
    elif decided:
        status = "assistant_reviewed"
    elif set(rules) & APPROVED_RULES:
        status = "rule_corrected"
    else:
        status = "machine_extracted"
    basis = sorted({d["basis"] for d in decided} | ({"assistant_reviewed_rule"} if set(rules)
                   & APPROVED_RULES else set()) | ({"extraction_fix"} if set(rules)
                   & EXTRACTION_FIXES else set()))  # fmt: skip
    b["verification"] = {"status": status, "basis": basis, "rules": rules,
                         "decisions": [d["id"] for d in decided]}  # fmt: skip
    reason = b.get("retrieval_exclusion") or NOT_RETRIEVED_KINDS.get(b["kind"])
    if b.get("text") in (None, "") and reason is None:
        reason = "no text"
    b["retrieval"] = {"include": reason is None, "reason": reason}
