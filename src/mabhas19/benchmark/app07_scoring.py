"""Offline, reproducible scoring of appendix 7 extractions (Phase 2B).

Scores cached Gemini responses (whole page and row crops) and, optionally, local Tesseract
runs against the human-verified appendix 7 references. Never calls an external API: it
only reads cached response files under data/interim/gemini-phase2b/<model>/responses/.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mabhas19.benchmark import phase2b as pb
from mabhas19.benchmark.extractors import parse_density
from mabhas19.benchmark.metrics import label_match
from mabhas19.benchmark.references import PageReference, ReferenceItem
from mabhas19.text.normalize import normalize_for_comparison
from mabhas19.text.numbers import parse_numeric, same_value

# Standard-tier prices for gemini-3.8-flash, USD per token, valid through 2026-12-31
# (thinking tokens are billed as output). Source: ai.google.dev/gemini-api/docs/pricing.
PRICE_INPUT_PER_TOKEN = 0.75 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 3.75 / 1_000_000

APP07_PAGE_IDS = ("app07-p003", "app07-p006", "app07-p013")
LABEL_FIELDS = ("category", "subgroup", "material", "condition")
# Reference values that are NOT printed on the scored page (inherited from earlier pages,
# per the 2026-10-06 human review): page 6 continues group 2 from page 3 and the
# «بتن با سنگدانهٔ بسیار سبک» sub-group / material from page 5.
NOT_ON_PAGE: frozenset[tuple[str, str]] = frozenset(
    {(f"app07-p006-r{i:02d}", "category") for i in range(1, 15)}
    | {
        ("app07-p006-r01", "subgroup"),
        ("app07-p006-r02", "subgroup"),
        ("app07-p006-r01", "material"),
    }
)
_FOOTNOTE_TAIL = re.compile(r"\s*[:：]?\s*[۰-۹0-9¹²³]*\s*[:：]?\s*$")


def lenient_label(text: str | None) -> str | None:
    """Comparison form ignoring a leading list dash, a trailing colon and footnote markers."""
    if text is None:
        return None
    body = text.strip().removeprefix("-").strip()
    body = re.sub(r"[¹²³]", "", body)
    body = _FOOTNOTE_TAIL.sub("", body)
    return normalize_for_comparison(body) or None


# ------------------------------------------------------------------- numeric scoring


@dataclass
class NumericScore:
    """Counters for one method over a set of reference rows."""

    method: str
    counts: Counter[str] = field(default_factory=Counter)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def ratio(self, ok: str, total: str) -> str:
        t = self.counts[total]
        return f"{self.counts[ok]}/{t}" if t else "n/a"

    def summary(self) -> dict[str, str]:
        return {
            "numeric_exact": self.ratio("num_ok", "num_total"),
            "rows_fully_numeric": self.ratio("rows_ok", "rows_total"),
            "decimal_preserved": self.ratio("dec_ok", "dec_total"),
            "integer_conductivity": self.ratio("int_ok", "int_total"),
            "five_digit_density": self.ratio("five_ok", "five_total"),
            "ranges": self.ratio("range_ok", "range_total"),
            "qualifiers": self.ratio("qual_ok", "qual_total"),
        }


def _density_kind_ok(qualifier: str | None, kind: str | None) -> bool:
    if qualifier == "بیش از":
        return kind == "more_than"
    if qualifier == "کمتر از":
        return kind == "less_than"
    return kind in ("range", "single")


def _score_conductivity(
    score: NumericScore, ref: ReferenceItem, row: pb.ExtractedRow | None
) -> bool:
    c, expected = score.counts, ref.expected_fields["conductivity"]
    raw = row.conductivity_raw if row else None
    value = parse_numeric(raw or "", slash_is_decimal=True).normalized_value if row else None
    ok = same_value(expected, value)
    c["num_total"] += 1
    c["num_ok"] += ok
    if "." in expected:
        c["dec_total"] += 1
        c["dec_ok"] += ok and bool(pb.decimal_separator_ok(expected, raw))
    else:
        c["int_total"] += 1
        c["int_ok"] += ok
    if not ok:
        score.errors.append(_err(ref.id, "conductivity", score.method, (expected, raw, value)))
    return ok


def _score_density(score: NumericScore, ref: ReferenceItem, row: pb.ExtractedRow | None) -> bool:
    """Both density bounds plus the range / qualifier categories; True if all correct."""
    c, f = score.counts, ref.expected_fields
    raw = row.density_raw if row else None
    dens = parse_density(raw or "", "high") if row else None
    bounds_ok = True
    for bound in ("min", "max"):
        expected = f[f"density_{bound}"]
        got = getattr(dens, bound) if dens else None
        value = got.normalized_value if got else None
        if expected is None:
            if value is not None:  # e.g. an upper bound invented for «بیش از»
                bounds_ok = False
                score.errors.append(
                    _err(ref.id, f"density_{bound}", score.method, (None, raw, value))
                )
            continue
        ok = same_value(expected, value)
        c["num_total"] += 1
        c["num_ok"] += ok
        if len(expected) == 5:
            c["five_total"] += 1
            c["five_ok"] += ok
        if not ok:
            bounds_ok = False
            score.errors.append(
                _err(ref.id, f"density_{bound}", score.method, (expected, raw, value))
            )
    qualifier = f.get("density_qualifier")
    kind_ok = _density_kind_ok(qualifier, dens.kind if dens else None)
    if qualifier is None and f["density_min"] != f["density_max"]:
        c["range_total"] += 1
        c["range_ok"] += kind_ok and bounds_ok
    if qualifier is not None:
        c["qual_total"] += 1
        c["qual_ok"] += kind_ok and bounds_ok
    return bounds_ok


def score_numeric(
    method: str, refs: Iterable[ReferenceItem], found: dict[str, pb.ExtractedRow]
) -> NumericScore:
    """Exact numeric scoring; values are re-parsed from the *raw* text, never trusted."""
    score = NumericScore(method)
    for ref in refs:
        row = found.get(ref.id)
        k_ok = _score_conductivity(score, ref, row)
        d_ok = _score_density(score, ref, row)
        score.counts["rows_total"] += 1
        score.counts["rows_ok"] += k_ok and d_ok
    return score


def _err(item: str, name: str, method: str,
         seen: tuple[str | None, str | None, str | None]) -> dict[str, Any]:  # fmt: skip
    """seen = (source value, extracted raw text, extracted parsed value)."""
    source, raw, value = seen
    return {"item": item, "field": name, "source": source, "extracted_raw": raw,
            "extracted": value, "method": method}  # fmt: skip


# --------------------------------------------------------------------- label scoring


@dataclass
class LabelScore:
    counts: Counter[str] = field(default_factory=Counter)
    context: Counter[str] = field(default_factory=Counter)
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for fld in (*LABEL_FIELDS, "all"):
            t = self.counts[f"{fld}_total"]
            out[fld] = {"strict": f"{self.counts[f'{fld}_strict']}/{t}",
                        "lenient": f"{self.counts[f'{fld}_lenient']}/{t}"}  # fmt: skip
        out["present_in_row"] = f"{self.counts['present_ok']}/{self.counts['present_total']}"
        out["context"] = dict(self.context)
        return out


def score_labels(refs: list[ReferenceItem], raw_rows: dict[str, dict[str, Any]]) -> LabelScore:
    """Label/structure accuracy of a whole-page extraction's *_raw fields.

    - strict: exact after comparison normalization; lenient: also ignores list dash,
      trailing colon and footnote markers;
    - an inherited material (same as the previous row) left empty is correct and counted
      as context, as is an empty value for a field not printed on the page (NOT_ON_PAGE);
    - present_in_row: the expected text occurs in any raw label field of the row
      (detects text placed in the wrong field rather than misread).
    """
    score = LabelScore()
    c = score.counts
    prev_material: str | None = None
    for ref in refs:
        f = ref.expected_fields
        raw = raw_rows.get(ref.id, {})
        blob = lenient_label(" ".join(
            str(raw.get(f"{k}_raw")) for k in LABEL_FIELDS if raw.get(f"{k}_raw")
        )) or ""  # fmt: skip
        for fld in LABEL_FIELDS:
            expected, got = f.get(fld), raw.get(f"{fld}_raw")
            inherited = fld == "material" and expected is not None and expected == prev_material
            if inherited and got is None:
                score.context["inherited_not_repeated"] += 1
                continue
            if (ref.id, fld) in NOT_ON_PAGE:
                score.context["absent_correctly" if got is None else "filled_from_elsewhere"] += 1
                continue
            strict = label_match(expected, _str(got))
            lenient = lenient_label(expected) == lenient_label(_str(got))
            for key in (fld, "all"):
                c[f"{key}_total"] += 1
                c[f"{key}_strict"] += strict
                c[f"{key}_lenient"] += lenient
            if expected is not None and not inherited:
                c["present_total"] += 1
                c["present_ok"] += (lenient_label(expected) or "") in blob
            if not lenient:
                score.mismatches.append({"item": ref.id, "field": fld, "source": expected,
                                         "extracted": got})  # fmt: skip
        prev_material = f.get("material")
    return score


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


# --------------------------------------------------------------------- cached results


def load_response(responses_dir: Path, request_id: str) -> dict[str, Any] | None:
    path = responses_dir / f"{request_id}.json"
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@dataclass
class GeminiApp07:
    """Cached Gemini appendix 7 extractions aligned to reference rows."""

    page: dict[str, pb.ExtractedRow] = field(default_factory=dict)
    page_raw: dict[str, dict[str, Any]] = field(default_factory=dict)
    crop: dict[str, pb.ExtractedRow] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)


def collect_gemini(responses_dir: Path, page_id: str, refs: list[ReferenceItem]) -> GeminiApp07:
    out = GeminiApp07()
    resp = load_response(responses_dir, f"{page_id}.page")
    if resp and resp.get("status") == "ok":
        data = pb.parse_json(resp.get("text"))
        rows = data.get("rows", []) if isinstance(data, dict) else []
        extracted = pb.rows_from_json(data)
        aligned = pb.align_rows(refs, extracted)
        index = {id(r): k for k, r in enumerate(extracted)}
        out.page = aligned
        out.page_raw = {item: rows[index[id(row)]] for item, row in aligned.items()}
        matched = {index[id(r)] for r in aligned.values()}
        out.detection = {
            "expected": len(refs), "returned": len(extracted), "matched": len(aligned),
            "missing": [r.id for r in refs if r.id not in aligned],
            "extra": [k + 1 for k in range(len(extracted)) if k not in matched],
        }  # fmt: skip
    for item in refs:
        resp = load_response(responses_dir, f"{item.id}.crop")
        data = pb.parse_json(resp.get("text")) if resp and resp.get("status") == "ok" else None
        if isinstance(data, dict):
            out.crop[item.id] = pb.rows_from_json({"rows": [data]})[0]
    return out


def usage_summary(responses_dir: Path, request_ids: Iterable[str]) -> dict[str, Any]:
    """Status counts, attempt errors, latency, tokens and cost of cached requests."""
    status: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    by_kind: dict[str, Counter[str]] = {}
    latency: dict[str, list[float]] = {}
    for rid in request_ids:
        resp = load_response(responses_dir, rid)
        if resp is None:
            status["not_attempted"] += 1
            continue
        status[resp["status"]] += 1
        errors.update(a["error"] for a in resp.get("attempts", []) if a.get("error"))
        if resp["status"] != "ok":
            continue
        kind = rid.rsplit(".", 1)[-1]
        tokens = by_kind.setdefault(kind, Counter())
        tokens["requests"] += 1
        tokens.update({k: v or 0 for k, v in resp.get("usage", {}).items()})
        latency.setdefault(kind, []).append(float(resp.get("latency_s") or 0))
    kinds = {}
    for kind, t in by_kind.items():
        cost = request_cost(t["prompt_tokens"], t["output_tokens"], t["thinking_tokens"])
        lat = latency[kind]
        kinds[kind] = {**dict(t), "cost_usd": round(cost, 4),
                       "mean_latency_s": round(sum(lat) / len(lat), 1),
                       "min_latency_s": min(lat), "max_latency_s": max(lat)}  # fmt: skip
    total = round(sum(k["cost_usd"] for k in kinds.values()), 4)
    return {"status": dict(status), "attempt_errors": dict(errors), "by_kind": kinds,
            "total_cost_usd": total}  # fmt: skip


def request_cost(input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    return (input_tokens * PRICE_INPUT_PER_TOKEN
            + (output_tokens + thinking_tokens) * PRICE_OUTPUT_PER_TOKEN)  # fmt: skip


def verified_rows(ref: PageReference) -> list[ReferenceItem]:
    return [i for i in ref.items if i.type == "material_row" and pb.is_verified(i, ref)]
