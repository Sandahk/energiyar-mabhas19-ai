"""Errata audit of the curated main corpus: candidate list and standalone HTML review package.

Usage:
    uv run python -m mabhas19.cli.audit_main_errata

Reads curated/text/m19-ed5-1404-main/blocks.jsonl (and the curated main tables' captions);
writes data/processed/main_errata/: ``candidates.jsonl``, ``summary.json``, ``crops/`` and
``review_main_errata.html`` (crops embedded). Assistant visual verdicts are read from
``data/processed/main_errata/assistant_verdicts.yaml`` when present. Nothing in curated/ is
changed and no external API is called.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from mabhas19.core.manifest import load_manifest
from mabhas19.core.paths import CURATED_DIR, DATA_DIR, MANIFEST_PATH, SOURCES_DIR
from mabhas19.ingest.main_decisions import ERRATA_PATH
from mabhas19.ingest.main_errata import (
    SKIP_KINDS,
    Finding,
    apply_proposals,
    audit_block,
    caption_continuations,
    errata_decisions,
    lexicon,
)
from mabhas19.tables.main_audit import TABLES

DOC_ID = "m19-ed5-1404-main"
OUT = DATA_DIR / "processed" / "main_errata"
VERDICTS = OUT / "assistant_verdicts.yaml"
DETERMINISTIC = {"B1", "B2", "T1", "L1"}
NOTES = [
    "Rules: B2 = mirrored «)…(» pair around Persian text -> «(…)» (all 107 crops checked); "
    "T1 = tanwin separated from its letter; L1 = two non-word fragments whose join is attested "
    ">=3x in the corpus. Everything else is a guarded per-block correction.",
    "Root cause of most split words: kashida-justified lines (kashida glyphs extracted as runs "
    "of «ل», removed by repair_v1 leaving a gap read as a space). A kashida-gap join in "
    "repair_v1 would be the durable deterministic fix.",
    "Not proposed (printed spacing, source-faithful): 540 space-separated compounds whose "
    "attested form is a ZWNJ compound (e.g. «شبیه سازی», «ساختمان ها»), plus the spaced forms "
    "confirmed in the crops («تا کنون», «فرو نشستگی», «به‌روز رسانی», «چرا که», «چند مولفه‌ای», "
    "«زیر ساخت‌های», «قدر السهم», «مستند سازی») and «حد اکثر», «خنک کن», «دودکش دار», «در صد», "
    "«غیر چگالشی», «بزرگتر».",
    "Printed misspelling, kept: table 19-5-2 caption «به روش تجویری» (crop shows «تجویری»; "
    "«تجویزی» elsewhere 78x) — possible source-issue candidate, not an extraction error.",
    "Captions changed by errata must later be synced into the structured tables' table.yaml "
    "(not modified here). main-p084-b004 is proposed as caption_continuation of main-p084-b003.",
    "Out of scope and untouched: page furniture, TOC entries, formula provenance, OCR text, "
    "table-cell data, source issues.",
]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def line_clip(b: dict[str, Any], found: str) -> pymupdf.Rect:
    """Approximate clip of the line holding ``found`` (blocks keep only their bbox)."""
    x0, y0, x1, y1 = b["bbox"]
    lines = b["text"].split("\n")
    first = found.split("\n", maxsplit=1)[0]
    idx = next((i for i, ln in enumerate(lines) if first in ln), 0)
    span = found.count("\n") + 1
    h = (y1 - y0) / max(1, len(lines))
    return pymupdf.Rect(x0 - 4, y0 + h * idx - 0.4 * h, x1 + 4, y0 + h * (idx + span) + 0.4 * h)


def citation_visible(b: dict[str, Any], titles: set[str]) -> bool:
    """Text a citation can show: retrievable block text, section titles, captions."""
    return bool(b["retrieval"]["include"]) or b["text"] in titles or "caption" in b["kind"]


def caption_checks(
    blocks: dict[str, dict[str, Any]], proposed: dict[str, str]
) -> list[dict[str, Any]]:
    """Corpus captions against the structured tables' table.yaml (which is not modified)."""
    out = []
    for spec in TABLES:
        meta_path = CURATED_DIR / "tables" / DOC_ID / f"table-{spec.number}" / "table.yaml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        corpus = " ".join(blocks[p]["text"] for p in meta["caption"]["blocks"])
        number = spec.number.replace("-", "")
        out.append(
            {
                "table": spec.number,
                "caption_blocks": meta["caption"]["blocks"],
                "same_text": corpus == meta["caption"]["text"],
                "table_number_in_caption": number
                in "".join(str("۰۱۲۳۴۵۶۷۸۹".find(c)) if c in "۰۱۲۳۴۵۶۷۸۹" else c for c in corpus),
                "corpus_text": corpus,
                "table_yaml_text": meta["caption"]["text"],
                "differs_after_errata": " ".join(
                    proposed.get(p, blocks[p]["text"]) for p in meta["caption"]["blocks"]
                )
                != corpus,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-decisions",
        action="store_true",
        help=f"write the approved errata decisions to {ERRATA_PATH.name}",
    )
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)
    blocks = _jsonl(CURATED_DIR / "text" / DOC_ID / "blocks.jsonl")
    by_id = {b["block_id"]: b for b in blocks}
    titles = {s["title"] for s in _jsonl(CURATED_DIR / "structure" / DOC_ID / "sections.jsonl")}
    verdicts = yaml.safe_load(VERDICTS.read_text(encoding="utf-8")) if VERDICTS.is_file() else {}
    verdicts = verdicts or {}
    freq = lexicon(blocks)
    extra: dict[str, list[Finding]] = {}
    for a in verdicts.get("additional_candidates", []):
        entry = {"verdict": "confirmed", "note": a["note"]}
        extra.setdefault(a["block"], []).append(
            Finding(
                a["kind"],
                a["found"],
                a["proposed"],
                "assistant targeted search; " + a["note"],
                extra={"verdict_entry": entry, "source": "assistant_targeted_search"},
            )
        )
    captions = caption_continuations(blocks)
    version = load_manifest(MANIFEST_PATH).get(DOC_ID).current
    candidates: list[dict[str, Any]] = []
    compounds = 0
    with pymupdf.open(SOURCES_DIR / version.path) as pdf:
        ctx = {"pdf": pdf, "titles": titles, "verdicts": verdicts}
        for b in blocks:
            if b["kind"] in SKIP_KINDS or not b.get("text"):
                continue
            found, n_compounds = audit_block(b, freq)
            compounds += n_compounds
            if b["block_id"] in captions:
                found.append(captions[b["block_id"]])
            found += extra.get(b["block_id"], [])
            for i, f in enumerate(found, 1):
                candidates.append(candidate(ctx, b, f, i))
    accepted: dict[str, list[Finding]] = {}
    for c in candidates:
        if c["verdict"] == "confirmed" and c["proposed"] is not None:
            accepted.setdefault(c["block_id"], []).append(c["_finding"])
    for c in candidates:
        text = by_id[c["block_id"]]["text"]
        c["found_in_text"] = c["proposed"] is None or text.count(c["found"]) >= 1
        fixes = accepted.get(c["block_id"], [])
        c["proposed_block_text"] = apply_proposals(text, fixes) if fixes else None
        del c["_finding"]
    with (OUT / "candidates.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in candidates:
            fh.write(
                json.dumps({k: v for k, v in c.items() if k != "crop_png"}, ensure_ascii=False)
                + "\n"
            )
    proposed = {
        c["block_id"]: c["proposed_block_text"]
        for c in candidates
        if c["proposed_block_text"] is not None
    }
    checks = caption_checks(by_id, proposed)
    summary = summarize(candidates, compounds, checks)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT / "review_main_errata.html").write_text(
        render(candidates, summary, checks), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.write_decisions:
        write_decisions(candidates, by_id, summary)
    return 0


def candidate(ctx: dict[str, Any], b: dict[str, Any], f: Finding, i: int) -> dict[str, Any]:
    """One candidate with its crop; ``ctx`` holds the PDF, section titles and verdicts."""
    pdf, titles, verdicts = ctx["pdf"], ctx["titles"], ctx["verdicts"]
    cid = f"E-{b['block_id'].removeprefix('main-')}-{i:02d}"
    x0, y0, x1, y1 = b["bbox"]
    clip = (
        pymupdf.Rect(x0 - 4, y0 - 4, x1 + 4, y1 + 4)
        if f.kind == "caption_continuation"
        else line_clip(b, f.found)
    )
    page = pdf[b["pdf_page"] - 1]
    png = page.get_pixmap(dpi=200, clip=clip & page.rect).tobytes("png")
    (OUT / "crops" / f"{cid}.png").write_bytes(png)
    v = (
        verdicts.get(cid)
        or f.extra.get("verdict_entry")
        or verdicts.get("defaults_by_defect", {}).get(f.kind)
        or {}
    )
    if "proposed" in v:
        f.found, f.proposed = v.get("found", f.found), v["proposed"]
    if f.kind == "caption_continuation" and v.get("verdict") == "confirmed":
        f.extra = {**f.extra, "proposal": "reclassify as caption_continuation; text unchanged"}
    return {
        "candidate_id": cid,
        "block_id": b["block_id"],
        "pdf_page": b["pdf_page"],
        "printed_page": b["printed_page"],
        "block_kind": b["kind"],
        "section_path": " › ".join(
            f"{s['number']} {s['title']}" for s in b.get("section_path") or []
        ),
        "defect": f.kind,
        "found": f.found,
        "proposed": f.proposed,
        "rule": f.rule,
        "fix_type": "deterministic" if f.rule in DETERMINISTIC else "per_block",
        "evidence": f.evidence,
        "extra": f.extra,
        "verdict": v.get("verdict", "unreviewed"),
        "verdict_note": v.get("note"),
        "retrieval_eligible": bool(b["retrieval"]["include"]),
        "citation_visible": citation_visible(b, titles),
        "text_raw": b.get("text_raw"),
        "text": b["text"],
        "crop": f"crops/{cid}.png",
        "crop_png": png,
        "_finding": f,
    }


def summarize(
    candidates: list[dict[str, Any]], compounds: int, checks: list[dict[str, Any]]
) -> dict[str, Any]:
    confirmed = [c for c in candidates if c["verdict"] == "confirmed"]
    return {
        "candidates": len(candidates),
        "by_defect": dict(Counter(c["defect"] for c in candidates)),
        "by_verdict": dict(Counter(c["verdict"] for c in candidates)),
        "confirmed_by_fix_type": dict(Counter(c["fix_type"] for c in confirmed)),
        "confirmed_by_rule": dict(Counter(c["rule"] or "per_block" for c in confirmed)),
        "confirmed_blocks": len({c["block_id"] for c in confirmed}),
        "confirmed_found_text_missing": [
            c["candidate_id"] for c in confirmed if not c["found_in_text"]
        ],
        "confirmed_pages": sorted({c["pdf_page"] for c in confirmed}),
        "confirmed_retrieval_eligible": sum(c["retrieval_eligible"] for c in confirmed),
        "confirmed_citation_visible": sum(c["citation_visible"] for c in confirmed),
        "space_compounds_kept_as_printed": compounds,
        "caption_checks": {c["table"]: c["same_text"] for c in checks},
        "table_yaml_captions_to_sync_after_errata": [
            c["table"] for c in checks if c["differs_after_errata"]
        ],
        "notes": NOTES,
    }


def render(
    candidates: list[dict[str, Any]], summary: dict[str, Any], checks: list[dict[str, Any]]
) -> str:
    e = html.escape
    rows = []
    for c in candidates:
        img = base64.b64encode(c["crop_png"]).decode()
        rows.append(
            f"<tr class='v-{e(c['verdict'])}'><td>{e(c['candidate_id'])}<br><small>"
            f"{e(c['block_id'])}<br>p{c['pdf_page']} / {e(str(c['printed_page']))}<br>"
            f"{e(c['block_kind'])}</small></td><td>{e(c['defect'])}<br><small>"
            f"{e(c['fix_type'])} {e(c['rule'] or '')}</small></td>"
            f"<td dir='rtl'>{e(c['found'])}</td><td dir='rtl'>{e(c['proposed'] or '—')}</td>"
            f"<td><b>{e(c['verdict'])}</b><br><small>{e(c['verdict_note'] or '')}</small></td>"
            f"<td><img src='data:image/png;base64,{img}'></td>"
            f"<td><small>{e(c['evidence'])}<br>{e(c['section_path'])}<br>retrieval: "
            f"{c['retrieval_eligible']} · citation: {c['citation_visible']}</small>"
            f"<details><summary>texts</summary><b>raw</b><pre dir='rtl'>{e(c['text_raw'] or '')}"
            f"</pre><b>current</b><pre dir='rtl'>{e(c['text'])}</pre><b>proposed block</b>"
            f"<pre dir='rtl'>{e(c['proposed_block_text'] or '—')}</pre></details></td></tr>"
        )
    cap = "".join(
        f"<tr><td>{e(k['table'])}</td><td>{e(', '.join(k['caption_blocks']))}</td>"
        f"<td>{k['same_text']}</td><td>{k['table_number_in_caption']}</td>"
        f"<td dir='rtl'>{e(k['corpus_text'])}</td></tr>"
        for k in checks
    )
    return f"""<!doctype html><html lang="fa"><head><meta charset="utf-8">
<title>Main corpus errata</title><style>
body{{font-family:Tahoma,sans-serif;margin:16px}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #bbb;padding:4px;vertical-align:top;font-size:13px}}
img{{max-width:520px}} .v-confirmed{{background:#eef8ee}}
.v-rejected{{background:#f6f6f6;color:#777}}
.v-kept_as_printed{{background:#fff8e6}} pre{{white-space:pre-wrap}}</style></head><body>
<h1>Main corpus errata audit (candidates; curated/ unchanged)</h1>
<pre>{e(json.dumps(summary, ensure_ascii=False, indent=1))}</pre>
<h2>Table captions vs curated table.yaml</h2><table><tr><th>table</th><th>blocks</th>
<th>same text</th><th>number found</th><th>caption text</th></tr>{cap}</table>
<h2>Notes</h2><ul>{"".join(f"<li>{e(n)}</li>" for n in NOTES)}</ul>
<h2>Candidates</h2><table><tr><th>id / block</th><th>defect</th><th>found</th><th>proposed</th>
<th>verdict</th><th>source crop</th><th>evidence</th></tr>{"".join(rows)}</table></body></html>"""


def write_decisions(candidates: list[dict[str, Any]], by_id: dict[str, dict[str, Any]],
                    summary: dict[str, Any]) -> None:  # fmt: skip
    decisions = errata_decisions(candidates, by_id)
    doc = {
        "schema_version": 1,
        "scope": "curated main corpus errata (approved review package, 2026-10-17)",
        "basis_default": "assistant_reviewed",
        "approved_on": "2026-10-17",
        "review_package": "data/processed/main_errata/ (candidates.jsonl, review_main_errata.html)",
        "rules": {"B2": "mirrored «)…(» pair -> «(…)», reviewed cases only",
                  "L1": "two non-word fragments joined, reviewed cases only",
                  "T1": "tanwin re-attached to its letter, reviewed cases only"},
        "no_generic_rule": "no unrestricted kashida-gap rule: only the listed reviewed cases apply",
        "counts": {"confirmed_candidates": summary["by_verdict"].get("confirmed", 0),
                   "by_rule": summary["confirmed_by_rule"],
                   "kept_as_printed": summary["by_verdict"].get("kept_as_printed", 0),
                   "rejected": summary["by_verdict"].get("rejected", 0), "blocks": len(decisions)},
        "decisions": decisions,
    }  # fmt: skip
    header = "# Approved main-corpus errata (guarded; applied by main_decisions.apply_errata).\n"
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100)
    ERRATA_PATH.write_text(header + body, encoding="utf-8", newline="\n")
    print(f"wrote {ERRATA_PATH} ({len(decisions)} blocks)")


if __name__ == "__main__":
    raise SystemExit(main())
