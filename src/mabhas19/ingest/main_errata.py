"""Errata audit of the curated main corpus (candidates only; nothing is changed).

Detectors (each candidate carries its evidence; the source crop decides):

* ``split_word``: two adjacent Persian tokens whose concatenation is a word attested
  elsewhere in the corpus, where at least one side is a fragment (a single letter or a token
  that is rare/absent on its own). Pairs where both sides are words and only the
  ZWNJ-compound exists («شبیه سازی», «ساختمان ها») are printed spacing, not errata, and are
  counted separately (``space_compound``), never proposed.
* ``isolated_letter``: a single Persian letter token (other than «و») with no attested join.
* ``mirrored_brackets``: Persian text between a closing and an opening bracket «)…(»
  (the B1 situation for Persian content), and unbalanced brackets.
* ``caption_continuation``: a centred line directly below a table/figure caption that the
  corpus classified as a heading or paragraph.
* ``repair_artifact``: double spaces, ZWNJ next to a space, presentation-form letters,
  tatweel, stray combining marks.

Source-faithful printed forms (KEEP_AS_PRINTED) are never proposed. Page furniture, TOC,
formula provenance, OCR text and table cells are out of scope.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

ZWNJ = "‌"
SKIP_KINDS = frozenset(
    {
        "running_header",
        "page_number",
        "toc_entry",
        "formula",
        "formula_fragment",
        "ocr_text",
        "table",
        "figure",
    }
)
KEEP_AS_PRINTED = frozenset({"حد اکثر", "خنک کن", "دودکش دار", "در صد", "غیر چگالشی", "بزرگتر"})
# Persian letters and the marks that belong inside a word (harakat, shadda, superscript alef)
_LETTERS = "ء-غف-يپچژکگیۀآ"
_MARKS = "ً-ْٰ"
TOKEN = re.compile(rf"[{_LETTERS}{_MARKS}{ZWNJ}]+")
WORDS_ALONE = frozenset(
    {
        "و",
        "از",
        "با",
        "در",
        "به",
        "تا",
        "یا",
        "که",
        "را",
        "این",
        "آن",
        "هر",
        "بر",
        "هم",
        "نه",
        "چه",
        "پس",
        "اگر",
        "یک",
        "دو",
        "سه",
        "کف",
        "تن",
        "فن",
        "بی",
        "چند",
        "می",
        "نمی",
        "شده",
        "شود",
        "است",
        "های",
        "ها",
        "ای",
        "تر",
        "ترین",
        "گر",
    }
)
FRAGMENT_MAX_FREQ = 2  # a token seen at most this often on its own is treated as a fragment
# list markers «الف)», «ب)», «۱)» are single closing brackets by design
_MARKER = re.compile(rf"(?:^|(?<=[\s«]))(?:[{_LETTERS}]{{1,3}}|[۰-۹0-9]{{1,2}})\)")
# a letter used as an abbreviation, marker or code: «ص.», «ک-۱۱۴۳», «پ ۱۳-۱», «ب)»
_ABBREV_AFTER = re.compile(r"\s*[.\-:/)،]|\s*[۰-۹0-9]")
CAPTION_KINDS = frozenset({"table_caption", "figure_caption"})
CAPTION_GAP = 20.0  # points between a caption and its continuation line
CAPTION_CENTRE = 40.0  # max horizontal offset of the continuation's centre


def _bare(token: str) -> str:
    return token.strip(ZWNJ)


def lexicon(blocks: list[dict[str, Any]]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for b in blocks:
        if b["kind"] not in SKIP_KINDS and b.get("text"):
            freq.update(_bare(t) for t in TOKEN.findall(b["text"]))
    return freq


@dataclass
class Finding:
    kind: str
    found: str
    proposed: str | None
    evidence: str
    count_in_block: int = 1
    rule: str | None = None  # deterministic rule name when the fix is mechanical
    extra: dict[str, Any] = field(default_factory=dict)


def _is_fragment(token: str, freq: Counter[str]) -> bool:
    t = _bare(token)
    if t in WORDS_ALONE:
        return False
    return len(t) == 1 or freq[t] <= FRAGMENT_MAX_FREQ


def split_words(text: str, freq: Counter[str]) -> tuple[list[Finding], int]:
    """Split-word candidates of one block, plus the number of printed-spacing compounds."""
    out: dict[str, Finding] = {}
    compounds = 0
    for line in text.split("\n"):
        toks = list(TOKEN.finditer(line))
        for a, b in pairwise(toks):
            if line[a.end() : b.start()] != " ":
                continue
            x, y = a.group(), b.group()
            pair = f"{x} {y}"
            if pair in KEEP_AS_PRINTED or _bare(x) == "و" or _bare(y) == "و":
                continue
            prev = out.get(pair)
            if prev:
                prev.count_in_block += 1
                continue
            finding = _split_finding(x, y, freq)
            if finding is None:
                continue
            if finding.kind == "space_compound":
                compounds += 1
            out[pair] = finding
    return [f for f in out.values() if f.kind != "space_compound"], compounds


def _split_finding(x: str, y: str, freq: Counter[str]) -> Finding | None:
    pair = f"{x} {y}"
    if unicodedata.category(_bare(y)[0]) == "Mn":  # «کاملا ً»: a mark cannot stand alone
        return Finding(
            "detached_mark", pair, x + y, "combining mark separated from its letter", rule="T1"
        )
    joined, zjoined = _bare(x) + _bare(y), _bare(x) + ZWNJ + _bare(y)
    fx, fy = _is_fragment(x, freq), _is_fragment(y, freq)
    if not (fx or fy) or freq[joined] + freq[zjoined] == 0:
        return Finding("space_compound", pair, None, "") if freq[zjoined] else None
    if freq[zjoined] > freq[joined]:
        # the attested form is a ZWNJ compound: printed spacing, not proposed
        return Finding("space_compound", pair, None, "")
    lead = x[: len(x) - len(x.lstrip(ZWNJ))]
    return Finding(
        "split_word",
        pair,
        lead + joined,
        f"«{joined}» occurs {freq[joined]}x in the corpus; "
        f"«{_bare(x)}» {freq[_bare(x)]}x, «{_bare(y)}» {freq[_bare(y)]}x on their own",
        rule="L1" if fx and fy and freq[joined] >= 3 else None,
    )


def isolated_letters(text: str, freq: Counter[str], covered: set[str]) -> list[Finding]:
    out = []
    for m in re.finditer(
        rf"(?<![{_LETTERS}{_MARKS}{ZWNJ}])([{_LETTERS}])(?![{_LETTERS}{_MARKS}{ZWNJ}])", text
    ):
        ch = m[1]
        if ch == "و" or any(ch in c.split() for c in covered):
            continue
        if _ABBREV_AFTER.match(text, m.end()):
            continue
        ctx = text[max(0, m.start() - 12) : m.end() + 12].replace("\n", " ")
        out.append(Finding("isolated_letter", ctx, None, f"single letter «{ch}» on its own"))
    return out


def mirrored_spans(text: str) -> tuple[list[tuple[int, int]], int]:
    """Spans «)…(» where a closing bracket opens (no bracket open at that point) and the next
    opening bracket closes it; list markers are masked first. Returns the spans and the
    number of brackets left unmatched."""
    masked = _MARKER.sub(lambda m: "_" * len(m[0]), text)
    spans, depth, start, unmatched = [], 0, None, 0
    for i, ch in enumerate(masked):
        if start is not None:
            if ch == "(":
                spans.append((start, i + 1))
                start = None
            elif ch == ")":
                unmatched += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth:
                depth -= 1
            else:
                start = i
    return spans, unmatched + depth + (start is not None)


def bracket_findings(text: str) -> list[Finding]:
    out = []
    spans, unmatched = mirrored_spans(text)
    for a, b in spans:
        inner = text[a + 1 : b - 1]
        latin = bool(re.match(r"[A-Za-z0-9]", inner)) and not re.search(f"[{_LETTERS}]", inner)
        out.append(
            Finding(
                "mirrored_brackets",
                text[a:b],
                f"({inner})",
                "Latin text between «)» and «(» (B1 not applied)"
                if latin
                else "text between «)» and «(»: a mirrored bracket pair",
                rule="B1" if latin else "B2",
            )
        )
    if unmatched:
        out.append(
            Finding(
                "malformed_brackets",
                text[:80],
                None,
                f"{unmatched} unmatched bracket(s) after list markers are masked",
            )
        )
    return out


def artifacts(text: str) -> list[Finding]:
    out = []
    checks = [
        ("double_space", re.compile(r"[^\s] {2,}[^\s]"), "two or more spaces inside a line"),
        ("zwnj_next_to_space", re.compile(rf" {ZWNJ}|{ZWNJ} "), "ZWNJ adjacent to a space"),
        ("tatweel", re.compile("ـ"), "tatweel (kashida) left in the text"),
    ]
    for name, rx, why in checks:
        hits = rx.findall(text)
        if hits:
            out.append(
                Finding(
                    name, repr(hits[0]), None, f"{why} ({len(hits)}x)", count_in_block=len(hits)
                )
            )
    forms = [c for c in text if "PRESENTATION FORM" in unicodedata.name(c, "")]
    if forms:
        out.append(
            Finding(
                "presentation_forms",
                "".join(forms[:5]),
                None,
                f"{len(forms)} Arabic presentation-form characters",
            )
        )
    return out


def caption_continuations(blocks: list[dict[str, Any]]) -> dict[str, Finding]:
    """Lines directly below a caption, centred on it, that the corpus calls something else."""
    out: dict[str, Finding] = {}
    by_page: dict[int, list[dict[str, Any]]] = {}
    for b in blocks:
        by_page.setdefault(b["pdf_page"], []).append(b)
    for b in blocks:
        if b["kind"] not in CAPTION_KINDS:
            continue
        x0, _, x1, y1 = b["bbox"]
        centre = (x0 + x1) / 2
        for o in by_page[b["pdf_page"]]:
            if o is b or o["kind"] in SKIP_KINDS or o["kind"] in CAPTION_KINDS or not o.get("text"):
                continue
            ox0, oy0, ox1, _ = o["bbox"]
            short = "\n" not in o["text"].strip() and len(o["text"]) <= 80
            near = 0 <= oy0 - y1 <= CAPTION_GAP
            if short and near and abs((ox0 + ox1) / 2 - centre) <= CAPTION_CENTRE:
                out[o["block_id"]] = Finding(
                    "caption_continuation",
                    o["text"],
                    None,
                    f"{o['kind']} {oy0 - y1:.1f} pt below caption {b['block_id']}, centred on it",
                    rule=None,
                    extra={
                        "caption_block": b["block_id"],
                        "current_kind": o["kind"],
                        "proposed_kind": "caption_continuation",
                    },
                )
    return out


def audit_block(b: dict[str, Any], freq: Counter[str]) -> tuple[list[Finding], int]:
    text = b["text"]
    splits, compounds = split_words(text, freq)
    covered = {f.found for f in splits}
    found = (
        splits + isolated_letters(text, freq, covered) + bracket_findings(text) + artifacts(text)
    )
    return found, compounds


BRACKET_KINDS = frozenset({"mirrored_brackets", "malformed_brackets"})


def ordered(findings: list[Finding]) -> list[Finding]:
    """Bracket fixes first: their spans were found on the uncorrected text and may contain
    split words that are corrected afterwards."""
    return sorted(findings, key=lambda f: f.kind not in BRACKET_KINDS)


def replacements(text: str, findings: list[Finding]) -> tuple[str, list[dict[str, Any]]]:
    """Apply the proposals in order; returns the text and each replacement with its exact
    occurrence count (identical proposals are applied once)."""
    out, reps, seen = text, [], set()
    for f in ordered(findings):
        if f.proposed is None or (f.found, f.proposed) in seen:
            continue
        seen.add((f.found, f.proposed))
        n = out.count(f.found)
        if n == 0:
            raise ValueError(f"«{f.found}» not found when applying its correction")
        reps.append({"from": f.found, "to": f.proposed, "count": n, "rule": f.rule})
        out = out.replace(f.found, f.proposed)
    return out, reps


def apply_proposals(text: str, findings: list[Finding]) -> str:
    return replacements(text, findings)[0]


def errata_decisions(
    candidates: list[dict[str, Any]], blocks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Guarded per-block errata decisions from the confirmed review candidates.

    Each block entry pins the sha256 of the text it expects and the exact count of every
    replacement; deterministic rule cases keep their rule name, per-block cases have none.
    """
    by_block: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        if c["verdict"] == "confirmed":
            by_block.setdefault(c["block_id"], []).append(c)
    out = []
    for bid in (b for b in blocks if b in by_block):
        cs = by_block[bid]
        text = blocks[bid]["text"]
        findings = [Finding(c["defect"], c["found"], c["proposed"], "", rule=c["rule"]) for c in cs]
        final, reps = replacements(text, findings)
        for r in reps:
            r["candidates"] = [c["candidate_id"] for c in cs
                               if (c["found"], c["proposed"]) == (r["from"], r["to"])]  # fmt: skip
            r["basis"] = "assistant_reviewed_rule" if r["rule"] else "assistant_reviewed"
        entry: dict[str, Any] = {
            "id": f"ER-{bid.removeprefix('main-')}",
            "block": bid,
            "expect_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "result_text_sha256": hashlib.sha256(final.encode("utf-8")).hexdigest(),
            "replacements": reps,
        }
        caption = [c for c in cs if c["defect"] == "caption_continuation"]
        if caption:
            entry["reclassify"] = {"kind": "caption_continuation",
                                   "caption_block": caption[0]["extra"]["caption_block"],
                                   "candidates": [c["candidate_id"] for c in caption]}  # fmt: skip
        out.append(entry)
    return out


def space_compounds(text: str, freq: Counter[str]) -> Counter[str]:
    """Printed space-separated compounds of a text (kept as printed, never proposed)."""
    out: Counter[str] = Counter()
    for line in text.split("\n"):
        toks = list(TOKEN.finditer(line))
        for a, b in pairwise(toks):
            if line[a.end() : b.start()] != " ":
                continue
            f = _split_finding(a.group(), b.group(), freq)
            if f is not None and f.kind == "space_compound":
                out[f.found] += 1
    return out
