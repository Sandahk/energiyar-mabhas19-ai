"""Mabhas 19 scope map and scope classification (answerability v3.1).

The scope map is derived from curated/ (chapter/section hierarchy, numbered clauses, defined
terms of 19-1, table captions and structured table domains, the appendix list, explicit
hand-offs to other Mabhas and standards) plus one reviewed taxonomy of neighbouring domains
that Mabhas 19 does not cover (``eval/retrieval/external_domains.yaml``). It is written to
``eval/retrieval/scope_map.yaml`` (versioned, with the SHA-256 of every input) by
``python -m mabhas19.cli.build_scope_map``; a test keeps the file equal to a fresh build.

Scope classes (kept distinct internally; the router may still expose ``unsupported`` with the
class as its reason):

* ``supported`` / ``supported_structured`` / ``supported_mixed`` - answered from Mabhas 19;
* ``external_domain`` - the question's focus is a neighbouring domain Mabhas 19 does not cover;
* ``invalid_reference`` - an explicit section/clause/table/appendix number that does not exist;
* ``unknown_entity`` - a structured lookup whose entity is not printed in the curated tables;
* ``uncertain`` - no confident decision: routed to prose retrieval (recall first).

External evidence must outweigh the Mabhas 19 concepts found outside it in the same query, or
consist of a word that never occurs in Mabhas 19. Low lexical coverage is never evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from mabhas19.core.paths import CURATED_DIR, EVAL_DIR, REPO_ROOT
from mabhas19.retrieval.entities import FUNCTION_WORDS, NUMBER_WORDS, etokens, keys
from mabhas19.text.normalize import normalize_for_search

DOC_ID = "m19-ed5-1404-main"
SCOPE_MAP_PATH = EVAL_DIR / "retrieval" / "scope_map.yaml"
EXTERNAL_PATH = EVAL_DIR / "retrieval" / "external_domains.yaml"
MABHAS_ORDINALS = {
    "اول": 1,
    "دوم": 2,
    "سوم": 3,
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
    "هفتم": 7,
    "هشتم": 8,
    "نهم": 9,
    "دهم": 10,
    "یازدهم": 11,
    "دوازدهم": 12,
    "سیزدهم": 13,
    "چهاردهم": 14,
    "پانزدهم": 15,
    "شانزدهم": 16,
    "هفدهم": 17,
    "هجدهم": 18,
    "هیجدهم": 18,
    "نوزدهم": 19,
    "بیستم": 20,
    "بیست و یکم": 21,
    "بیست و دوم": 22,
}
_MABHAS_REF = re.compile(
    r"مبحث\s*(" + "|".join(sorted(MABHAS_ORDINALS, key=len, reverse=True)) + r"|\d+)(?![ؠ-ۿ\d])"
)
_STANDARD_REF = re.compile(r"\b(iso|ashrae|iec|astm|din|en)\s*\d+")
# units and measure words never anchor a concept
UNIT_WORDS = frozenset(
    {"متر", "مربع", "مکعب", "وات", "کیلووات", "ساعت", "درجه", "درصد", "کیلوگرم", "بین"}
)
# words that never anchor a Mabhas 19 concept on their own
GENERIC_WORDS = frozenset(
    {
        "ساختمان",
        "ساختمانها",
        "ساختمانهای",
        "ساختمانی",
        "مبحث",
        "مقررات",
        "ملی",
        "کلیات",
        "تعاریف",
        "الزامات",
        "روش",
        "طراحی",
        "اجرا",
        "اجرای",
        "حداقل",
        "حداکثر",
        "میزان",
        "مقدار",
        "تعداد",
        "نوع",
        "انواع",
        "کل",
        "سایر",
        "عمومی",
        "شرایط",
        "مورد",
        "نیاز",
        "لازم",
        "فضای",
        "فضاهای",
        "فضا",
        "واحد",
        "بخش",
        "جدول",
        "پیوست",
    }
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


# ------------------------------------------------------------------ building the map


def build_scope_map(  # noqa: PLR0912, PLR0915
    curated: Path = CURATED_DIR, external: Path = EXTERNAL_PATH
) -> dict[str, Any]:
    """The scope map as plain data (deterministic; written to scope_map.yaml)."""
    sections_path = curated / "structure" / DOC_ID / "sections.jsonl"
    blocks_path = curated / "text" / DOC_ID / "blocks.jsonl"
    sections = _jsonl(sections_path)
    blocks = _jsonl(blocks_path)
    section_numbers = sorted({s["section_number"] for s in sections}, key=_num_key)
    clause_numbers = sorted(
        {b["section_number"] for b in blocks if b["kind"] == "clause" and b["section_number"]}
        - set(section_numbers),
        key=_num_key,
    )
    by_heading = {b["block_id"]: i for i, b in enumerate(blocks)}
    defined = []
    for s in sections:
        if not s["section_number"].startswith("19-1-"):
            continue
        english = None
        i = by_heading.get(s["heading_block"])
        if i is not None:
            for b in blocks[i + 1 : i + 4]:
                if b.get("structural_class") == "definition_body" and b["text"]:
                    first = b["text"].split("\n")[0]
                    english = first if re.fullmatch(r"[A-Za-z0-9 ()/\-,.&]+", first) else None
                    break
        defined.append({"number": s["section_number"], "term": s["title"], "english": english})
    captions = {}
    for b in blocks:
        if b["kind"] == "table_caption" and b["text"]:
            m = re.search(r"جدول\s*(19-\d+-\d+)", normalize_for_search(b["text"]))
            if m:
                captions[m[1]] = b["text"]
    appendices = []
    for b in blocks:
        m = re.match(r"پیوست\s*(\d+)\s*:\s*(.+)", normalize_for_search(b["text"] or ""))
        if b["pdf_page"] == 146 and m:
            appendices.append({"number": int(m[1]), "title": b["text"].split(":", 1)[1].strip()})
    appendices.sort(key=lambda a: a["number"])
    curated_tables = {
        2: "city_climate_classes",
        3: "max_energy_use_intensity",
        7: "material_thermal_properties",
    }
    for a in appendices:
        a["curated_dataset"] = curated_tables.get(a["number"])
    handoffs: dict[str, dict[str, Any]] = {}
    for b in blocks:
        if not b["text"] or not b["retrieval"]["include"]:
            continue
        t = normalize_for_search(b["text"])
        for m in _MABHAS_REF.findall(t):
            num = MABHAS_ORDINALS.get(m) or int(m)
            if num == 19:
                continue
            h = handoffs.setdefault(
                f"mabhas-{num}", {"target": f"مبحث {num}", "mabhas": num, "blocks": []}
            )
            if b["block_id"] not in h["blocks"]:
                h["blocks"].append(b["block_id"])
        for m in _STANDARD_REF.findall(t):
            h = handoffs.setdefault(
                f"standard-{m}", {"target": m.upper(), "mabhas": None, "blocks": []}
            )
            if b["block_id"] not in h["blocks"]:
                h["blocks"].append(b["block_id"])
    entities = _structured_entities(curated)
    ext = yaml.safe_load(external.read_text(encoding="utf-8"))
    for d in ext["domains"]:
        d["handed_off_by"] = sorted(
            {
                bid
                for n in d.get("mabhas", [])
                for bid in handoffs.get(f"mabhas-{n}", {}).get("blocks", [])
            }
        )
    inputs = {
        _rel(p): _sha_file(p)
        for p in (sections_path, blocks_path, curated / "tables", external)
        if p.is_file()
    }
    # structured entity labels come from every curated table file
    tables = sorted(p for p in (curated / "tables").rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for table_file in tables:
        digest.update(_rel(table_file).encode() + b"\0" + _sha_file(table_file).encode() + b"\n")
    inputs[_rel(curated / "tables") + "/**"] = digest.hexdigest()
    return {
        "schema_version": 1,
        "description": "Mabhas 19 scope map (derived from curated/ plus the reviewed "
        "external-domain taxonomy)",
        "inputs_sha256": inputs,
        "identifiers": {
            "chapters": [s["section_number"] for s in sections if s["level"] == 1],
            "sections": section_numbers,
            "clauses": clause_numbers,
            "tables": sorted(captions, key=_num_key),
            "appendices": appendices,
        },
        "supported": {
            "chapters": [
                {"number": s["section_number"], "title": s["title"]}
                for s in sections
                if s["level"] == 1
            ],
            "section_titles": [
                {"number": s["section_number"], "title": s["title"]}
                for s in sections
                if not s["section_number"].startswith("19-1-")
            ],
            "defined_terms": defined,
            "table_captions": [{"table": k, "caption": v} for k, v in sorted(captions.items())],
            "structured_entities": entities,
        },
        "handoffs": sorted(handoffs.values(), key=lambda h: (h["mabhas"] or 99, h["target"])),
        "external_domains": ext["domains"],
        "external_taxonomy": {"version": ext["version"], "reviewed_on": ext["reviewed_on"]},
    }


def _structured_entities(curated: Path) -> dict[str, list[str]]:
    from mabhas19.retrieval.structured import StructuredStore  # noqa: PLC0415

    store = StructuredStore(curated)
    labels: dict[str, list[str]] = {
        "occupancies": [o["label"] for o in store.occupancy_index],
        "places": sorted(store.places),
        "materials": sorted(
            {
                str((r.get(k) or {}).get("text"))
                for r in store.app07
                for k in ("category", "subgroup", "material")
                if (r.get(k) or {}).get("text")
            }
        ),
    }
    for tid, t in store.main.items():
        if t.labels is not None:
            labels[f"rows_{tid.split('.t')[-1]}"] = [
                " ".join(lab.tokens) for lab in t.labels.labels
            ]
    return labels


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _num_key(number: str) -> tuple[int, ...]:
    return tuple(int(x) for x in number.split("-") if x.isdigit())


def write_scope_map(path: Path = SCOPE_MAP_PATH) -> dict[str, Any]:
    data = build_scope_map()
    path.write_text(
        "# Derived file: python -m mabhas19.cli.build_scope_map (do not edit by hand).\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    return data


# ------------------------------------------------------------------ runtime


@dataclass
class ReferenceCheck:
    ref: str
    kind: str  # section | table | appendix | chapter
    status: str  # valid | nonexistent_child | nonexistent | malformed
    detail: str = ""


@dataclass
class ScopeDecision:
    scope: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


_LOOSE_SECTION = re.compile(r"(?<![\d\-/])19(?:-\d*){1,7}(?![\d/])")
_LOOSE_TABLE = re.compile(r"جدول\s*(?:شماره\s*)?(19(?:-\d*)*|پ\s?\d+(?:-\d*)*)")
_APPENDIX = re.compile(r"پیوست\s*(?:شماره\s*)?[«\"(]?\s*(\d+)(?![\d-])")
# «فصل نهم», «فصل ۹»: a chapter named by number or ordinal word («فصل سرد» is not matched)
_CHAPTER_ORDINALS = {
    w: i + 1
    for i, w in enumerate(
        (
            "اول",
            "دوم",
            "سوم",
            "چهارم",
            "پنجم",
            "ششم",
            "هفتم",
            "هشتم",
            "نهم",
            "دهم",
            "یازدهم",
            "دوازدهم",
            "سیزدهم",
            "چهاردهم",
            "پانزدهم",
        )
    )
} | {"یکم": 1}
_CHAPTER = re.compile(r"فصل\s*(?:شماره\s*)?(\d+|[^\s\d]+)(?![\d-])")
# tables whose row labels are regulated concepts (requirements, envelope parts, equipment)
_CONCEPT_ROWS = ("rows_19-2-", "rows_19-5-1", "rows_19-5-2", "rows_19-5-3", "rows_19-5-4")
_STRICT_SECTION = re.compile(r"19(?:-\d+){1,5}")
_STRICT_TABLE = re.compile(r"19-\d+-\d+|پ\d+-\d+")


class ScopeMap:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        ids = data["identifiers"]
        self.sections = set(ids["sections"]) | set(ids["clauses"]) | set(ids["chapters"])
        self.chapters = set(ids["chapters"])
        self.tables = set(ids["tables"])
        self.appendices = {a["number"]: a for a in ids["appendices"]}
        sup = data["supported"]
        texts = [c["title"] for c in sup["chapters"]]
        texts += [s["title"] for s in sup["section_titles"]]
        texts += [d["term"] for d in sup["defined_terms"]]
        texts += [d["english"] for d in sup["defined_terms"] if d["english"]]
        texts += [c["caption"] for c in sup["table_captions"]]
        texts += [a["title"] for a in ids["appendices"]]
        # concept vocabulary only: structured entity labels (occupancies, places, equipment)
        # name *what* a question is about, not whether Mabhas 19 regulates it, so they never
        # outweigh external-domain evidence
        generic = {
            k
            for w in GENERIC_WORDS | FUNCTION_WORDS | set(NUMBER_WORDS) | UNIT_WORDS
            for k in keys(w)
        }
        self.supported_keys = {
            k
            for t in texts
            for w in etokens(t)
            if len(w) > 2 and not w.isdigit()
            for k in keys(w)
            if k not in generic
        }
        # strong in-scope evidence: a Mabhas 19 concept phrase (two consecutive content words of
        # a title, caption or defined term) or a defined term that is a single word; a single
        # shared word («مصرف», «شهر») is only weak overlap
        self.supported_bigrams: set[tuple[str, str]] = set()
        # row labels of the requirement, envelope and equipment tables name regulated concepts
        # («شارژر خودرو برقی», «پمپ دورمتغیر»); climate classes and space types do not
        entities = sup.get("structured_entities", {})
        concept_rows = [
            label
            for key in sorted(entities)
            if key.startswith(_CONCEPT_ROWS)
            for label in entities[key]
        ]
        for t in [*texts, *concept_rows]:
            words = [w for w in etokens(t) if w not in FUNCTION_WORDS and not w.isdigit()]
            for a, b in pairwise(words):
                self.supported_bigrams |= {(x, y) for x in keys(a) for y in keys(b)}
        self.generic_keys = generic
        self.single_terms = {
            k
            for d in sup["defined_terms"]
            if len(etokens(d["term"])) == 1
            for k in keys(etokens(d["term"])[0])
            if k not in generic
        }
        # words printed in curated table rows (occupancies, materials, equipment…): Mabhas 19
        # vocabulary that an everyday-domain term may share («درمان» ~ «درمانی»)
        self.curated_row_keys = {
            k for vals in entities.values() for v in vals for w in etokens(v) for k in keys(w)
        }
        self.external = []
        for d in data["external_domains"]:
            terms = [tuple(etokens(t)) for t in d["terms"]]
            self.external.append((d, [t for t in terms if t]))

    @classmethod
    def load(cls, path: Path = SCOPE_MAP_PATH) -> ScopeMap:
        if path.is_file():
            return cls(yaml.safe_load(path.read_text(encoding="utf-8")))
        return cls(build_scope_map())

    # -------------------------------------------------------------- references
    def check_references(self, norm: str) -> list[ReferenceCheck]:
        out: list[ReferenceCheck] = []
        table_spans = []
        for m in _LOOSE_TABLE.finditer(norm):
            table_spans.append(m.span(1))
            ref = m[1].replace(" ", "")
            out.append(self._check_table(ref))
        for m in _LOOSE_SECTION.finditer(norm):
            if any(a <= m.start() < b for a, b in table_spans):
                continue
            out.append(self._check_section(m[0]))
        for m in _APPENDIX.finditer(norm):
            n = int(m[1])
            status = "valid" if n in self.appendices else "nonexistent"
            detail = (
                f"appendix {n}: {self.appendices[n]['title']}"
                if n in self.appendices
                else f"appendices are numbered {min(self.appendices)}-{max(self.appendices)}"
            )
            out.append(ReferenceCheck(f"پیوست {n}", "appendix", status, detail))
        for m in _CHAPTER.finditer(norm):
            word = m[1]
            chapter = int(word) if word.isdigit() else _CHAPTER_ORDINALS.get(word)
            if chapter is None:
                continue
            ok = f"19-{chapter}" in self.chapters
            detail = f"chapter 19-{chapter}" if ok else f"chapters: {sorted(self.chapters)}"
            status = "valid" if ok else "nonexistent"
            out.append(ReferenceCheck(f"فصل {chapter}", "chapter", status, detail))
        return out

    def _check_table(self, ref: str) -> ReferenceCheck:
        if not _STRICT_TABLE.fullmatch(ref):
            return ReferenceCheck(ref, "table", "malformed", "table numbers are 19-x-y or پN-M")
        if ref.startswith("پ"):
            n = int(ref[1:].split("-", maxsplit=1)[0])
            if n in self.appendices:
                return ReferenceCheck(ref, "table", "valid", f"table of appendix {n}")
            return ReferenceCheck(ref, "table", "nonexistent", f"appendix {n} does not exist")
        if ref in self.tables:
            return ReferenceCheck(ref, "table", "valid")
        return ReferenceCheck(ref, "table", "nonexistent", f"tables: {sorted(self.tables)}")

    def _check_section(self, ref: str) -> ReferenceCheck:
        if not _STRICT_SECTION.fullmatch(ref):
            return ReferenceCheck(ref, "section", "malformed", "section numbers are 19-x[-y…]")
        if ref in self.sections:
            return ReferenceCheck(ref, "section", "valid")
        parts = ref.split("-")
        for i in range(len(parts) - 1, 1, -1):
            parent = "-".join(parts[:i])
            if parent in self.sections:
                return ReferenceCheck(
                    ref, "section", "nonexistent_child", f"{parent} exists but has no {ref}"
                )
        return ReferenceCheck(ref, "section", "nonexistent", "no such chapter or section")

    # -------------------------------------------------------------- classification
    def classify_text(self, norm: str, corpus_vocabulary: set[str]) -> ScopeDecision:
        """supported / external_domain / uncertain for a query without structured lookup."""
        q = etokens(norm)
        qkeys = [keys(t) for t in q]
        found: list[tuple[str, tuple[str, ...], int, int]] = []
        for d, terms in self.external:
            for term in terms:
                found += [(d["id"], term, i, n) for i, n in _find_term(q, term)]

        def _anchors(spans: list[tuple[str, tuple[str, ...], int, int]]) -> list[str]:
            pos = {j for _, _, i, n in spans for j in range(i, i + n)}
            return sorted(
                {
                    q[i]
                    for i, ks in enumerate(qkeys)
                    if i not in pos and ks & self.supported_keys and q[i] not in FUNCTION_WORDS
                }
            )

        # a single-word external term that is printed curated table vocabulary is the technical
        # word, not the everyday domain, when the question has Mabhas 19 concepts of its own
        shared = [x for x in found if len(x[1]) == 1 and keys(x[1][0]) & self.curated_row_keys]
        if shared and _anchors(found):
            found = [x for x in found if x not in shared]
        ext_pos: dict[int, str] = {}
        ext_terms: dict[str, list[str]] = {}
        for domain, term, i, n in found:
            for j in range(i, i + n):
                ext_pos.setdefault(j, domain)
            ext_terms.setdefault(domain, []).append(" ".join(term))
        anchors = _anchors(found)
        content = [i for i in range(len(q)) if q[i] not in FUNCTION_WORDS and i not in ext_pos]
        strong_pos = {i for i in content if qkeys[i] & self.single_terms}
        for a, b in pairwise(content):
            if any((x, y) in self.supported_bigrams for x in qkeys[a] for y in qkeys[b]):
                strong_pos |= {a, b}
        strong = sorted({q[i] for i in strong_pos if not qkeys[i] & self.generic_keys})
        if not ext_pos:
            if anchors:
                return ScopeDecision("supported", "Mabhas 19 concepts", {"anchors": anchors})
            return ScopeDecision("uncertain", "no Mabhas 19 concept and no external domain")
        ext_words = sorted({q[i] for i in ext_pos})
        absent = [w for w in ext_words if len(w) > 2 and not keys(w) & corpus_vocabulary]
        domains = sorted(ext_terms, key=lambda d: -len(ext_terms[d]))
        evidence: dict[str, Any] = {
            "external": {d: sorted(set(t)) for d, t in ext_terms.items()},
            "anchors": anchors,
            "strong_anchors": strong,
            "absent_from_mabhas19": absent,
        }
        top = next(d for d, _ in self.external if d["id"] == domains[0])
        if top.get("handed_off_by"):
            evidence["handed_off_by"] = top["handed_off_by"][:5]
        # out-of-scope evidence wins unless strong in-scope evidence outweighs it
        if len(ext_words) > len(strong):
            why = f"external domain {top['id']} ({top['label']})"
            if top.get("covered_by"):
                why += f"; covered by {', '.join(top['covered_by'])}"
            return ScopeDecision("external_domain", why, evidence)
        return ScopeDecision("uncertain", "external and Mabhas 19 evidence both present", evidence)


def _find_term(q: list[str], term: tuple[str, ...]) -> list[tuple[int, int]]:
    """(start, length) of each occurrence of the term (spacing/joined forms and morphology
    tolerated)."""
    joined = "".join(term)
    target = keys(joined) if len(term) == 1 else frozenset({joined})
    out = []
    for i in range(len(q)):
        for n in (1, 2, 3, 4):
            if i + n > len(q):
                break
            span = "".join(q[i : i + n])
            if (keys(span) if n == 1 else frozenset({span}) | keys(span)) & target:
                out.append((i, n))
                break
    return out
