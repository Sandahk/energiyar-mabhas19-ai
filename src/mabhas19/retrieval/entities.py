"""Source-derived entity resolution for the structured tables (routing v3).

One resolver for Appendix 7 materials, Appendix 3 occupancies and Appendix 2 / table 19-3-2
places. Resolution is anchored to curated labels: a label (or a label path) is matched *inside*
the query, never the other way round, and the most specific match wins. Words of the query that
match no label are reported as leftovers (notes), not blockers; they block a material only when
they name a conflicting curated material or when nothing more specific than a generic category
was matched next to an unknown word (a likely product or unlisted material).

Entity normalization (this module only; BM25 keeps its own frozen normalization):

* search normalization (Arabic/Persian letters, ASCII digits, ZWNJ as a break, no kashida);
* punctuation is never part of a token («لار،» -> «لار», «چقدره؟» -> «چقدره»);
* spacing / ZWNJ / joined spellings match: label and query spans of up to three tokens are also
  compared with their spaces removed («هایپر مارکت» = «هایپرمارکت», «قائمشهر» = «قائم شهر»);
* reversible morphology: plural «ها/های», Arabic plural «ات» (also «ات» -> «ه»), adjectival or
  possessive «ی/ای/ش/اش»;
* number words and ordinals map to digits; a leading number word joined to a label word is split
  («چهارلوله» -> «4 لوله») only when the rest is a label word.

No alias is taken from evaluation wording. Alternative forms must be printed in the curated
sources or listed in a reviewed registry (none yet).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mabhas19.text.normalize import normalize_for_search

# letters of the Arabic block without its punctuation (، ؛ ؟ ٪ ٫ ٬ ۔), digits, Latin, operators
_ETOKEN = re.compile(r"[a-z]*\d+(?:\.\d+)?[a-z]*|[a-z]+|[ؠ-يٮ-ۓە-ۿ]+|[≥≤<>]")
NUMBER_WORDS = {
    "یک": "1",
    "دو": "2",
    "سه": "3",
    "چهار": "4",
    "پنج": "5",
    "شش": "6",
    "هفت": "7",
    "هشت": "8",
    "ده": "10",
}
ORDINALS = {
    "اول": "1",
    "یکم": "1",
    "دوم": "2",
    "سوم": "3",
    "چهارم": "4",
    "پنجم": "5",
    "ششم": "6",
}
_SUFFIXES = ("های", "ها", "ات", "ای", "اش", "ی", "ش")
_DISTINCTIVE = math.log(1 + 20)  # IDF of a word found in at most ~5% of labels
# function words never used as identity anchors
FUNCTION_WORDS = frozenset(
    {
        "و",
        "در",
        "به",
        "از",
        "که",
        "این",
        "را",
        "رو",
        "با",
        "است",
        "هست",
        "برای",
        "یا",
        "آن",
        "ان",
        "تا",
        "بر",
        "هر",
        "می",
        "شود",
        "شده",
        "باشد",
        "نیز",
        "های",
        "ها",
        "ای",
        "هم",
        "اگر",
        "باید",
        "چه",
        "چی",
        "چیه",
        "چیست",
        "کدام",
        "کدوم",
        "چقدر",
        "چقدره",
        "چند",
        "چگونه",
        "چطور",
        "ی",
        "دارد",
        "داره",
        "دارن",
        "دارند",
        "کند",
        "کنه",
        "طبق",
        "مطابق",
        "حدود",
        "حدودا",
        "تقریبا",
        "مثل",
        "مانند",
        "یعنی",
        "the",
        "of",
        "and",
    }
)


def etokens(text: str) -> list[str]:
    """Entity tokens of raw or normalized text (punctuation never attached)."""
    return _ETOKEN.findall(normalize_for_search(text))


def keys(token: str) -> frozenset[str]:
    """Matching keys of one token: itself, digit form of number words/ordinals, and stems with a
    reversible suffix removed (stem length >= 3)."""
    out = {token}
    if token in NUMBER_WORDS:
        out.add(NUMBER_WORDS[token])
    if token in ORDINALS:
        out.add(ORDINALS[token])
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            out.add(stem)
            if suffix == "ات":
                out.add(stem + "ه")
    return frozenset(out)


def split_number_prefix(token: str, vocabulary: set[str]) -> list[str]:
    """«چهارلوله» -> ["چهار", "لوله"] when «لوله» is a label word; otherwise [token]."""
    for word in NUMBER_WORDS:
        rest = token[len(word) :]
        if token.startswith(word) and len(rest) >= 3 and rest in vocabulary:
            return [word, rest]
    return [token]


@dataclass(frozen=True)
class _Span:
    start: int
    end: int  # exclusive
    keys: frozenset[str]


def _spans(toks: Sequence[str], max_len: int = 3) -> list[_Span]:
    out = []
    for i in range(len(toks)):
        for n in range(1, max_len + 1):
            if i + n > len(toks):
                break
            part = toks[i : i + n]
            joined = "".join(part)
            k = keys(joined) if n == 1 else frozenset({joined}) | keys(joined)
            out.append(_Span(i, i + n, k))
    return out


@dataclass
class Label:
    """One matchable label: tokens with a level per token (0 = most general)."""

    id: str
    tokens: list[str]
    levels: list[int]
    spans: list[_Span] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.spans = _spans(self.tokens)


@dataclass
class Match:
    label: Label
    label_idx: set[int]
    query_idx: set[int]
    score: float
    coverage: float = 1.0  # share of the deepest printed level's words matched
    specificity: tuple[float, ...] = ()  # coverage per level, deepest first (tie-break)

    @property
    def max_level(self) -> int:
        return max((self.label.levels[i] for i in self.label_idx), default=-1)


class LabelIndex:
    """Labels matched inside a query by token/joined-span keys, scored by IDF of matched label
    tokens (rarer label words are more specific)."""

    def __init__(self, labels: list[Label], ignore: Iterable[str] = ()) -> None:
        self.labels = labels
        self.ignore = frozenset(ignore) | FUNCTION_WORDS
        df: Counter[str] = Counter()
        for lab in labels:
            df.update({t for t in lab.tokens})
        n = max(1, len(labels))
        self.idf = {t: math.log(1 + n / f) for t, f in df.items()}
        self.vocabulary = set(df)
        self.key_vocabulary = {k for lab in labels for t in lab.tokens for k in keys(t)}

    def query_tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for t in etokens(text):
            out += split_number_prefix(t, self.vocabulary)
        return out

    def match(self, q: Sequence[str]) -> list[Match]:  # noqa: PLR0912
        """Labels with at least one non-ignored word found in the query. Score: sum over the
        distinct matched label words of IDF, x0.85 when only a stem/variant matched."""
        qspans = _spans(q)
        by_key: dict[str, list[_Span]] = {}
        for s in qspans:
            if all(t in self.ignore for t in q[s.start : s.end]):
                continue
            for k in s.keys:
                by_key.setdefault(k, []).append(s)
        out = []
        for lab in self.labels:
            pairs = []
            for ls in lab.spans:
                if all(t in self.ignore for t in lab.tokens[ls.start : ls.end]):
                    continue
                joined = "".join(lab.tokens[ls.start : ls.end])
                for s in {id(s): s for k in ls.keys for s in by_key.get(k, [])}.values():
                    exact = "".join(q[s.start : s.end]) == joined
                    pairs.append((exact, ls.end - ls.start + s.end - s.start, ls, s))
            # each query word supports at most one label word: exact and longer spans first
            pairs.sort(key=lambda p: (not p[0], -p[1]))
            used_q: set[int] = set()
            weight: dict[int, float] = {}
            for exact, _, ls, s in pairs:
                qs = set(range(s.start, s.end))
                if qs & used_q:
                    continue
                used_q |= qs
                for i in range(ls.start, ls.end):
                    weight[i] = max(weight.get(i, 0.0), 1.0 if exact else 0.85)
            per_word: dict[str, float] = {}
            for i, w in weight.items():
                t = lab.tokens[i]
                if t not in self.ignore:
                    per_word[t] = max(per_word.get(t, 0.0), w)
            if per_word:
                score = sum(self.idf[t] * w for t, w in per_word.items())
                anchors = {i for i in weight if lab.tokens[i] not in self.ignore}
                # specificity: share of each printed level's words that matched, deepest
                # level first (a label with fewer unmatched words is more specific)
                per_level = []
                for lv in sorted(set(lab.levels), reverse=True):
                    words = {
                        t
                        for t, x in zip(lab.tokens, lab.levels, strict=True)
                        if x == lv and t not in self.ignore
                    }
                    if words:
                        per_level.append(len(words & set(per_word)) / len(words))
                spec = tuple(per_level)
                out.append(Match(lab, anchors, used_q, score, spec[0] if spec else 0.0, spec))
        return out


# ------------------------------------------------------------------ resolutions


@dataclass
class Resolution:
    """Outcome of resolving one entity type in a query.

    status: ``resolved`` (ids), ``none`` (no specific entity named), ``unknown`` (a specific
    entity is named but no curated label matches), ``ambiguous`` (conflicting curated
    identities named).
    """

    status: str
    ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    leftover: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    query_tokens: list[str] = field(default_factory=list)


def best_matches(matches: list[Match]) -> list[Match]:
    if not matches:
        return []
    top = max(m.score for m in matches)
    tied = [m for m in matches if m.score >= top - 1e-9]
    best_spec = max(m.specificity for m in tied)  # most specific label: fewest unmatched words
    return [m for m in tied if m.specificity == best_spec]


def _material_labels(rows: list[dict[str, Any]]) -> list[Label]:
    labels = []
    for r in rows:
        toks: list[str] = []
        levels: list[int] = []
        for level, key in enumerate(("category", "subgroup", "material", "condition")):
            v = r.get(key)
            text = v.get("text") if isinstance(v, dict) else None
            if not text:
                continue
            words = [t for t in etokens(text) if not t.replace(".", "").isdigit()]
            toks += words
            levels += [level] * len(words)
        labels.append(Label(r["id"], toks, levels))
    return labels


class EntityResolver:
    """Materials (Appendix 7), occupancies (Appendix 3) and places (Appendix 2, table 19-3-2)."""

    def __init__(  # noqa: PLR0913
        self,
        materials: list[dict[str, Any]],
        occupancies: list[dict[str, Any]],
        places: dict[str, list[tuple[str, dict[str, Any]]]],
        *,
        material_ignore: Iterable[str] = (),
        generic_material_words: Iterable[str] = (),
        occupancy_ignore: Iterable[str] = (),
    ) -> None:
        self.material_rows = materials
        self.generic = frozenset(normalize_for_search(w) for w in generic_material_words)
        self.generic_keys = frozenset(k for w in self.generic for k in keys(w))
        self.material_index = LabelIndex(
            _material_labels(materials), ignore=set(material_ignore) | self.generic
        )
        self.material_category = {r["id"]: (r.get("category") or {}).get("text") for r in materials}
        self.occupancy_index = LabelIndex(
            [
                Label(o["id"], toks, [0] * len(toks))
                for o in occupancies
                for toks in [
                    [t for t in etokens(o["label"]) if t not in "≥≤<>" and not t.isdigit()]
                ]
            ],
            ignore=occupancy_ignore,
        )
        self.place_keys = {k.replace(" ", ""): k for k in places}
        self.vocabulary: set[str] = set()  # corpus vocabulary (optional; set by the router)

    # ------------------------------------------------------------------ places
    def places(self, text: str) -> list[str]:
        """Curated place aliases named in the text (longest names; joined/spaced forms match)."""
        toks = etokens(text)
        found: list[tuple[int, int, str]] = []
        for i in range(len(toks)):
            for n in (3, 2, 1):
                if i + n > len(toks):
                    continue
                alias = self.place_keys.get("".join(toks[i : i + n]))
                if alias:
                    found.append((i, i + n, alias))
                    break
        # drop names covered by a longer one
        keep = [
            f for f in found if not any(g != f and g[0] <= f[0] and f[1] <= g[1] for g in found)
        ]
        return list(dict.fromkeys(a for _, _, a in keep))

    # ------------------------------------------------------------------ occupancies
    def occupancies(self, text: str) -> Resolution:
        q = self.occupancy_index.query_tokens(text)
        best = best_matches(self.occupancy_index.match(q))
        if not best:
            return Resolution("none", query_tokens=q)
        return Resolution(
            "resolved",
            [m.label.id for m in best],
            [" ".join(m.label.tokens) for m in best],
            query_tokens=q,
        )

    # ------------------------------------------------------------------ materials
    def material_anchor_words(self, text: str) -> list[str]:
        """Query words that could name a material (not property, generic or function words)."""
        idx = self.material_index
        return [t for t in idx.query_tokens(text) if t not in idx.ignore and not _is_number(t)]

    def materials(self, text: str) -> Resolution:
        idx = self.material_index
        q = idx.query_tokens(text)
        named = [t for t in q if t not in idx.ignore and not _is_number(t)]
        if not named:
            return Resolution("none", query_tokens=q)
        best = best_matches(idx.match(q))
        if not best:
            return Resolution("unknown", leftover=named, query_tokens=q)
        used = set().union(*(m.query_idx for m in best))
        leftover = [
            t
            for i, t in enumerate(q)
            if i not in used and t not in idx.ignore and not _is_number(t)
        ]
        notes = []
        # a leftover that is itself a curated material word of another category names a
        # conflicting identity
        cats = {self.material_category[m.label.id] for m in best}
        distinctive = {
            w
            for m in idx.labels
            if self.material_category[m.id] not in cats
            for w in m.tokens
            if w not in idx.ignore and w not in NUMBER_WORDS and idx.idf.get(w, 0) > _DISTINCTIVE
        }
        conflicting = [
            t
            for t in leftover
            if t not in NUMBER_WORDS
            and not keys(t) & self.generic_keys
            and any(keys(t) & keys(w) for w in distinctive)
        ]
        if conflicting:
            return Resolution(
                "ambiguous",
                leftover=leftover,
                notes=[f"words {conflicting} name another curated material"],
                query_tokens=q,
            )
        unknown_words = [t for t in leftover if self.vocabulary and t not in self.vocabulary]
        if max(m.max_level for m in best) == 0 and unknown_words:
            # only a category matched, next to a word found nowhere in the sources
            return Resolution("unknown", leftover=leftover, query_tokens=q)
        if leftover:
            notes.append(f"words not printed in the matched labels: {leftover}")
        return Resolution(
            "resolved",
            [m.label.id for m in best],
            [" ".join(m.label.tokens) for m in best],
            leftover,
            notes,
            q,
        )


def _is_number(token: str) -> bool:
    return token.replace(".", "").isdigit()
