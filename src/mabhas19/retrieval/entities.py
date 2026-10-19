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
from functools import cache
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


@cache
def keys(token: str) -> frozenset[str]:
    """Matching keys of one token: itself, the digit form of a cardinal number word, and stems
    with a reversible suffix removed (stem length >= 3). Ordinals («چهارم») are deliberately not
    mapped to digits: they express a year/rank dimension, never a number inside a label."""
    out = {token}
    if token in NUMBER_WORDS:
        out.add(NUMBER_WORDS[token])
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            out.add(stem)
            if suffix == "ات":
                out.add(stem + "ه")
    return frozenset(out)


# meaning-bearing prefixes kept attached to the word they modify («غیر چگالشی», «بدون پوسته»,
# «تک سرعته», «چند سرعته»): polarity and count are part of a label's identity
NEGATORS = ("غیر", "بدون", "فاقد")
COUNT_PREFIXES = ("تک", "چند")


def merge_modifiers(tokens: Sequence[str], allowed: set[str] | None = None) -> list[str]:
    """Join a negator or count prefix with the next word («غیر»+«چگالشی» -> «غیرچگالشی»).
    Negators always join; count prefixes join only when the joined form is in ``allowed``
    (the label vocabulary), so a question word «چند» never swallows the next word."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if i + 1 < len(tokens) and (
            t in NEGATORS
            or (t in COUNT_PREFIXES and (allowed is None or t + tokens[i + 1] in allowed))
        ):
            out.append(t + tokens[i + 1])
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def negated_word(token: str) -> str | None:
    """The word a merged negation token negates («بدونالیاف» -> «الیاف»)."""
    for n in NEGATORS:
        if token.startswith(n) and len(token) > len(n) + 1:
            return token[len(n) :]
    return None


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
        merged: list[str] = []
        levels: list[int] = []
        i = 0
        while i < len(self.tokens):
            t = self.tokens[i]
            if i + 1 < len(self.tokens) and t in (*NEGATORS, *COUNT_PREFIXES):
                merged.append(t + self.tokens[i + 1])
                levels.append(self.levels[i + 1])
                i += 2
                continue
            merged.append(t)
            levels.append(self.levels[i])
            i += 1
        self.tokens, self.levels = merged, levels
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
        self._match_cache: dict[tuple[str, ...], list[Match]] = {}

    def query_tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for t in etokens(text):
            for part in split_number_prefix(t, self.vocabulary):
                out.append(self._label_stem(part))
        return merge_modifiers(out, self.vocabulary)

    def _label_stem(self, token: str) -> str:
        """A query word reduced to a printed label word when only a suffix differs and the stem
        is itself a label word («کفی» -> «کف»); otherwise unchanged."""
        if token in self.vocabulary or token in self.ignore:
            return token
        for suffix in _SUFFIXES:
            stem = token[: -len(suffix)]
            if token.endswith(suffix) and len(stem) >= 2 and stem in self.vocabulary:
                return stem
        return token

    def match(self, q: Sequence[str]) -> list[Match]:
        """Labels with at least one non-ignored word found in the query. Score: sum over the
        distinct matched label words of IDF, x0.85 when only a stem/variant matched. Pure for
        a given index and token sequence, so results are memoised (bounded)."""
        key = tuple(q)
        hit = self._match_cache.get(key)
        if hit is None:
            if len(self._match_cache) > 20000:
                self._match_cache.clear()
            hit = self._match(q)
            self._match_cache[key] = hit
        return list(hit)

    def _match(self, q: Sequence[str]) -> list[Match]:  # noqa: PLR0912
        # «بدون X» in the query excludes labels that print X positively
        negated = {w for t in q if (w := negated_word(t))}
        negated_keys = {k for w in negated for k in keys(w)}
        qspans = _spans(q)
        by_key: dict[str, list[_Span]] = {}
        for s in qspans:
            if all(t in self.ignore for t in q[s.start : s.end]):
                continue
            for k in s.keys:
                by_key.setdefault(k, []).append(s)
        out = []
        for lab in self.labels:
            if negated_keys and any(keys(t) & negated_keys for t in lab.tokens):
                continue
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
            for i, wt in weight.items():
                t = lab.tokens[i]
                if t not in self.ignore:
                    per_word[t] = max(per_word.get(t, 0.0), wt)
            if per_word:
                score = sum(self.idf[t] * wt for t, wt in per_word.items())
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
        # words inside a negated phrase («بدون الیاف شیشه») qualify the material; they never
        # name a second one
        negated_pos = {i + 1 for i, t in enumerate(q) if negated_word(t)}
        leftover = [
            t
            for i, t in enumerate(q)
            if i not in used and i not in negated_pos and t not in idx.ignore and not _is_number(t)
        ]
        notes = []
        # a word found nowhere in the sources right next to a material word that only partly
        # names a printed material is a product or variant the regulation does not print (e.g. a
        # brand): unresolved. A fully named material keeps descriptive or glossing words as notes.
        vocab = self.vocabulary | idx.vocabulary
        partial = max(m.coverage for m in best) < 1.0
        unverified = [
            q[i]
            for i in range(len(q))
            if partial
            and i not in used
            and self.vocabulary
            and q[i] not in idx.ignore
            and not _is_number(q[i])
            and not keys(q[i]) & vocab
            and ({i - 1, i + 1} & used)
        ]
        if unverified:
            return Resolution(
                "unknown",
                leftover=leftover,
                notes=[f"unverified name {unverified} next to the matched material word"],
                query_tokens=q,
            )
        # a leftover that is itself a curated material word of another category names a
        # conflicting identity
        cats = {self.material_category[m.label.id] for m in best}
        # leftover words conflict only when together they fully name a printed category, subgroup
        # or material of another category; a single shared descriptor («خیلی», «ساخته») or a
        # condition qualifier («اسمی») is not a second identity
        left_keys = {
            k
            for t in leftover
            if t not in NUMBER_WORDS and not keys(t) & self.generic_keys
            for k in keys(t)
        }
        conflicting: list[str] = []
        for m in idx.labels:
            if self.material_category[m.id] in cats:
                continue
            for lv in set(m.levels) - {3}:  # condition level: qualifiers, not identities
                words = {
                    w
                    for w, x in zip(m.tokens, m.levels, strict=True)
                    if x == lv and w not in idx.ignore and w not in NUMBER_WORDS
                }
                if words and all(keys(w) & left_keys for w in words):
                    conflicting = sorted(words)
                    break
            if conflicting:
                break
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
