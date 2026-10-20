"""Source-exact entity identity: which printed rows a query really names.

Only normalization (letters, digits, spacing/ZWNJ), reversible morphology (``entities.keys``,
meaning-bearing prefixes kept on their word, joined/spaced forms) and reviewed aliases from the
explicit registry (``ALIASES``) may establish identity. Fuzzy similarity, the nearest or best
scoring label, a shared single word, substring resemblance and a shared broad category never do.

A row (or one «/»-separated alternative of an Appendix 3 occupancy label) is **named** by a query
when the query names

1. a *head* of the row: the first identity word of one of its printed levels (equipment,
   element, occupancy, material, category, requirement…), and
2. enough of the row to tell it from every other printed label: no other label prints every
   word the query names (so «کف مجاور هوای آزاد» needs «آزاد» beside «کف», «بویلر غیرچگالشی»
   needs «غیرچگالشی», «اداری عمومی» needs «عمومی», «مال» is not named by «مالی» of «موسسه
   مالی»); when the head word is printed in this label only, the whole level it heads must be
   named («پمپ‌های گریز از مرکز», not «پمپ» alone).

Identity words exclude function and request words, numbers, comparison operators, units, the
table's own output words and words every row of the table prints (e.g. «مجاور» in 19-5-1). A row
whose identity words are all numeric conditions (PF or power bands) is identified by its
numbers, not by words. Appendix 7 condition texts are qualifiers, never identities.

``status`` per row: ``exact`` (named), ``head_only`` (a head is named but a distinguishing word
is missing - several sibling rows fit, never a choice) or ``none``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval.entities import (
    COUNT_PREFIXES,
    DEGREE_PREFIXES,
    FUNCTION_WORDS,
    NEGATORS,
    NUMBER_WORDS,
    etokens,
    keys,
)
from mabhas19.retrieval.numeric import UNITS
from mabhas19.retrieval.schema import APP03, APP07, MAIN
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.polarity import Polarity
    from mabhas19.retrieval.structured import StructuredStore

# reviewed aliases (alias -> printed form), applied to the query before matching. Empty until an
# alias is reviewed: proposals are listed in the routing reports, never added automatically.
ALIASES: dict[str, str] = {}

_OPERATORS = frozenset(
    {
        "کمتر",
        "بیشتر",
        "بیش",
        "کم",
        "از",
        "بین",
        "تا",
        "کوچک",
        "کوچکتر",
        "بزرگ",
        "بزرگتر",
        "تر",
        "حداکثر",
        "حداقل",
        "حد",
        "اکثر",
        "≥",
        "≤",
        "<",
        ">",
        "مساوی",
        "یا",
        # a printed exception of a range («به غیر از 75 تا 200 کیلووات») is a condition
        "غیراز",
        "بغیراز",
        "بجز",
        "جز",
        "منهای",
    }
)
_UNIT_WORDS = frozenset(w for forms in UNITS.values() for f in forms for w in etokens(f))
_MODIFIERS = (*NEGATORS, *COUNT_PREFIXES, *DEGREE_PREFIXES)


@dataclass
class Alternative:
    row: str
    levels: list[list[str]]  # identity words per printed level (merged modifiers)
    words: list[str] = field(default_factory=list)  # all identity words, in order
    all_words: list[str] = field(default_factory=list)  # every printed word (for exclusions)

    @property
    def heads(self) -> list[str]:
        return [lv[0] for lv in self.levels if lv]


@dataclass
class Named:
    status: str  # exact | head_only | none
    words: list[str] = field(default_factory=list)  # label words the query names
    missing: list[str] = field(default_factory=list)  # distinguishing words not named
    head: str | None = None


def _merge(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if i + 1 < len(tokens) and t in _MODIFIERS:
            out.append(t + tokens[i + 1])
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def _same(a: str, b: str) -> bool:
    return bool(keys(a) & keys(b))


def _in(w: str, words: Sequence[str]) -> bool:
    return any(_same(w, x) for x in words)


def _numeral(w: str) -> bool:
    return w.replace(".", "").replace("/", "").isdigit()


def _counts(lv: Sequence[str], i: int) -> bool:
    """A printed numeral counts the label word after it («4 لوله», «3 فاز»): identity. One
    followed by a unit, an operator or nothing is a numeric condition, never identity."""
    nxt = lv[i + 1] if i + 1 < len(lv) else ""
    return bool(nxt) and not _numeral(nxt) and nxt not in _UNIT_WORDS and nxt not in _OPERATORS


def _same_name(scope: Sequence[str], levels: Sequence[Sequence[str]]) -> bool:
    other = [w for lv in levels for w in lv]
    return bool(other) and all(_in(w, other) for w in scope) and all(_in(w, scope) for w in other)


class IdentityIndex:
    """Identity words of every printed label of one dataset (built once)."""

    def __init__(
        self,
        dataset: str,
        rows: list[tuple[str, list[list[str]], list[str]]],
        ignore: Iterable[str],
        *,
        coordinated_alternatives: bool = False,
        conditions: dict[str, list[str]] | None = None,
    ) -> None:
        self.dataset = dataset
        ignore_keys = {k for w in ignore for k in keys(w)}

        def clean(lv: list[str]) -> list[str]:
            return [
                w
                for i, w in enumerate(lv)
                if w not in FUNCTION_WORDS
                and w not in _OPERATORS
                and w not in _UNIT_WORDS
                and not keys(w) & ignore_keys
                and (not _numeral(w) or _counts(lv, i))
            ]

        by_label: dict[tuple[tuple[str, ...], ...], Alternative] = {}
        self.by_row: dict[str, list[Alternative]] = {}
        # every printed word of each row (rows printing one name share an alternative, never
        # their conditions)
        self.printed: dict[str, list[str]] = {}
        for rid, levels, raw in rows:
            self.printed.setdefault(rid, [])
            self.printed[rid] += [w for w in raw if w not in self.printed[rid]]
            heads = [levels[0]] if levels else [[]]
            if coordinated_alternatives and levels:
                # «دوش و وان», «بانک و موسسه مالی»: each coordinated name identifies the row
                parts: list[list[str]] = [[]]
                for w in levels[0]:
                    if w == "و":
                        parts.append([])
                    else:
                        parts[-1].append(w)
                heads = [p for p in parts if p] or [levels[0]]
            for head_level in heads:
                lv = [clean(head_level), *(clean(x) for x in levels[1:])]
                key = tuple(tuple(x) for x in lv)
                alt = by_label.get(key)
                if alt is None:
                    alt = by_label[key] = Alternative(rid, lv, [w for x in lv for w in x], raw)
                self.by_row.setdefault(rid, []).append(alt)
        alts = list(by_label.values())
        # words every printed label carries have no identity («مجاور» of every 19-5-1 element)
        if len(alts) > 1:
            common = [
                w for w in alts[0].words if all(any(_same(w, x) for x in a.words) for a in alts)
            ]
            for a in alts:
                a.levels = [
                    [w for w in x if not any(_same(w, c) for c in common)] for x in a.levels
                ]
                a.words = [w for x in a.levels for w in x]
        self.alternatives = alts
        # a printed condition without a number is a required qualifier of the row's name
        self.conditions = {r: clean(ws) for r, ws in (conditions or {}).items()}

    def group(self, head: str) -> list[Alternative]:
        """Every printed label containing the head word (the rows it could name)."""
        return [a for a in self.alternatives if any(_same(head, w) for w in a.words)]

    @cached_property
    def vocabulary(self) -> set[str]:
        return {w for a in self.alternatives for w in a.words}

    def named(self, row: str, query: Sequence[str]) -> Named:
        """How the query (positive tokens, merged modifiers) names one row."""
        forms = query_forms(query)
        best = Named("none")
        for a in self.by_row.get(row, []):
            if not a.words:
                return Named("exact")  # identified by its printed numeric condition only
            hit = [w for w in a.words if _named(w, a.words, forms)]
            for li, level in enumerate(a.levels):
                if not level or level[0] not in hit:
                    continue
                # the head's level and the more specific levels identify the row; a more general
                # level (an Appendix 7 category) is never required
                scope = [w for lv in a.levels[li:] for w in lv]
                others = [g for g in self.group(level[0]) if g is not a and g.row != a.row]
                if not others:
                    # the only label printing this head: its whole name must be named
                    missing = [w for w in level if w not in hit]
                else:
                    # another label printing every named word fits as well: the words that
                    # would tell this row from it are missing
                    covering = [g for g in others if all(_in(w, g.words) for w in hit)]
                    missing = [
                        w
                        for w in scope
                        if covering and w not in hit and not all(_in(w, g.words) for g in covering)
                    ]
                    # a more general level is required only where it is what tells this row from
                    # a sibling printing the same name at this depth under another group
                    # («متراکم» of «سنگدانه متداول» vs «سرباره کوره آهن گدازی»)
                    upper = [w for lv in a.levels[:li] for w in lv]
                    twins = [g for g in covering if _same_name(scope, g.levels[li:])]
                    missing += [
                        w for w in upper if w not in hit and any(not _in(w, g.words) for g in twins)
                    ]
                # a more specific level the query names in part is named by its own head: «طبیعی»
                # of «چوب‌های طبیعی» and a stray «ریز» do not name «درخت صمغی … (برگ ریز)»
                missing += [
                    lv[0]
                    for lv in a.levels[li + 1 :]
                    if lv and lv[0] not in hit and any(w in hit for w in lv)
                ]
                missing += self._condition_missing(row, a, forms)
                cand = Named("head_only" if missing else "exact", hit, missing, level[0])
                if not missing:
                    return cand
                if best.status == "none":
                    best = cand
        return best

    def prints(self, token: str) -> bool:
        """Some printed label of this dataset carries the (merged) token."""
        return any(_same(token, w) for ws in self.printed.values() for w in ws)

    def negations(self, pol: Polarity) -> tuple[list[str], list[list[str]]]:
        """Positive tokens and the extra exclusions of a parsed query on this dataset.

        A negative form the dataset prints («بدون تهویه», «غیر چگالشی») is a required
        qualifier. One it never prints negates its whole phrase: «پلی پروپیلن (بدون الیاف
        شیشه)» excludes the label printing «الیاف شیشه», whose words then name nothing."""
        toks = query_tokens(pol.positive)
        extra: list[list[str]] = []
        for form, scope in zip(pol.negative_forms, pol.negated, strict=True):
            merged = query_tokens(form.text)
            if not merged or self.prints(merged[0]):
                continue
            phrase = query_tokens(scope.text)
            extra.append(phrase)
            for t in (merged[0], *phrase[1:]):
                if t in toks:
                    toks.remove(t)
        return toks, extra

    def _condition_missing(self, row: str, a: Alternative, forms: set[str]) -> list[str]:
        """Condition words the query must name to tell the row from a twin that prints the same
        name with another condition («یک حجم پرلیت …» vs «دو حجم پرلیت …»): a word only this
        row's condition prints, or - when it has none - its whole condition."""
        mine = self.conditions.get(row)
        if mine is None:
            return []
        # rows printing the same name share one alternative: twins are found by row
        twins = [
            r
            for r, alts in self.by_row.items()
            if r != row and a in alts and self.conditions.get(r) != mine
        ]
        missing: list[str] = []
        for t in twins:
            other = self.conditions.get(t, [])
            # words of the row's name («سرباره» of «سنگدانه سرباره …») tell no condition apart
            own = [w for w in mine if not _in(w, other) and not _in(w, a.words)] or mine
            if not any(_named(w, own, forms) for w in own) or (
                own is mine and not all(_named(w, mine, forms) for w in mine)
            ):
                missing += [w for w in own if w not in missing]
        return missing

    def named_words(self, row: str, query: Sequence[str]) -> set[str]:
        """Every printed identity word of the row (any alternative) that the query names."""
        forms = query_forms(query)
        cond = self.conditions.get(row, [])
        return {
            w for a in self.by_row.get(row, []) for w in a.words if _named(w, a.words, forms)
        } | {w for w in cond if _named(w, cond, forms)}

    def excluded(self, row: str, phrases: Sequence[Sequence[str]]) -> list[str]:
        """Excluded phrases (merged tokens) whose every content word the row's label prints."""
        printed = self.printed.get(row, [])
        out = []
        for ph in phrases:
            content = [w for w in ph if w not in FUNCTION_WORDS]
            if content and all(any(_same(w, x) for x in printed) for w in content):
                out.append(" ".join(ph))
        return sorted(set(out))


def query_forms(query: Sequence[str]) -> set[str]:
    """Matching forms of query tokens: keys of every token and joined bigrams/trigrams."""
    out: set[str] = set()
    for i, t in enumerate(query):
        out |= keys(ALIASES.get(t, t))
        for part in _split_number(t):  # «چهارلوله» also names «لوله» (never rewritten)
            out |= keys(part)
        for n in (2, 3):
            if i + n <= len(query):
                out |= keys("".join(query[i : i + n]))
    return out


def _named(w: str, words: Sequence[str], forms: set[str]) -> bool:
    if keys(w) & forms:
        return True
    i = words.index(w)
    for lo, hi in ((i - 1, i + 1), (i, i + 2), (i - 2, i + 1), (i, i + 3)):
        if lo >= 0 and hi <= len(words) and keys("".join(words[lo:hi])) & forms:
            return True
    return False


def query_tokens(text: str) -> list[str]:
    """Query tokens with meaning-bearing prefixes kept on their word."""
    return _merge(etokens(normalize_for_search(text)))


def _split_number(token: str) -> list[str]:
    for word in NUMBER_WORDS:
        rest = token[len(word) :]
        if token.startswith(word) and len(rest) >= 3 and token != word:
            return [word, rest]
    return [token]


# ------------------------------------------------------------------ dataset indexes


def build_indexes(store: StructuredStore) -> dict[str, IdentityIndex]:
    """Identity indexes of the label (main tables), occupancy and material datasets."""
    out: dict[str, IdentityIndex] = {}
    for tid, t in store.main.items():
        if t.labels is None or t.label_rows is None:
            continue
        rows: list[tuple[str, list[list[str]], list[str]]] = []
        for lab in t.labels.labels:
            levels: dict[int, list[str]] = {}
            for w, lv in zip(lab.tokens, lab.levels, strict=True):
                levels.setdefault(lv, []).append(w)
            lv_words = [levels[k] for k in sorted(levels)]
            for pos in t.label_rows[lab.id]:
                rows.append((t.records[pos]["record_id"], lv_words, list(lab.tokens)))
        out[tid] = IdentityIndex(
            tid, rows, _table_ignore(store, tid, t.labels.ignore), coordinated_alternatives=True
        )
    occ_rows = []
    for o in store.occupancy_index:
        label = normalize_for_search(str(o["label"]))
        for alt in label.split("/"):
            toks = _merge([w for w in etokens(alt) if w not in "≥≤<>"])
            occ_rows.append((str(o["id"]), [toks], _merge(etokens(label))))
    out[APP03] = IdentityIndex(
        APP03,
        occ_rows,
        _table_ignore(store, APP03, store.resolver.occupancy_index.ignore),
        coordinated_alternatives=True,
    )
    mat_rows, conditions = _material_rows(store)
    out[APP07] = IdentityIndex(
        APP07,
        mat_rows,
        _table_ignore(store, APP07, store.resolver.material_index.ignore),
        conditions=conditions,
    )
    return out


def _material_rows(
    store: StructuredStore,
) -> tuple[list[tuple[str, list[list[str]], list[str]]], dict[str, list[str]]]:
    """Appendix 7 names per row (category, subgroup, material; each listed name on its own)
    and the printed conditions without a number."""
    from mabhas19.retrieval.planner import condition_quantities  # noqa: PLC0415

    mat_rows: list[tuple[str, list[list[str]], list[str]]] = []
    conditions: dict[str, list[str]] = {}
    for r in store.app07:
        names: list[list[str]] = []
        for key in ("category", "subgroup", "material"):
            v = r.get(key)
            text = v.get("text") if isinstance(v, dict) else None
            if text:
                # a printed list number («۲. بتن و فرآورده‌های بتنی») is not a name
                text = _NUMBERING.sub("", normalize_for_search(text))
                names.append(_merge(etokens(text)))
        cond = (r.get("condition") or {}).get("text") or ""
        raw = [w for lv in names for w in lv] + _merge(etokens(normalize_for_search(cond)))
        if cond and not condition_quantities(r):
            # a printed condition without a number («با ماسهٔ رودخانه‌ای», «CFC؛ با پوسته سطحی»,
            # «دو حجم پرلیت … برای یک حجم گچ») qualifies the row's name
            conditions[str(r["id"])] = _merge(etokens(normalize_for_search(cond)))
        mat_rows.append((str(r["id"]), names, raw))
        # a printed list of names («بلوط، الش، زبان گنجشک، …»): each name identifies the row
        material = (r.get("material") or {}).get("text") or ""
        pieces = _list_items(normalize_for_search(material))
        if len(pieces) > 1 and names:
            for piece in pieces:
                mat_rows.append((str(r["id"]), [*names[:-1], _merge(etokens(piece))], raw))
    return mat_rows, conditions


_NUMBERING = re.compile(r"^\s*\d+\s*[.٫]\s*")


def _list_items(text: str) -> list[str]:
    """Comma-separated items of a printed label, commas inside parentheses kept."""
    items, depth, cur = [], 0, ""
    for ch in text:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch in "،," and depth == 0:
            items.append(cur)
            cur = ""
        else:
            cur += ch
    items.append(cur)
    return [x.strip() for x in items if x.strip()]


def _table_ignore(store: StructuredStore, ds: str, ignore: Iterable[str]) -> set[str]:
    """Ignored words of a label index plus the table's own output names (an output header such
    as «حداکثر ضریب انتقال حرارتی» is not an identity) and generic material words."""
    words = set(ignore)
    schema = store.schemas.get(ds)
    if schema is not None:
        words |= {w for o in schema.outputs for n in (*o.names, *o.cues) for w in etokens(n)}
    if ds == APP07:
        words |= set(store.resolver.generic)
    return words


def row_label(store: StructuredStore, ds: str, r: dict[str, Any]) -> str:
    """The printed label text of a record (for certification), normalized."""
    if ds == APP03:
        return normalize_for_search(
            str(store.occupancies.get(r["occupancy_id"], {}).get("label", ""))
        )
    if ds == APP07:
        parts = [
            (r.get(k) or {}).get("text") or ""
            for k in ("category", "subgroup", "material", "condition")
        ]
        return normalize_for_search(" ".join(parts))
    from mabhas19.retrieval.structured import LABEL_FIELDS, _field  # noqa: PLC0415

    fields = LABEL_FIELDS.get(ds.removeprefix(MAIN), ())
    return normalize_for_search(" ".join(x for path, _ in fields if (x := _field(r, path))))
