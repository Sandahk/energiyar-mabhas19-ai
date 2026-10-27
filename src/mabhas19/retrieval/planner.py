"""Structured query planner: query text -> one typed :class:`StructuredQueryPlan` per dataset.

The planner interprets the structured part of a query against one dataset's schema and records
every meaningful component with a state; it never returns records. Shared pre-pass (spans of the
normalized text are *consumed* by what interprets them):

* references (table/appendix/chapter/section numbers, «مبحث ۱۹», calendar years) - consumed;
* climate classes, ratings, building groups, years, station codes - category constraints
  validated against the schema domains;
* quantities (:mod:`mabhas19.retrieval.numeric`) - attached to a dimension or output by cue
  words / units, to a printed numeric condition inside a row label, or left **unresolved**;
* computation requests (totals, averages, ratios, interpolation, area products…);
* negation («غیر», «بدون», «نه», «به جز») and count qualifiers («تک/دو/چند …»).

Family planners (``schema.family``): ``place`` (Appendix 2, 19-3-2), ``eui`` (Appendix 3),
``material`` (Appendix 7), ``label`` (19-2-1, 19-5-x), ``criterion`` (19-3-1) and
``rating_range`` (19-3-3). A row is identified only when its printed label is fully named; a
distinguishing word of a sibling row, a contradicting count, an unknown word glued to the name
or a number that nothing interprets leaves the plan unresolved, so the gate fails closed.
Geography never infers a value: a town that is not printed is unknown even when its province
or province capital is named.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from mabhas19.retrieval import polarity
from mabhas19.retrieval import ratings as rating_parser
from mabhas19.retrieval.answerability import has_words
from mabhas19.retrieval.dimensions import (
    GROUP_MORE,
    groups_requested,
    ratings_requested,
    years_requested,
)
from mabhas19.retrieval.entities import (
    FUNCTION_WORDS,
    NEGATORS,
    Label,
    LabelIndex,
    Match,
    etokens,
    keys,
)
from mabhas19.retrieval.identity import build_indexes
from mabhas19.retrieval.identity import query_tokens as identity_tokens
from mabhas19.retrieval.numeric import (
    OPEN,
    NumInterval,
    Quantity,
    find_quantities,
    printed_interval,
)
from mabhas19.retrieval.operations import requested as requested_operation
from mabhas19.retrieval.plan import (
    Constraint,
    Derivation,
    EntityRef,
    Enumeration,
    Qualifier,
    StructuredQueryPlan,
)
from mabhas19.retrieval.schema import APP03, APP07, MAIN, TableSchema
from mabhas19.text.normalize import normalize_for_search

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import MainTable, StructuredStore

# ------------------------------------------------------------------ shared vocabulary

_CLIMATE = re.compile(r"(?<![a-z0-9])(\d)\s?([a-z])(?![a-z0-9/])")
_RATING_SPAN = re.compile(
    r"(?:رده|رتبه|کلاس)\s*(?:بازدهی\s*)?(?:انرژی\s*)?[a-z]\+{0,3}(?![a-z0-9+])"
)
_GROUP_SPAN = re.compile(
    r"گروه\s*(?:های\s*)?(?:ساختمانی\s*|ساختمان\s*(?:های\s*)?)?[«\"'(]?\s*(الف|ب|پ|ج|د|ه|و|ی)\s*[»\"')]?(?![ؠ-ۿ])"
)
_GROUP_MENTION = re.compile(r"گروه\s*(?:های\s*)?(?:ساختمانی|ساختمان)?\s*[«\"'(]?\s*\S+")
_YEAR_SPAN = re.compile(r"سال\s*(?:اول|یکم|دوم|سوم|چهارم|پنجم|ششم|هفتم|\d+)(?![ؠ-ۿ\d])")
_WMO = re.compile(r"(?<!\d)\d{6}(?!\d)")
_REFERENCE = re.compile(
    r"(?:جدول|پیوست|فصل|بند|بخش|ماده|تبصره|صفحه|ردیف|ویرایش|مبحث|شماره)\s*(?:شماره\s*)?"
    r"(?:پ\s?)?\d+(?:\s?[-–]\s?\d+)*"
    r"|(?<![\d-])19(?:\s?-\s?\d+){1,5}(?![\d-])|پ\s?\d+(?:-\d+)+"
    r"|(?:سال|مصوب|ویرایش)\s*1[34]\d\d"
)
_EXISTENCE = ("وجود دارد", "وجود داره", "هست", "هستش", "داریم", "پیدا میشه", "پیدا می شود")
_TOTAL_ENERGY = re.compile(r"(?:کیلووات\s?ساعت|kwh)(?!\s*(?:بر|در|/)\s*(?:متر|m2))")
# nouns that head a place name («شهر یزد», «بندر انزلی»)
PLACE_MARKERS = frozenset({"شهر", "شهرستان", "روستا", "روستای", "بخش", "شهرک", "بندر"})
# measured quantities a question may ask for (a table must print the one asked)
QUANTITY_WORDS = frozenset(
    {
        "نشت",
        "دما",
        "دمای",
        "فشار",
        "قطر",
        "ضخامت",
        "سرعت",
        "دبی",
        "ابعاد",
        "طول",
        "ارتفاع",
        "وزن",
        "قیمت",
        "هزینه",
        "عمر",
        "صدا",
        "روشنایی",
        "لوکس",
        "تعداد",
        "شیب",
        "عرض",
        "دلار",
        "تومان",
        "تومن",
        "ریال",
        "یورو",
    }
)
_NEGATION_WORDS = ("نه", "به جز", "بجز", "جز", "منهای", "به غیر از", "بغیر از")
_COUNT = {"تک": "1", "یک": "1", "دو": "2", "سه": "3", "چهار": "4", "چند": "n"}
_GENERIC = frozenset(
    {
        "ساختمان",
        "ساختمانی",
        "ساختمانها",
        "ساختمانهای",
        "مبحث",
        "مقررات",
        "جدول",
        "پیوست",
        "طبق",
        "مطابق",
        "حداقل",
        "حداکثر",
        "مقدار",
        "میزان",
        "مجاز",
        "لازم",
        "باید",
        "اقلیم",
        "اقلیمی",
        "رده",
        "منطقه",
    }
)


@dataclass
class Features:
    """Shared interpretation of one structured query text."""

    norm: str
    tokens: list[str]
    consumed: list[tuple[int, int]] = field(default_factory=list)
    quantities: list[Quantity] = field(default_factory=list)
    climates: list[str] = field(default_factory=list)
    ratings: list[str] = field(default_factory=list)
    rating_mentioned: bool = False
    rating_conflicts: list[str] = field(default_factory=list)
    rating_mentions: list[rating_parser.RatingMention] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    group_mentioned: bool = False
    years: list[int] = field(default_factory=list)
    wmo: list[str] = field(default_factory=list)
    computation: str | None = None
    existence: bool = False
    entity_text: str = ""  # the structured part: entities and row labels are read from it

    def free(self, q: Quantity) -> bool:
        return all(q.end <= a or q.start >= b for a, b in self.consumed)


def features(norm: str) -> Features:
    f = Features(norm, etokens(norm))
    for m in _REFERENCE.finditer(norm):
        f.consumed.append(m.span())
    for m in _CLIMATE.finditer(norm):
        f.climates.append(f"{m[1]}{m[2].upper()}")
        f.consumed.append(m.span())
    # «5b,5c» / «2a و 2b»: every class is listed above
    f.climates = sorted(set(f.climates))
    for m in _RATING_SPAN.finditer(norm):
        f.consumed.append(m.span())
    for mention in rating_parser.parse(norm):
        f.consumed.append((mention.start, mention.end))
        f.rating_mentions.append(mention)
        if mention.conflict:
            f.rating_conflicts.append(f"{mention.text}: {mention.conflict}")
    f.ratings = ratings_requested(norm)
    f.rating_mentioned = bool(re.search(r"(?:رده|رتبه)(?!\s*(?:اقلیم|بازدهی انرژی\s*[a-z]))", norm))
    for m in _GROUP_SPAN.finditer(norm):
        f.groups.append(m[1])
        f.consumed.append(m.span())
        pos = m.end()
        while more := GROUP_MORE.match(norm, pos):  # «گروه‌های ج و د»
            f.groups.append(more[1])
            f.consumed.append(more.span())
            pos = more.end()
    f.groups = sorted(set(f.groups) | set(groups_requested(norm)))
    f.group_mentioned = bool(
        re.search(r"گروه\s*(?:های\s*)?(?:ساختمانی\s*)?[«\"'(]?\s*[ؠ-ۿ]{1,3}(?![ؠ-ۿ])", norm)
    )
    for m in _YEAR_SPAN.finditer(norm):
        f.consumed.append(m.span())
    f.years = years_requested(norm)
    for m in _WMO.finditer(norm):
        f.wmo.append(m[0])
        f.consumed.append(m.span())
    f.quantities = [q for q in find_quantities(norm) if f.free(q)]
    f.computation = requested_operation(norm)  # typed operation parser (never executed)
    f.existence = bool(has_words(norm, _EXISTENCE))
    return f


def _near_before(norm: str, q: Quantity, cues: Iterable[str], window: int = 40) -> str | None:
    """The cue that ends closest before a quantity, with only filler in between."""
    found = _cue_gap(norm, q, cues, window)
    return found[1] if found else None


def _cue_gap(
    norm: str, q: Quantity, cues: Iterable[str], window: int = 40
) -> tuple[int, str] | None:
    """(gap in characters, cue) of the nearest cue before a quantity (ties: longer cue)."""
    head = norm[max(0, q.start - window) : q.start]
    best: tuple[int, int, str] | None = None
    for cue in cues:
        c = normalize_for_search(cue)
        for m in re.finditer(rf"(?<![ؠ-ۿa-z]){re.escape(c)}(?![a-z])", head):
            gap = head[m.end() :]
            if re.search(r"\d", gap) or len(etokens(gap)) > 4:
                continue
            key = (len(gap.strip()), -len(c), cue)
            if best is None or key < best:
                best = key
    return (best[0], best[2]) if best else None


def _keyset(words: Iterable[str]) -> set[str]:
    return {k for w in words for k in keys(w)}


def _names_all(words: Sequence[str], window: set[str], *, any_word: bool = False) -> bool:
    """Every (or, with ``any_word``, some) printed name word is in the window, alone or written
    solid with its printed neighbour («میل گرد» ~ «میلگرد»)."""
    hits = [
        bool(keys(w) & window)
        or any(
            j >= 0 and j + 1 < len(words) and keys(words[j] + words[j + 1]) & window
            for j in (i - 1, i)
        )
        for i, w in enumerate(words)
    ]
    return any(hits) if any_word else all(hits)


# ------------------------------------------------------------------ planner


class Planner:
    def __init__(self, store: StructuredStore, schemas: dict[str, TableSchema]) -> None:
        self.store = store
        self.schemas = schemas
        self.provinces = {
            normalize_for_search(r["province"]["text"]) for r in store.main[MAIN + "19-3-2"].records
        }
        # (name, cues, weak cues, units) of every dimension/output, for cue competition
        self.cue_sets: dict[
            str, list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]]
        ]
        self.cue_sets = {
            ds: [(d.name, d.cues, d.weak_cues, d.units) for d in s.dimensions]
            + [(o.name, o.cues, o.weak_cues, o.units) for o in s.outputs]
            for ds, s in schemas.items()
        }
        # words naming a table's outputs (never a distinguishing row word)
        self.output_words: dict[str, set[str]] = {
            ds: _keyset(w for o in s.outputs for n in (*o.names, *o.cues) for w in etokens(n))
            for ds, s in schemas.items()
        }
        self._current = ""
        # source-exact identity of printed rows (normalization, morphology, reviewed aliases)
        self.identity = build_indexes(store)
        self._vocab_keys: set[str] | None = None
        self._vocab_size = -1
        # quantity vocabulary of every dataset (outputs, dimension cues), for foreign-output checks
        self.own_words: dict[str, set[str]] = {}
        for ds, s in schemas.items():
            words: set[str] = set()
            for o in s.outputs:
                words |= {w for n in (*o.names, *o.cues) for w in etokens(n)}
            for d in s.dimensions:
                words |= {w for n in (*d.cues, *d.weak_cues) for w in etokens(n)}
            if ds in store.main:
                t = store.main[ds]
                words |= set(t.caption_words)
                if t.labels:
                    words |= t.labels.vocabulary
            elif s.family == "material":  # printed material and condition words are inputs
                words |= store.resolver.material_index.vocabulary
            elif s.family == "eui":
                words |= store.resolver.occupancy_index.vocabulary
            self.own_words[ds] = _keyset(words)

    def vocab_keys(self) -> set[str]:
        """Matching keys of the corpus vocabulary (computed once; the router sets it)."""
        vocab = self.store.resolver.vocabulary
        if self._vocab_keys is None or self._vocab_size != len(vocab):
            self._vocab_keys = _keyset(vocab)
            self._vocab_size = len(vocab)
        return self._vocab_keys

    # ------------------------------------------------------------------ identity
    def identify(
        self, dataset: str, text: str, full: str | None = None
    ) -> tuple[list[str], list[str], list[str]]:
        """(exactly named rows, rows whose head only is named, excluded phrases) of a dataset
        for the positive text of a query (exclusions removed and returned). A row naming a
        strict subset of the query words another exact row names is not chosen («ماسه» beside
        «ماسه سنگ آهکی»; «الیافی» beside «… تقویت‌شده با الیاف معدنی»)."""
        pol = polarity.parse(text)
        idx = self.identity[dataset]
        toks, negated = idx.negations(pol)
        # an exclusion anywhere in the question applies («… (نه X)» cut by a clause break)
        exclusions = polarity.parse(full).exclusions if full else pol.exclusions
        excl = [identity_tokens(e.text) for e in exclusions] + negated
        exact: dict[str, frozenset[int]] = {}
        head: list[str] = []
        for row in idx.by_row:
            if idx.excluded(row, excl):
                continue
            nm = idx.named(row, toks)
            if nm.status == "exact":
                exact[row] = idx.named_positions(row, toks)
            elif nm.status == "head_only":
                head.append(row)
        best = [r for r, ws in exact.items() if not any(ws < o for o in exact.values())]
        return best, head, [e.text for e in exclusions] + [" ".join(x) for x in negated]

    # ------------------------------------------------------------------ entry
    def plan(
        self,
        dataset: str,
        norm: str,
        *,
        explicit_reference: bool = False,
        full_norm: str | None = None,
    ) -> StructuredQueryPlan:
        """The plan for one dataset. ``norm`` is the structured part of the query; ``full_norm``
        (the whole query) is also checked for computation requests."""
        schema = self.schemas[dataset]
        # every constraint stated anywhere in the question is interpreted (a decomposition must
        # never drop one); entities and row labels are read from the structured part
        f = features(full_norm or norm)
        f.entity_text = norm
        p = StructuredQueryPlan(dataset, explicit_reference=explicit_reference)
        p.computation = self._computation(f, schema)
        self._requested_field(p, f, schema)
        family = getattr(self, f"_plan_{schema.family}")
        family(p, f, schema)
        for q in f.quantities:
            if f.free(q) and (q.raw.strip() in norm or _coordinated_with_consumed(f, q)):
                p.unresolved.append(f"number «{q.raw.strip()}»")
        for e in p.enumerations:
            e.parent = "; ".join(x.text for x in p.entities if x.status == "resolved" and x.text)
            e.members = self._members(p, schema, e.dimension)
        return p

    def _computation(self, f: Features, schema: TableSchema) -> str | None:
        if f.computation:
            return f.computation
        area = [q for q in f.quantities if q.unit == "m2"]
        if schema.family == "eui" and area and _TOTAL_ENERGY.search(f.norm):
            return "area_product"
        return None

    def _requested_field(self, p: StructuredQueryPlan, f: Features, schema: TableSchema) -> None:
        """The output the question asks for. A measured quantity or currency this table does
        not print decides first (-> unsupported_output, even when the table's own output is
        mentioned: «R … چند دلار است» asks a price); else a named output; else the table's
        output."""
        own = self.own_words[schema.dataset]
        asked = etokens(f.entity_text) if f.entity_text else f.tokens  # the structured clause
        foreign = [w for w in asked if w in QUANTITY_WORDS and not keys(w) & own]
        if foreign:
            p.requested_field = foreign[0]
            return
        for o in schema.outputs:
            if has_words(f.norm, o.names):
                p.requested_field = o.name
                return
        p.requested_field = schema.outputs[0].name

    # ------------------------------------------------------------------ category dimensions
    def _categories(  # noqa: PLR0912
        self, p: StructuredQueryPlan, f: Features, schema: TableSchema
    ) -> None:
        """Climate class / rating / group / year: validated against the printed domains."""
        dim = schema.dim("climate_class")
        if f.climates:
            if dim is None:
                p.notes.append(
                    f"the table has no climate dimension; {f.climates} does not filter it"
                )
                p.constraints.append(
                    Constraint("climate_class", f.climates, state="unsupported_by_table")
                )
                p.constraints.append(
                    Constraint("climate_class", f.climates, state="unsupported_by_table")
                )
            else:
                bad = [c for c in f.climates if c not in dim.domain]
                state, detail = (
                    ("invalid", f"climate_class_not_in_table: {bad} not printed ({dim.domain})")
                    if bad
                    else ("applied", "")
                )
                p.constraints.append(
                    Constraint(
                        "climate_class",
                        f.climates,
                        state=state,
                        detail=detail,
                        reason=_reason(detail),
                    )
                )
        elif dim is not None and dim.enumerable:
            self._enumerate(p, f, schema, "climate_class")
        dim = schema.dim("rating")
        if f.rating_conflicts and dim is not None:
            detail = f"rating_conflict: {'; '.join(f.rating_conflicts)}"
            p.constraints.append(
                Constraint("rating", [], state="invalid", detail=detail, reason="rating_conflict")
            )
        elif f.ratings:
            if dim is None:
                p.notes.append(f"the table has no rating dimension; {f.ratings} does not filter it")
                p.constraints.append(Constraint("rating", f.ratings, state="unsupported_by_table"))
            else:
                bad = [r for r in f.ratings if r not in dim.domain]
                state, detail = (
                    ("invalid", f"rating_not_in_table: {bad} not printed ({dim.domain})")
                    if bad
                    else ("applied", "")
                )
                ok = [m for m in f.rating_mentions if m.conflict is None]
                described = [m for m in ok if m.description]
                p.constraints.append(
                    Constraint(
                        "rating",
                        f.ratings,
                        source="; ".join(m.text for m in ok),
                        state=state,
                        detail=detail,
                        reason=_reason(detail),
                        rule="rating_description" if described else "rating_letter",
                        attested_in=sorted({b for m in described for b in m.attested_in}),
                    )
                )
        elif dim is not None and dim.enumerable and self._enumerate(p, f, schema, "rating"):
            pass  # «همه رده‌ها»: explicitly requested alternatives
        elif dim is not None and f.rating_mentioned and schema.family == "eui":
            p.constraints.append(
                Constraint(
                    "rating",
                    [],
                    state="unresolved",
                    detail="a rating is referred to but not stated",
                )
            )
        elif dim is not None and dim.enumerable:
            self._enumerate(p, f, schema, "rating")
        for name, values, mentioned in (
            ("group", f.groups, f.group_mentioned),
            ("year", [str(y) for y in f.years], bool(f.years)),
        ):
            dim = schema.dim(name)
            if dim is None:
                if values:
                    p.notes.append(
                        f"the table has no {name} dimension; {values} does not filter it"
                    )
                    p.constraints.append(Constraint(name, values, state="unsupported_by_table"))
                continue
            if not values:
                if dim.enumerable and self._enumerate(p, f, schema, name):
                    continue  # «همه سال‌ها», «گروه‌های مختلف»: explicitly requested alternatives
                if mentioned:
                    p.constraints.append(
                        Constraint(
                            name, [], state="unresolved", detail=f"{name} mentioned but not read"
                        )
                    )
                continue
            bad = [v for v in values if v not in dim.domain]
            state, detail = (
                ("invalid", f"{name}_not_in_table: {bad} not printed ({dim.domain})")
                if bad
                else ("applied", "")
            )
            p.constraints.append(
                Constraint(
                    name,
                    values,
                    state=state,
                    detail=detail,
                    reason=_reason(detail),
                    rule="coordinated_values" if len(values) > 1 else "category",
                )
            )

    # ------------------------------------------------------------------ numeric attachment
    def _attach(  # noqa: PLR0913
        self,
        p: StructuredQueryPlan,
        f: Features,
        name: str,
        cues: Sequence[str],
        units: Sequence[str],
        *,
        weak_cues: Sequence[str] = (),
        reverse: bool = False,
    ) -> Constraint | None:
        """Attach the first free quantity cued by ``cues``/``units`` to a dimension/output: the
        nearest cue among all of the table's dimensions and outputs owns a number, and a number
        whose unit belongs to another dimension never attaches."""
        others = [x for x in self.cue_sets.get(p.target, []) if x[0] != name]
        for q in f.quantities:
            if not f.free(q):
                continue
            if reverse and q.raw.strip() not in f.entity_text:
                continue  # an output value is read only from the structured clause
            if q.unit and units and q.unit not in units:
                continue
            if q.unit and not units and any(q.unit in x[3] for x in others):
                continue
            mine = _cue_gap(f.norm, q, cues)
            if mine is None and weak_cues:
                weak = _cue_gap(f.norm, q, weak_cues)
                if weak is not None and not p.explicit_reference:
                    continue
                mine = weak
            if mine is None and not (q.unit and q.unit in units):
                continue
            if mine is not None and any(
                (g := _cue_gap(f.norm, q, x[1])) is not None and g[0] < mine[0] for x in others
            ):
                continue  # another dimension's cue is nearer: the number is not this one's
            f.consumed.append((q.start, q.end))
            c = Constraint(name, interval=q.interval, source=q.raw.strip(), rule="cue")
            if q.ratio:
                c.ratio = (str(q.ratio[0]), str(q.ratio[1]))
            _coordinated_bound(f, q, c)
            p.constraints.append(c)
            if reverse:
                p.operation = "reverse"
            return c
        return None

    # ================================================================== families
    # ------------------------------------------------------------------ places
    def _plan_place(  # noqa: PLR0911
        self, p: StructuredQueryPlan, f: Features, schema: TableSchema
    ) -> None:
        self._categories(p, f, schema)
        if schema.dim("wmo_code") and f.wmo:
            p.constraints.append(Constraint("wmo_code", f.wmo))
        places, unknown = self.resolve_places(f.entity_text)
        provinces = self.named_provinces(f.entity_text)
        if not places and not unknown and provinces:
            if schema.dim("province") is not None:
                # «استان X»: the table prints a province column; its row is the printed answer
                p.constraints.append(Constraint("province", provinces, source="استان"))
                return
            p.entities.append(
                EntityRef(
                    "place",
                    ", ".join(provinces),
                    "unknown",
                    detail="a province is named; this table prints cities only and no city is "
                    "inferred from a province",
                )
            )
            return
        if not places and not unknown:
            unknown = self._unknown_word(f.entity_text, schema.dataset)
        if unknown:
            p.entities.append(
                EntityRef(
                    "place",
                    unknown,
                    "unknown",
                    detail=f"«{unknown}» is not a printed place; no value is inferred from "
                    "its province or neighbours",
                )
            )
            return
        if places:
            p.entities.append(EntityRef("place", ", ".join(places), "resolved", places))
            cc = p.constraint("climate_class")
            if cc is not None and cc.state == "applied":
                cc.state = "checked"  # «is X in 3B?»: the place's printed class answers it
                p.operation = "existence"
            return
        if p.constraint("wmo_code"):
            return
        cc = p.constraint("climate_class")
        if cc is not None:
            p.operation = "reverse"
            return
        if p.explicit_reference and _whole_request(f):
            p.operation = "whole_table"
            return
        p.entities.append(EntityRef("place", "", "none"))

    def named_provinces(self, norm: str) -> list[str]:
        """Provinces named as «استان X» (printed in table 19-3-2's province column)."""
        toks = etokens(norm)
        out = []
        for i, t in enumerate(toks[:-1]):
            if t != "استان":
                continue
            for n in (3, 2, 1):
                name = " ".join(toks[i + 1 : i + 1 + n])
                if normalize_for_search(name) in self.provinces:
                    out.append(normalize_for_search(name))
                    break
        return list(dict.fromkeys(out))

    def resolve_places(self, norm: str) -> tuple[list[str], str | None]:
        """(curated places asked about, unknown place name or None). A curated place counts as
        a qualifier - not the asked place - after «استان», inside parentheses, or right after an
        unknown place-like word («بافق (یزد)», «اهرم بوشهر», «بافت تو استان کرمان»)."""
        toks = etokens(norm)
        vocab = self.store.resolver.vocabulary
        keys_ = self.store.resolver.place_keys
        found: list[tuple[int, int, str]] = []
        for i in range(len(toks)):
            for n in (3, 2, 1):
                if i + n <= len(toks) and (alias := keys_.get("".join(toks[i : i + n]))):
                    found.append((i, i + n, alias))
                    break
        found = [
            x for x in found if not any(g != x and g[0] <= x[0] and x[1] <= g[1] for g in found)
        ]
        paren = {normalize_for_search(m) for m in re.findall(r"\(([^)]*)\)", norm)}
        asked: list[str] = []
        unknown: str | None = None
        marker = PLACE_MARKERS
        starts = {a for a, _, _ in found}
        for i, t in enumerate(toks[:-1]):
            nxt = toks[i + 1]
            if any(a <= i < b for a, b, _ in found):
                continue  # the marker is part of a printed place name («بندر لنگه»)
            if (
                t in marker
                and i + 1 not in starts
                and nxt not in FUNCTION_WORDS
                and not nxt.isdigit()
                and nxt not in _GENERIC
                and nxt not in {"های", "ها", "ی"}
            ):
                unknown = nxt
        for a, _b, alias in found:
            before = toks[a - 1] if a > 0 else ""
            qualifier = before == "استان" or any(alias in x for x in paren)
            if a > 0 and toks[a - 1] not in vocab and toks[a - 1] not in FUNCTION_WORDS:
                prev = toks[a - 1]
                if (
                    alias in self.provinces
                    and prev not in _GENERIC
                    and not keys(prev) & self.own_words.get(MAIN + "19-3-2", set())
                    and prev not in marker
                ):
                    unknown = unknown or prev
                    qualifier = True
            if not qualifier:
                asked.append(alias)
        if unknown and unknown in asked:
            unknown = None
        return list(dict.fromkeys(asked)), unknown

    def _unknown_word(self, text: str, dataset: str) -> str | None:
        """A content word of the structured part that names nothing this table prints (a place,
        space or product it does not list); None when every word is explained."""
        own = self.own_words[dataset]
        vocab = self.store.resolver.vocabulary
        place_words = (
            {w for a in self.store.places for w in etokens(a)}
            if dataset.endswith(("app02.city_climate_classes", "t19-3-2"))
            else set()
        )
        for w in etokens(text):
            if (
                w in FUNCTION_WORDS
                or w in _GENERIC
                or w in _WHOLE_WORDS
                or w in QUANTITY_WORDS
                or w.replace(".", "").isdigit()
                or re.fullmatch(r"\d[a-z]|[a-z]\+*|ه|ی", w)
                or keys(w) & own
                or w in place_words
            ):
                continue
            if dataset.endswith(("app02.city_climate_classes", "t19-3-2")):
                if vocab and w not in vocab and not keys(w) & self.vocab_keys():
                    return w
                continue
            if keys(w) & _STRUCTURAL_WORDS:
                continue
            return w
        return None

    # ------------------------------------------------------------------ appendix 3
    def _plan_eui(  # noqa: PLR0912, PLR0915
        self, p: StructuredQueryPlan, f: Features, schema: TableSchema
    ) -> None:
        out = schema.outputs[0]
        # an EUI value with an identified occupancy is a claim to check; without one it is a
        # reverse lookup, which the gate refuses (no inversion defined)
        self._attach(p, f, "max_eui", out.cues, out.units, reverse=True)
        self._categories(p, f, schema)
        store = self.store
        exact, head, excluded = self.identify(APP03, f.entity_text, f.norm)
        for ph in excluded:
            p.qualifiers.append(Qualifier(ph, "negative", "applied"))
        enumerated = False
        if exact:
            ids = sorted(exact, key=lambda o: list(store.occupancies).index(o))
        elif len(head) > 1 and self._enumerate(p, f, schema, "occupancy"):
            ids, enumerated = sorted(head), True  # «به تفکیک کاربری»: every variant of the head
        elif head:
            p.entities.append(
                EntityRef(
                    "occupancy",
                    ", ".join(sorted(store.occupancies[o]["label"] for o in head)),
                    "ambiguous",
                    sorted(head),
                    detail="the named occupancy fits several printed variants the question "
                    "does not distinguish",
                )
            )
            return
        else:
            p.entities.append(
                EntityRef(
                    "occupancy",
                    "",
                    "unknown",
                    detail="no printed occupancy is named exactly (normalization, morphology or "
                    "a reviewed alias)",
                )
            )
            return
        # several different occupancies tied (not size variants of one label) -> ambiguous
        bases = {re.sub(r"[≥≤<>]?\s*\d.*$", "", store.occupancies[o]["label"]).strip() for o in ids}
        if len(bases) > 1 and not enumerated:
            # variants of one printed head word that the question does not tell apart
            # («اداری» -> عمومی/خصوصی) are enumerated, each record carrying its own label;
            # different occupancies are ambiguous
            heads = {etokens(b)[0] for b in bases if etokens(b)}
            idx = store.resolver.occupancy_index
            q = idx.query_tokens(f.entity_text)
            matched = {frozenset(m.query_idx) for m in idx.match(q) if m.label.id in set(ids)}
            if not (
                len(heads) == 1 and len(matched) == 1 and self._enumerate(p, f, schema, "occupancy")
            ):
                p.entities.append(
                    EntityRef("occupancy", ", ".join(sorted(bases)), "ambiguous", ids)
                )
                return
        area = self._attach(p, f, "area", schema.dim("area").cues, ("m2",))  # type: ignore[union-attr]
        sized = {
            o: printed_interval(normalize_for_search(store.occupancies[o]["label"])) for o in ids
        }
        if area is not None and area.interval is not None:
            keep = [
                o
                for o, iv in sized.items()
                if iv is None or iv.relation(area.interval) in ("within", "boundary")
            ]
            if not keep:
                area.state = "applied"
                p.row_ids = []
                p.entities.append(EntityRef("occupancy", ", ".join(ids), "resolved", ids))
                p.notes.append(
                    f"no printed size condition of the occupancy covers {area.interval.describe()}"
                )
                return
            ids = keep
        elif sum(1 for v in sized.values() if v) > 1 and not self._enumerate(p, f, schema, "area"):
            p.missing.append("area")  # the printed values differ by size
        self._qualifiers(p, f, store.resolver.occupancy_index, ids)
        p.entities.append(EntityRef("occupancy", ", ".join(ids), "resolved", ids))
        if not f.climates:
            places, unknown = self.resolve_places(f.entity_text)
            if unknown:
                p.entities.append(
                    EntityRef(
                        "place", unknown, "unknown", detail=f"«{unknown}» is not a printed place"
                    )
                )
                return
            codes: list[str] = []
            used: list[str] = []
            issues: list[str] = []
            for place in places:
                for _ds, r in store.places[place]:
                    used.append(str(r.get("id") or r.get("record_id")))
                    issues += list(r.get("source_issue_refs") or [])
                    if r["climate_class"]["code"] not in codes:
                        codes.append(r["climate_class"]["code"])
            if codes:
                p.derivations.append(
                    Derivation("climate_from_place", used, ",".join(codes), sorted(set(issues)))
                )
                p.notes.append(
                    "climate class derived from the named place: "
                    + ", ".join(places)
                    + f" -> {', '.join(codes)} ({', '.join(used)})"
                )
                dim = schema.dim("climate_class")
                bad = [c for c in codes if dim is not None and c not in dim.domain]
                p.constraints.append(
                    Constraint(
                        "climate_class",
                        codes,
                        source=", ".join(places),
                        state="invalid" if bad else "applied",
                        detail=f"climate_class_not_in_table: {bad}"
                        if bad
                        else "derived from the named place",
                        reason="climate_class_not_in_table" if bad else "",
                    )
                )
        p.row_ids = ids
        self._claims(p, f, "max_eui")

    def _enumerate(
        self, p: StructuredQueryPlan, f: Features, schema: TableSchema, dimension: str
    ) -> bool:
        """Record an explicitly requested enumeration of a schema-enumerable dimension; False
        (nothing recorded) when the dimension is not enumerable or the question does not ask
        for the printed alternatives."""
        d = schema.dim(dimension)
        if d is None or not d.enumerable:
            return False
        request = enumeration_request(f.norm, d.nouns)
        if request is None:
            return False
        p.enumerations.append(Enumeration(dimension, "explicit_request", request))
        return True

    def _members(self, p: StructuredQueryPlan, schema: TableSchema, dimension: str) -> list[str]:
        """Printed members of an enumerated dimension within the plan's resolved scope."""
        from mabhas19.retrieval.executor import enumeration_value  # noqa: PLC0415

        ds = p.target
        s = self.store
        if ds == APP03:
            occ = set(p.row_ids or [])
            rows = [r for r in s.app03 if r["occupancy_id"] in occ]
        elif ds == APP07:
            ids = set(p.row_ids or [])
            rows = [r for r in s.app07 if r["id"] in ids]
        elif ds in s.main:
            ids = set(p.row_ids or [])
            rows = [r for r in s.main[ds].records if not ids or r["record_id"] in ids]
        else:
            d = schema.dim(dimension)
            return list(d.domain) if d else []
        values = {enumeration_value(s, ds, r, dimension) for r in rows}
        return sorted(v for v in values if v)

    def _claims(self, p: StructuredQueryPlan, f: Features, output: str) -> None:
        """A number that claims the printed value of an identified row («… همون ۱٫۰۹ هست؟»,
        «… ۴۳ درجه‌ست دیگه؟») is checked against the returned record, never used to filter or
        to reverse-look-up: the row is identified by its dimensions, the claim is ``checked``.
        When the rows are not fully identified (an enumerated dimension), the stated value
        narrows them to the rows printing it («… تو همه اقلیم‌ها ۰٫۵۷ ـه؟»)."""
        narrowing = bool(p.enumerations)
        if not narrowing:
            for c in p.constraints:
                if c.dimension == output and c.state == "applied":
                    c.state = "checked"
                    p.operation = "existence"
        for q in f.quantities:
            if not f.free(q) or q.raw.strip() not in f.entity_text:
                continue
            before = f.norm[max(0, q.start - 14) : q.start]
            after = f.norm[q.end : q.end + 16]
            if re.search(r"(?:همون|همان|عدد|مقدار)\s*$", before) or re.match(
                r"\s*(?:[ؠ-ۿ]{0,12}\s*)?(?:است|هست|ـه|ه|ست)?\s*[،,]?\s*"
                r"(?:دیگه|دیگر|درسته|صحیحه|؟|\?)",
                after,
            ):
                f.consumed.append((q.start, q.end))
                state = "applied" if narrowing else "checked"
                p.constraints.append(
                    Constraint(
                        output, interval=q.interval, source=q.raw.strip(), state=state, rule="claim"
                    )
                )
                p.operation = "reverse" if narrowing else "existence"

    # ------------------------------------------------------------------ appendix 7
    def _plan_material(  # noqa: PLR0912, PLR0915
        self, p: StructuredQueryPlan, f: Features, schema: TableSchema
    ) -> None:
        store = self.store
        lam = schema.output("thermal_conductivity")
        assert lam is not None
        self._categories(p, f, schema)
        # colloquial function words («همون», «تو») never name or block a material
        text = " ".join(w for w in f.entity_text.split() if w not in _COLLOQUIAL)
        # identity reads the printed words as stated («یک حجم پرلیت … برای یک حجم گچ»)
        exact, head, excluded = self.identify(APP07, f.entity_text, f.norm)
        for ph in excluded:
            p.qualifiers.append(Qualifier(ph, "negative", "applied"))
        res = store.resolver.materials(polarity.parse(text).positive)
        rows = [r["id"] for r in store.app07]
        if exact:
            # a printed material named exactly (its density/condition variants stay candidates)
            labels = [store.app07[store._app07_pos[i]]["display_name"] for i in exact]
            res = type(res)("resolved", exact, labels, res.leftover, res.notes, res.query_tokens)
        elif head and _LAMBDA.search(f.norm):
            # only a category is named («کدام عایق‌ها …»): no material identity; a stated λ is
            # looked up within that category
            rows = sorted(head, key=lambda i: store._app07_pos[i])
            res = type(res)("none", query_tokens=res.query_tokens)
        elif head:
            names = sorted({store.app07[store._app07_pos[i]]["display_name"] for i in head})
            res = type(res)("ambiguous", leftover=names[:5], notes=[
                "the named material fits several printed rows the question does not distinguish"
            ], query_tokens=res.query_tokens)  # fmt: skip
        elif res.status == "resolved":
            # words of a label matched, but no printed material is named exactly
            res = type(res)(
                "unknown",
                leftover=res.leftover or res.labels[:1],
                notes=["no printed material is named exactly"],
                query_tokens=res.query_tokens,
            )
        # numeric conditions printed inside the candidate rows («چگالی اسمی: ۱۱۰۰», «نسبت: ۱ به ۶»)
        candidate = res.ids if res.status == "resolved" else rows
        self._conditions(p, f, candidate)
        lam_c = self._attach(p, f, "thermal_conductivity", lam.cues, ("w/mk",), reverse=True)
        dens = schema.dim("density")
        assert dens is not None
        self._attach(p, f, "density", dens.cues, dens.units)
        if res.status == "resolved":
            self._qualifiers(p, f, store.resolver.material_index, res.ids)
            p.entities.append(EntityRef("material", ", ".join(res.labels), "resolved", res.ids))
            p.row_ids = list(res.ids)
            if (
                len(res.ids) == 1
                and lam_c is not None
                and lam_c.interval
                and lam_c.interval.is_point
            ):
                # one identified row: a stated λ is a claim to check, not a reverse lookup
                lam_c.state = "checked"
                p.operation = "existence"
        elif res.status == "none":
            if lam_c is None:
                p.entities.append(EntityRef("material", "", "none"))
                if p.explicit_reference and not p.constraints and _whole_request(f):
                    p.operation = "whole_table"
                else:
                    p.constraints.append(
                        Constraint("material", [], state="unresolved", detail="no material named")
                    )
            else:
                p.row_ids = rows  # reverse over every printed row
        else:
            reason = "ambiguous" if res.status == "ambiguous" else "unknown"
            p.entities.append(
                EntityRef(
                    "material",
                    " ".join(res.leftover),
                    reason,
                    detail="; ".join(res.notes)
                    or "no curated material label matches the named material",
                )
            )
        if (
            p.row_ids
            and len(p.row_ids) > 1
            and p.constraint("density") is None
            and p.operation != "reverse"
            and not [c for c in p.constraints if c.dimension.startswith("condition:")]
        ):
            densities = {store.app07[store._app07_pos[i]]["density"]["raw"] for i in p.row_ids}
            if len(densities) > 1:
                if not self._enumerate(p, f, schema, "density"):
                    p.missing.append("density")  # the printed values differ by density
            elif not self._enumerate(p, f, schema, "material"):
                names = sorted(
                    {store.app07[store._app07_pos[i]]["display_name"] for i in p.row_ids}
                )
                p.entities.append(
                    EntityRef(
                        "material",
                        "; ".join(names),
                        "ambiguous",
                        detail="several printed rows of the named material differ by their "
                        "printed condition; the question selects none",
                    )
                )

    def _material_identity_named(self, text: str, ids: list[str]) -> bool:
        """A resolved material is named by at least one content word of its printed label -
        never only by operator or generic words of its condition («کمتر», «با», «درصد»)."""
        idx = self.store.resolver.material_index
        q = idx.query_tokens(text)
        wanted = set(ids)
        weak = _OPERATOR_WORDS | _CONDITION_GENERIC | FUNCTION_WORDS
        return any(
            any(m.label.tokens[i] not in weak for i in m.label_idx)
            for m in idx.match(q)
            if m.label.id in wanted
        )

    def _conditions(self, p: StructuredQueryPlan, f: Features, row_ids: list[str]) -> None:
        """Stated numbers that name a printed numeric condition of the candidate rows (the
        condition's name words are read on both sides of its number, as printed)."""
        names: set[str] = set()
        for rid in row_ids:
            r = self.store.app07[self.store._app07_pos[rid]]
            names |= {n for n, _ in condition_quantities(r)}
        others = [x for x in self.cue_sets.get(p.target, []) if x[1]]
        for name in sorted(names):
            specific = [w for w in name.split() if w not in _CONDITION_GENERIC]
            for q in f.quantities:
                if not f.free(q):
                    continue
                window = _keyset(_window_words(f.norm, q))
                if not specific or not _names_all(specific, window, any_word=True):
                    continue
                if not _names_all(name.split(), window | _keyset(_CONDITION_GENERIC)):
                    continue
                # the nearest cue owns a number: «چگالی ۱۶۰۰ تا ۱۸۰۰» is a density even when a
                # printed condition name («ماسه رودخانه») follows it
                own = _cue_gap(f.norm, q, specific)
                if any(
                    (g := _cue_gap(f.norm, q, x[1])) is not None and (own is None or g[0] < own[0])
                    for x in others
                ):
                    continue
                f.consumed.append((q.start, q.end))
                c = Constraint(
                    f"condition:{name}",
                    interval=q.interval,
                    source=q.raw.strip(),
                    rule="printed_condition",
                )
                if q.ratio:
                    c.ratio = (str(q.ratio[0]), str(q.ratio[1]))
                p.constraints.append(c)
                break

    # ------------------------------------------------------------------ main tables with row labels
    def _plan_label(self, p: StructuredQueryPlan, f: Features, schema: TableSchema) -> None:
        t = self.store.main[schema.dataset]
        self._categories(p, f, schema)
        out = schema.outputs[0]
        # numbers printed in row labels (PF bands, power ranges) are interpreted first; only a
        # remaining number cued by an output is a reverse lookup
        rows, entity = self.label_rows(t, f, p)
        self._attach(p, f, out.name, out.cues, out.units, weak_cues=out.weak_cues, reverse=True)
        if entity is not None:
            p.entities.append(entity)
        if rows is not None:
            self._claims(p, f, out.name)
        if rows is None:
            if p.operation == "reverse":
                p.row_ids = list(range_ids(t))
                return
            unknown = self._unknown_word(f.entity_text, schema.dataset)
            if unknown:
                p.entities.append(
                    EntityRef(
                        "label",
                        unknown,
                        "unknown",
                        detail=f"«{unknown}» is not printed in the table's rows; no row is chosen",
                    )
                )
            elif p.explicit_reference and _whole_request(f):
                p.operation = "whole_table"
                p.row_ids = list(range_ids(t))
            return
        p.row_ids = rows

    def label_rows(  # noqa: PLR0911, PLR0912, PLR0915
        self, t: MainTable, f: Features, p: StructuredQueryPlan
    ) -> tuple[list[str] | None, EntityRef | None]:
        """Rows whose printed label is fully named, after qualifier, count and numeric-condition
        checks; ``(None, None)`` when no row word is named at all."""
        if t.labels is None or t.label_rows is None:
            return None, None
        self._current = t.table_id
        idx = t.labels
        # «(نه سقف رو به فضای آزاد)», «به جز …»: the negated *phrase* names the excluded row
        positive, phrases = negated_phrases(f.entity_text)
        q = idx.query_tokens(positive)
        matches = idx.match(q)
        excluded = [
            _keyset(w for w in etokens(ph) if w not in idx.ignore and w not in FUNCTION_WORDS)
            for ph in phrases
        ]
        excluded = [x for x in excluded if x]
        for ph in phrases:
            p.qualifiers.append(Qualifier(ph, "negative", "applied"))
        content = [
            m
            for m in matches
            if m.label_idx and not any(x <= _keyset(m.label.tokens) for x in excluded)
        ]
        if not content:
            return None, None
        # 1. a row contradicted by the query (a sibling row's distinguishing word, a different
        #    count, a negated word) is never a candidate
        consistent: list[Match] = []
        contradicted: list[tuple[Match, list[str]]] = []
        for m in content:
            bad = self._label_contradictions(idx, q, m, content)
            if bad:
                contradicted.append((m, bad))
            else:
                consistent.append(m)
        if not consistent:
            seen = {q.word for q in p.qualifiers}
            for m, bad in contradicted:
                # dominated rows are not qualifier contradictions; each qualifier once
                for word in (b for b in bad if not b.startswith("named more fully")):
                    if word in seen:
                        continue
                    seen.add(word)
                    p.qualifiers.append(
                        Qualifier(
                            word,
                            "positive",
                            "contradicted",
                            f"not printed in «{' '.join(m.label.tokens)}»",
                        )
                    )
            return [], None
        # 2. the rows that explain the most query words; 3. one row, or one fully named row, or
        #    rows that differ only in printed numbers (narrowed by stated numbers, else
        #    enumerated); anything else is ambiguous - no row is chosen
        score = {m.label.id: self._explained(t, f, q, m) for m in consistent}
        top = max(score.values())
        tied = [m for m in consistent if score[m.label.id] == top]
        kept = tied
        numeric_tie = len({m.label.id for m in tied}) > 1 and (
            _numeric_variants(tied) or self._same_identity(t, tied)
        )
        if numeric_tie:
            # rows that differ only in printed numbers (PF bands, power bands): the stated
            # numbers decide through the numeric conditions (checked below)
            pass
        elif len({m.label.id for m in tied}) > 1:
            # a unique row whose every level, else its most specific level, else its entity
            # level is fully named («دیگ بخار» vs «دیگ بخار ری‌هیت‌دار», «پایش» vs «زیرپایش»)
            for named in (
                [m for m in tied if _fully(m) and min(m.specificity) >= 1.0],
                [m for m in tied if _fully(m)],
                [m for m in tied if _top_level_named(m)],
                [m for m in tied if _head_named(m)],
            ):
                if len({m.label.id for m in named}) == 1:
                    kept = named
                    break
            if kept is tied:
                names = sorted({" ".join(m.label.tokens) for m in tied})
                return [], EntityRef(
                    "label",
                    "; ".join(names),
                    "ambiguous",
                    detail="several printed rows fit the named words equally ("
                    + "; ".join(names)
                    + "); no row is chosen",
                )
        # source-exact identity: a kept row must be named exactly, never by a shared word
        exact_rows, head_rows, _ = self.identify(t.table_id, positive, f.norm)
        exact_set = set(exact_rows)
        label_rows = t.label_rows

        def named_exactly(m: Match) -> bool:
            return any(t.records[i]["record_id"] in exact_set for i in label_rows[m.label.id])

        named_kept = [m for m in kept if named_exactly(m)]
        if not named_kept:
            alt = [m for m in consistent if named_exactly(m)]
            if len({m.label.id for m in alt}) == 1:
                named_kept = alt
            elif alt or head_rows:
                names = sorted({" ".join(m.label.tokens) for m in (alt or kept)})
                return [], EntityRef(
                    "label",
                    "; ".join(names),
                    "ambiguous",
                    detail="several printed rows fit the named words and none is named exactly",
                )
            else:
                return None, None  # no printed row is named: shared words are not an identity
        kept = named_kept
        # printed numeric conditions inside the kept labels («بیش از ۷۰ کیلووات», «کمتر از %۲۰»)
        positions = sorted({i for m in kept for i in t.label_rows[m.label.id]})
        rows = [t.records[i] for i in positions]
        before = len(p.constraints)
        rows = self._label_numbers(p, f, t, rows)
        if (
            numeric_tie
            and len(p.constraints) == before
            and len({" ".join(_label_texts(t, r)) for r in rows}) > 1
            and not self._enumerate(p, f, self.schemas[t.table_id], "row_condition")
        ):
            p.missing.append("row_condition")  # the printed values differ by the band
        # a bare number printed in the chosen row's own label («۴ لوله») is part of its name
        printed = {w for m in kept for w in m.label.tokens if w.replace(".", "").isdigit()}
        for qty in f.quantities:
            if (
                f.free(qty)
                and qty.unit is None
                and qty.interval.is_point
                and qty.raw.strip() in f.entity_text
                and str(qty.numbers[0]) in printed
            ):
                f.consumed.append((qty.start, qty.end))
        # an unknown word glued to the matched name names something not printed («وان جکوزی»)
        used = set().union(*(m.query_idx for m in kept))
        vocab = self.store.resolver.vocabulary
        glued = [
            q[i]
            for i in range(len(q))
            if i not in used
            and vocab
            and ({i - 1, i + 1} & used)
            and q[i] not in idx.ignore
            and q[i] not in _COLLOQUIAL
            and _negated(q[i]) is None
            and not keys(q[i]) & _STRUCTURAL_WORDS
            and not keys(q[i].removesuffix("ه")) & (_UNIT_WORDS | _STRUCTURAL_WORDS)
            and not q[i].replace(".", "").isdigit()
            and not keys(q[i]) & self.vocab_keys()
            and not keys(q[i]) & idx.key_vocabulary
        ]
        if glued:
            return [], EntityRef(
                "label",
                " ".join(glued),
                "unknown",
                detail=f"«{' '.join(glued)}» is not printed in the table's rows",
            )
        for w in q:
            if (n := _negated(w)) is not None:
                p.qualifiers.append(Qualifier(n, "negative", "applied"))
        p.entities.append(
            EntityRef("label", "; ".join(" ".join(m.label.tokens) for m in kept), "resolved")
        )
        return [r["record_id"] for r in rows], None

    def _same_identity(self, t: MainTable, matches: list[Match]) -> bool:
        """Rows whose source-derived identity words are equal (numbers, operators, units and
        printed exceptions removed): they differ only in their printed numeric conditions
        («با توان بیش از ۱۲۰ وات و کمتر از ۱۰۰۰ کیلووات (به غیر از …)» / «بین ۷۵ تا ۲۰۰»)."""
        idx = self.identity.get(t.table_id)
        if idx is None or t.label_rows is None:
            return False
        shapes = set()
        for m in matches:
            rows = t.label_rows[m.label.id]
            if not rows:
                return False
            rid = t.records[rows[0]]["record_id"]
            shapes.add(frozenset(tuple(a.words) for a in idx.by_row.get(rid, [])))
        return len(shapes) == 1

    def _explained(self, t: MainTable, f: Features, q: list[str], m: Match) -> int:
        """Query words a row explains, plus the row's printed numeric condition when the
        question states a number in its unit («PF ۲۰ درصد») or explicitly asks for that band's
        alternatives («PF های مختلف») - not a skylight row without the band."""
        stated = {
            x.unit for x in f.quantities if f.free(x) and x.unit and x.raw.strip() in f.entity_text
        }
        assert t.label_rows is not None
        first = t.records[t.label_rows[m.label.id][0]]
        printed = {x.unit for x in label_quantities(t, first) if x.unit}
        bands = self.schemas[t.table_id].dim("row_condition")
        asked = (
            bool(printed)
            and bands is not None
            and enumeration_request(f.norm, bands.nouns) is not None
        )
        return len(_word_idx(q, m)) + len(printed & stated) + int(asked)

    def _label_contradictions(
        self, idx: LabelIndex, q: list[str], m: Match, content: list[Match]
    ) -> list[str]:
        """Why a row cannot be the one asked about: another row explains a strict superset of
        the query words it explains (the query names more of the other row - «بویلر چگالشی»
        excludes «بویلر غیرچگالشی»), or a count qualifier («تک/دو/چند …») differs from the
        row's. Negated words are excluded earlier by the label index."""
        siblings = [x for x in content if x.label.id != m.label.id]
        bad = [
            "named more fully by «" + " ".join(x.label.tokens) + "»"
            for x in siblings
            if _word_idx(q, m) < _word_idx(q, x)
        ][:1]
        qc, lc = _counts(q), _counts(m.label.tokens)
        bad += [f"{c} {h}" for h, c in qc.items() if h in lc and lc[h] != c]
        bad += [
            f"{c} {h}"
            for h, c in qc.items()
            if h not in lc and any(h in _counts(x.label.tokens) for x in siblings)
        ]
        return bad

    def _label_numbers(
        self, p: StructuredQueryPlan, f: Features, t: MainTable, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter rows by numbers printed in their labels when the query states a number with
        the same unit; such numbers are consumed (interpreted)."""
        conds = {r["record_id"]: label_quantities(t, r) for r in rows}
        units = {q.unit for v in conds.values() for q in v if q.unit}
        for qty in f.quantities:
            if not f.free(qty) or not qty.unit or qty.unit not in units:
                continue
            f.consumed.append((qty.start, qty.end))
            c = Constraint(
                f"label:{qty.unit}",
                interval=qty.interval,
                source=qty.raw.strip(),
                rule="label_condition",
            )
            p.constraints.append(c)
            # a row printing no condition in that unit is not constrained by the number
            rows = [r for r in rows if label_condition_ok(conds[r["record_id"]], qty)]
        return rows

    # ------------------------------------------------------------------ 19-3-1
    def _plan_criterion(self, p: StructuredQueryPlan, f: Features, schema: TableSchema) -> None:
        self._categories(p, f, schema)
        for var, cues in (
            ("HDD18°C", ("hdd", "روز درجه گرمایش", "درجه روز گرمایش", "روز-درجه گرمایش")),
            ("CDD10°C", ("cdd", "روز درجه سرمایش", "درجه روز سرمایش", "روز-درجه سرمایش")),
        ):
            c = self._attach(p, f, f"degree_days:{var}", cues, (), reverse=True)
            if c is not None:
                cc = p.constraint("climate_class")
                if cc is not None:
                    cc.state = "checked"
        if p.operation != "reverse":
            dd = schema.dim("degree_days")
            if dd is not None and self._attach(p, f, "degree_days", dd.cues, ()):
                p.constraints[-1].state = "unresolved"
                p.constraints[-1].detail = "heating or cooling degree days not stated"
        if (
            p.operation == "direct"
            and not p.constraint("climate_class")
            and "climate_class" not in p.enumerated
        ):
            whole = p.explicit_reference and _whole_request(f)
            p.operation = "whole_table" if whole else "direct"
            if not whole:
                p.constraints.append(
                    Constraint(
                        "climate_class",
                        [],
                        state="unresolved",
                        detail="no climate class or degree-day value stated",
                    )
                )

    # ------------------------------------------------------------------ 19-3-3
    def _plan_rating_range(self, p: StructuredQueryPlan, f: Features, schema: TableSchema) -> None:
        self._categories(p, f, schema)
        d = schema.dim("r_ratio")
        assert d is not None
        c = self._attach(p, f, "r_ratio", d.cues, (), weak_cues=d.weak_cues, reverse=True)
        if c is None and not p.constraint("rating") and "rating" not in p.enumerated:
            if p.explicit_reference and _whole_request(f):
                p.operation = "whole_table"
            else:
                p.constraints.append(
                    Constraint(
                        "rating", [], state="unresolved", detail="no rating or R ratio stated"
                    )
                )

    # ------------------------------------------------------------------ qualifiers (entities)
    def _qualifiers(
        self, p: StructuredQueryPlan, f: Features, idx: LabelIndex, ids: list[str]
    ) -> None:
        """Count qualifiers of the query must agree with the resolved entity's label."""
        q = idx.query_tokens(f.entity_text)
        labels = [lab for lab in idx.labels if lab.id in set(ids)]
        qc = _counts(q)
        for lab in labels:
            lc = _counts(lab.tokens)
            for h, c in qc.items():
                if h in lc and lc[h] != c:
                    p.qualifiers.append(
                        Qualifier(
                            f"{c} {h}",
                            "positive",
                            "contradicted",
                            f"«{' '.join(lab.tokens)}» prints {lc[h]} {h}",
                        )
                    )


# ------------------------------------------------------------------ helpers


_WHOLE_WORDS = frozenset(
    {
        "همه",
        "تمام",
        "تمامی",
        "کامل",
        "لیست",
        "فهرست",
        "مقادیر",
        "مقادیری",
        "اعداد",
        "ردیفها",
        "ردیفهای",
        "ستونها",
        "ستونهای",
        "محتوا",
        "محتوای",
        "شامل",
        "پوشش",
        "نشان",
        "چیزی",
        "کدام",
        "کدوم",
        "کدامها",
    }
)
# question/predicate words that never name an entity
_STRUCTURAL_WORDS = frozenset(
    k
    for w in (
        "چقدر",
        "چقدره",
        "چنده",
        "چند",
        "چیست",
        "چیه",
        "است",
        "هست",
        "باشد",
        "باشه",
        "بشه",
        "داره",
        "دارد",
        "دارن",
        "دارند",
        "شود",
        "میشه",
        "می",
        "شده",
        "گرفته",
        "تعریف",
        "کرده",
        "حد",
        "مقدار",
        "میزان",
        "علامت",
        "علامتی",
        "وضعیت",
        "وضعیتی",
        "تیک",
        "ضربدر",
        "سال",
        "گروه",
        "برای",
        "طبق",
        "جدول",
        "پیوست",
        "مبحث",
        "نوزدهم",
        "ابلاغ",
        "بگو",
        "بده",
        "میخوام",
        "بدونم",
        "لطفا",
        "دقیقا",
        "حدودا",
        "یعنی",
        "چطور",
        "آیا",
        "ایا",
        "اگه",
        "اگر",
        "رو",
        "را",
        "تو",
        "توی",
        "واسه",
        "باید",
        "لازمه",
        "لازم",
        "ستون",
        "ردیف",
        "بخش",
        "بند",
        "حداقل",
        "حداکثر",
        "مجاز",
        "اقلیم",
        "اقلیمی",
        "رده",
        "انرژی",
        "ساختمان",
        "ساختمون",
        "چنده",
        "چقدره",
        "چیه",
        "کدومه",
        "کدامه",
        "چطوره",
        "درسته",
        "صحیحه",
        "دیگه",
    )
    for k in keys(w)
)


def _whole_request(f: Features) -> bool:
    """The question asks for a table as a whole («همه/کامل/فهرست/چه مقادیری …»)."""
    return any(w in _WHOLE_WORDS for w in etokens(f.norm))


_ENUMERATION_GENERIC = re.compile(
    r"به\s?تفکیک|انواع|هر\s?کدام|هر\s?یک|همه\s?حالت|حالت\s?های\s?مختلف|چه\s?مقادیری"
    r"|همه\s?مقادیر|تمام\s?مقادیر|فهرست|لیست"
)
_ALL = r"(?:همه\s?ی|همه|تمامی|تمام|کلیه|هر)"


def enumeration_request(norm: str, nouns: Sequence[str]) -> str | None:
    """The words that explicitly ask for the printed alternatives of a dimension («همه
    اقلیم‌ها», «اقلیم‌های مختلف», «کدام رده‌ها», «به تفکیک», «انواع»…); None otherwise. A dimension
    that is merely absent is never a request."""
    m = _ENUMERATION_GENERIC.search(norm)
    if m:
        return m[0]
    for noun in nouns:
        n = re.escape(normalize_for_search(noun))
        for pattern in (
            rf"{_ALL}\s*{n}",
            rf"{n}\s*(?:ها|های|هایی)?\s*(?:ی\s+)?(?:مختلف|گوناگون)",
            rf"(?:کدام|کدوم|چه)\s*{n}\s*(?:ها|های|هایی)",
        ):
            m = re.search(pattern, norm)
            if m:
                return m[0]
    return None


_COORDINATION = re.compile(r"\s*(?:و|ولی|اما|،|,)?\s*")


def _one_sided(iv: NumInterval | None) -> bool:
    return iv is not None and (iv.lo is None) != (iv.hi is None)


def _coordinated_bound(f: Features, q: Quantity, c: Constraint) -> None:
    """«بالای ۱۵۰ ولی زیر ۱۷۵», «بیشتر از ۳۰۰۰ و حداکثر ۴۰۰۰»: a one-sided bound followed only by
    a coordinator («و», «ولی», «اما») and the complementary one-sided bound states one interval
    of the same dimension (each bound keeps its printed inclusivity). Bounds on the same side,
    a unit change or any other word in between are not coordinated; contradictory bounds make
    the constraint invalid."""
    nxt = next((x for x in f.quantities if x.start >= q.end and f.free(x)), None)
    if nxt is None or not _COORDINATION.fullmatch(f.norm[q.end : nxt.start]):
        return
    a, b = q.interval, nxt.interval
    if not (_one_sided(a) and _one_sided(b)) or (a.lo is None) == (b.lo is None):
        return
    if q.unit and nxt.unit and q.unit != nxt.unit:
        return
    lo, hi = a.lo or b.lo, a.hi or b.hi
    assert lo is not None
    assert hi is not None
    f.consumed.append((nxt.start, nxt.end))
    c.source = f.norm[q.start : nxt.end].strip()
    c.rule = "coordinated_bound"
    if lo.value > hi.value or (lo.value == hi.value and OPEN in (lo.kind, hi.kind)):
        c.state = "invalid"
        c.detail = f"contradictory_bounds: {c.source}"
        c.reason = "contradictory_bounds"
    else:
        c.interval = NumInterval(lo, hi)


def _coordinated_with_consumed(f: Features, q: Quantity) -> bool:
    """A comparison («بالای ۱۷۵») joined only by a coordinator to a number this plan read
    («بالای ۱۵۰ و بالای ۱۷۵») constrains the same thing, even when a decomposition put it in
    another clause: left uninterpreted it is unresolved, never dropped."""
    if q.raw.strip().replace(".", "").isdigit():
        return False
    return any(
        e <= q.start and _COORDINATION.fullmatch(f.norm[e : q.start]) for _s, e in f.consumed
    )


def _reason(detail: str) -> str:
    """The specific reason code at the head of an invalid constraint's detail."""
    head = detail.split(":", 1)[0]
    return head if head.endswith("_not_in_table") else ""


def range_ids(t: MainTable) -> list[str]:
    return [r["record_id"] for r in t.records]


def label_condition_ok(printed: list[Quantity], q: Quantity) -> bool:
    """A stated number against the numeric conditions printed in a row label (same unit): it
    restates one of them exactly or lies within it, and lies in no printed exclusion; a row
    printing no condition in that unit is not constrained."""
    same = [x for x in printed if x.unit == q.unit]
    if not same:
        return True
    plain = [x for x in same if not x.negated]
    if plain and not any(
        _same_interval(x.interval, q.interval)
        or x.interval.relation(q.interval) in ("within", "boundary")
        for x in plain
    ):
        return False
    return not any(
        x.negated and x.interval.relation(q.interval) in ("within", "boundary") for x in same
    )


def _same_interval(a: NumInterval, b: NumInterval) -> bool:
    """The same printed condition restated (bound values and openness; an unspecified printed
    end matches any stated end)."""
    for x, y in ((a.lo, b.lo), (a.hi, b.hi)):
        if (x is None) != (y is None):
            return False
        if x is not None and y is not None:
            if x.value != y.value:
                return False
            if "unspecified" not in (x.kind, y.kind) and x.kind != y.kind:
                return False
    return True


_UNIT_WORDS = frozenset(
    k
    for w in ("وات", "کیلووات", "درصد", "متر", "مربع", "مکعب", "کیلوگرم", "درجه", "سانتیگراد")
    for k in keys(w)
)


_NUMERIC_WORDS = frozenset(
    {"کمتر", "بیشتر", "بیش", "از", "بین", "تا", "و", "کوچک", "بزرگ", "تر", "حداکثر", "حداقل"}
)


def _numeric_variants(matches: list[Match]) -> bool:
    """Rows whose labels differ only in printed numbers/comparison words (PF bands, power
    bands): their words outside numbers are identical."""
    shapes = {
        tuple(
            w
            for w in m.label.tokens
            if not w.replace(".", "").isdigit() and w not in _NUMERIC_WORDS and w not in "≥≤<>"
        )
        for m in matches
    }
    return len(shapes) == 1


# colloquial function words never name an entity («وقتی», «همون», «دیگه»)
_COLLOQUIAL = frozenset(
    {
        "وقتی",
        "زمانی",
        "همون",
        "همان",
        "همین",
        "دیگه",
        "دیگر",
        "اینکه",
        "چون",
        "اگه",
        "اگر",
        "اون",
        "این",
        "باشه",
        "باشد",
        "هستش",
        "ست",
        "ه",
        "یه",
        "یک",
        "تو",
        "توی",
        "واسه",
        "برای",
        "مثلا",
        "مثلاً",
        "دقیقا",
        "حدودا",
        "تقریبا",
        "میشه",
        "می",
        "بشه",
        "کنه",
        "داره",
    }
)
_CONDITION_GENERIC = frozenset({"چگالی", "درصد", "با", "بدون", "کمتر", "بیش", "بین", "حداکثر"})
_OPERATOR_WORDS = frozenset({"بین", "بیش", "بیشتر", "کمتر", "حداکثر", "حداقل", "یا", "مساوی", "تا"})


_PRINTED_CACHE: dict[tuple[str, str], Any] = {}
_LAMBDA = re.compile(r"λ|لاندا|لامبدا|ضریب\s?هدایت|هدایت\s?حرارتی")


def condition_quantities(r: dict[str, Any]) -> list[tuple[str, Quantity]]:
    """(name, quantity) for each number printed in an Appendix 7 row's condition. The name is
    the content words before the number, or - when none - the words after it up to the next
    clause («با کمتر از ۱۰ درصد ماسهٔ رودخانه» -> «درصد ماسه رودخانه»). Memoised per row
    (curated rows never change)."""
    key = ("condition", str(r.get("id") or r.get("record_id")))
    if key not in _PRINTED_CACHE:
        _PRINTED_CACHE[key] = _condition_quantities(r)
    result: list[tuple[str, Quantity]] = _PRINTED_CACHE[key]
    return result


def label_quantities(t: MainTable, r: dict[str, Any]) -> list[Quantity]:
    """Numbers printed in a main-table row's label (memoised per row)."""
    key = ("label", str(r["record_id"]))
    if key not in _PRINTED_CACHE:
        text = normalize_for_search(" ".join(_label_texts(t, r)))
        _PRINTED_CACHE[key] = find_quantities(text, printed=True)
    result: list[Quantity] = _PRINTED_CACHE[key]
    return result


def _condition_quantities(r: dict[str, Any]) -> list[tuple[str, Quantity]]:
    text = normalize_for_search(((r.get("condition") or {}).get("text")) or "")
    out = []
    for q in find_quantities(text, printed=True):
        head = re.split(r"[؛;،,]|\sو\s", text[: q.start])[-1]
        before = [
            w for w in etokens(head)[-3:] if w not in FUNCTION_WORDS and w not in _OPERATOR_WORDS
        ]
        if not before:
            tail = re.split(r"[؛;،,]|\sو\s", text[q.end :])[0]
            before = [w for w in etokens(tail)[:3] if w not in FUNCTION_WORDS]
        out.append((" ".join(before), q))
    return out


def _window_words(norm: str, q: Quantity) -> list[str]:
    """Content words within three tokens on each side of a stated number."""
    return (
        etokens(norm[max(0, q.start - 40) : q.start])[-3:] + etokens(norm[q.end : q.end + 40])[:3]
    )


def _word_idx(q: Sequence[str], m: Match) -> set[int]:
    """Query positions a label explains with words; numbers printed in labels are interpreted
    by the numeric conditions only («کمتر از %۲۰» is not named by «۲۰»)."""
    return {i for i in m.query_idx if not q[i].replace(".", "").isdigit()}


def _head_named(m: Match) -> bool:
    """The label's head noun (its first printed word: «کف», «سقف», «دیوار») is named."""
    return 0 in m.label_idx


def _top_level_named(m: Match) -> bool:
    """Every word of the label's most general (entity) level is named."""
    levels = sorted(set(m.label.levels))
    if len(levels) < 2:
        return False
    named = {m.label.tokens[i] for i in m.label_idx}
    top = {w for w, lv in zip(m.label.tokens, m.label.levels, strict=True) if lv == levels[0]}
    return bool(top) and top <= named


def _fully(m: Match) -> bool:
    return bool(m.specificity) and m.specificity[0] >= 1.0 and min(m.specificity) > 0


def _label_texts(t: MainTable, r: dict[str, Any]) -> list[str]:
    from mabhas19.retrieval.structured import LABEL_FIELDS, _field  # noqa: PLC0415

    fields = LABEL_FIELDS.get(t.table_id.removeprefix(MAIN), ())
    return [x for path, _ in fields if (x := _field(r, path))]


def negated_phrases(text: str) -> tuple[str, list[str]]:
    """(text without excluded phrases, the excluded phrases), from the typed contrast parser
    (:mod:`mabhas19.retrieval.polarity`): «نه X», «به جز X», «(نه X)», «X، نه Y»…"""
    pol = polarity.parse(text)
    return pol.positive, [e.text for e in pol.exclusions]


def _negate(tokens: list[str], vocabulary: set[str]) -> list[str]:
    """«نه X», «به جز X», «منهای X» negate X like «غیر X» when X is a printed label word."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        two = f"{t} {tokens[i + 1]}" if i + 1 < len(tokens) else ""
        if two in ("به جز", "به غیر") and i + 2 < len(tokens):
            nxt = tokens[i + 3] if tokens[i + 2] == "از" and i + 3 < len(tokens) else tokens[i + 2]
            skip = 4 if tokens[i + 2] == "از" else 3
            if keys(nxt) & _keyset(vocabulary):
                out.append("غیر" + nxt)
                i += skip
                continue
        if (
            t in ("نه", "بجز", "جز", "منهای")
            and i + 1 < len(tokens)
            and keys(tokens[i + 1]) & _keyset(vocabulary)
        ):
            out.append("غیر" + tokens[i + 1])
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def _negated(token: str) -> str | None:
    for n in NEGATORS:
        if token.startswith(n) and len(token) > len(n) + 1:
            return token[len(n) :]
    return None


def _counts(tokens: Sequence[str]) -> dict[str, str]:
    """Head word -> count qualifier («تک فاز» -> {فاز: 1}, «دو اثره» -> {اثره: 2})."""
    out: dict[str, str] = {}
    for i, t in enumerate(tokens):
        if t in _COUNT and i + 1 < len(tokens) and tokens[i + 1] not in FUNCTION_WORDS:
            out[tokens[i + 1]] = _COUNT[t]
            continue
        for c in ("تک", "چند"):
            if t.startswith(c) and len(t) > len(c) + 2:
                out[t[len(c) :]] = _COUNT[c]
    return out


__all__ = ["Decimal", "Features", "Label", "Planner", "features"]
