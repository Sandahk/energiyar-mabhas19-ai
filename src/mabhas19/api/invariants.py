"""Deterministic robustness invariants (audit and tests only; the runtime does not call this).

Two kinds of check, both built from the curated store and the deterministic quantity reader
(:mod:`mabhas19.api.quantities`), never from hand-written examples:

* **Namespace collisions** (:func:`collisions`): a normalized word that names things in more than
  one namespace (listed cities, Appendix 7 materials, printed table rows, Appendix 3 occupancies,
  printed output names, family words), exactly or as a strict prefix of another word («تبریز» ->
  «تبریزی»). The reviewed list is ``eval/robustness/collisions_baseline.yaml``.
* **Answer invariants**: quantity preservation (:func:`quantity_violations`: once one table
  quantity is identified, what the answer shows comes from that quantity's tables unless a
  reviewed exception of ``eval/robustness/exceptions.yaml`` applies) and no unwritten word
  (:func:`unwritten_entity_violations`: a resolved entity the answer shows must rest on a word
  the user wrote).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import yaml

from mabhas19.api import quantities as Q
from mabhas19.api import wording as W
from mabhas19.api.cities import catalogue
from mabhas19.core.paths import EVAL_DIR
from mabhas19.generation import labels as L
from mabhas19.generation import options as O
from mabhas19.retrieval.schema import APP02, APP03, APP07, MAIN

if TYPE_CHECKING:
    from mabhas19.generation.contract import Answer
    from mabhas19.retrieval.models import RetrievalResult
    from mabhas19.retrieval.structured import StructuredStore

ROBUSTNESS_DIR = EVAL_DIR / "robustness"
T19_3_2 = f"{MAIN}19-3-2"
T19_2_1 = f"{MAIN}19-2-1"
CONTEXT_TABLES = frozenset({APP02, T19_3_2})  # X3: city rows shown as supporting context only
MIN_WORD = 3  # shorter keys are particles, never names
MIN_PREFIX = 4  # a prefix collision needs at least this many letters
# function words and the generic words every row label repeats (never a distinguishing name)
STOPWORDS = frozenset(
    {
        "و",
        "یا",
        "در",
        "از",
        "با",
        "به",
        "برای",
        "که",
        "تا",
        "بر",
        "های",
        "ها",
        "این",
        "آن",
        "مجاور",
        "فضای",
        "حداکثر",
        "حداقل",
        "حد",
        "اکثر",
        "بین",
        "کمتر",
        "بیشتر",
        "نشده",
        "کنترل",
        "نیمه",
        "باز",
    }
)
_WORD = re.compile(r"[^\W\d_]+")
# X6: printed symbols of quantities the deterministic reader does not count on their own
SYMBOLS = {
    "u_value": re.compile(r"(?<![A-Za-z])U(?![A-Za-z])"),
    "shgc": re.compile(r"SHGC", re.IGNORECASE),
    "lambda": re.compile(r"λ"),
    "r_value": re.compile(r"(?<![A-Za-z])R(?![A-Za-z])"),
}


def word_keys(text: str) -> list[str]:
    """The comparison keys of the name words of ``text`` (stopwords and particles dropped)."""
    stop = {O.label_key(w) for w in STOPWORDS}
    out = []
    for w in _WORD.findall(W.letters(text).replace("‌", " ")):
        k = O.label_key(w)
        if len(k) >= MIN_WORD and k not in stop:
            out.append(k)
    return out


# ---------------------------------------------------------------- namespaces and collisions


def namespaces(store: StructuredStore) -> dict[str, dict[str, set[str]]]:
    """namespace -> word key -> printed labels containing it. Table 19-3-2 rows are the listed
    cities themselves (same entity), so they are not a separate namespace."""
    ns: dict[str, dict[str, set[str]]] = {}

    def add(name: str, label: str) -> None:
        for k in word_keys(label):
            ns.setdefault(name, {}).setdefault(k, set()).add(label)

    for e in catalogue(store):
        add("city", e.name)
    for r in store.app07:
        add("material", str(r.get("display_name") or ""))
    for dataset in sorted(store.main):
        if dataset == T19_3_2:
            continue
        for field in O.LABEL_KINDS:
            for label in O.printed_options(store, dataset, field):
                add(f"row:{dataset.rsplit('.', 1)[-1]}", label)
    for label in O.printed_options(store, APP03, "occupancy"):
        add("occupancy", label)
    for label in L.OUTPUT_LABELS.values():
        add("quantity", label)
    for fam in W.FAMILIES:
        add("family", fam.shown)
    return ns


def group(namespace: str) -> str:
    return "row" if namespace.startswith("row:") else namespace


@dataclass(frozen=True)
class Collision:
    """A word key in more than one namespace (``exact``), or a key that is a strict prefix of a
    key of another namespace (``prefix``: ``word`` -> ``longer``)."""

    kind: str
    word: str
    longer: str
    namespaces: tuple[str, ...]  # of ``word``
    longer_namespaces: tuple[str, ...]  # of ``longer`` (prefix only)
    examples: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.word}" + (f">{self.longer}" if self.longer else "")


def collisions(store: StructuredStore) -> list[Collision]:
    """Every cross-namespace collision of the curated catalogue, in a stable order."""
    ns = namespaces(store)
    index: dict[str, set[str]] = {}
    for name, words in ns.items():
        for k in words:
            index.setdefault(k, set()).add(name)

    def examples(k: str, names: set[str]) -> tuple[str, ...]:
        return tuple(sorted({sorted(ns[n][k])[0] for n in names}))

    out = [
        Collision("exact", k, "", tuple(sorted(n)), (), examples(k, n))
        for k, n in sorted(index.items())
        if len({group(x) for x in n}) > 1
    ]
    keys = sorted(index)
    for a in keys:
        if len(a) < MIN_PREFIX:
            continue
        for b in keys:
            if b == a or not b.startswith(a):
                continue
            ga, gb = {group(x) for x in index[a]}, {group(x) for x in index[b]}
            if ga - gb or gb - ga:  # the longer word belongs to a namespace the shorter does not
                out.append(
                    Collision(
                        "prefix",
                        a,
                        b,
                        tuple(sorted(index[a])),
                        tuple(sorted(index[b])),
                        examples(a, index[a]) + examples(b, index[b]),
                    )
                )
    return out


# ---------------------------------------------------------------- reviewed definitions


@cache
def exceptions() -> dict[str, Any]:
    data = yaml.safe_load((ROBUSTNESS_DIR / "exceptions.yaml").read_text(encoding="utf-8"))
    return dict(data)


@cache
def baseline() -> dict[str, Any]:
    data = yaml.safe_load((ROBUSTNESS_DIR / "collisions_baseline.yaml").read_text(encoding="utf-8"))
    return dict(data)


# ---------------------------------------------------------------- answer invariants


def own_tables(quantity: Q.Quantity) -> set[str]:
    return {d for d, _ in quantity.tables}


def shown_tables(answer: Answer) -> set[str]:
    """The tables whose rows or questions the answer shows (X7: unused plans are not shown)."""
    out = {v.dataset for v in answer.values}
    out |= {c.dataset for c in answer.clarifications if c.dataset}
    out |= {r.dataset for r in answer.refusals if r.dataset}
    return out


def _explicit_reference(result: RetrievalResult | None) -> bool:
    signals = result.route.signals if result is not None else {}
    return bool(signals.get("table_ref") or signals.get("section_ref"))


def _material_written(question: str, answer: Answer) -> bool:
    """X1: an Appendix 7 label word the answer names is written in the question."""
    written = set(word_keys(question))
    names = [i.value for i in answer.interpretation if i.key == "material"]
    names += [v.row_label for v in answer.values if v.dataset == APP07]
    return any(set(word_keys(n)) & written for n in names)


def _second_quantity_written(question: str, dataset: str, asked: Q.Quantity) -> bool:
    """X6: another quantity whose tables include ``dataset`` is written by its symbol."""
    text = W.letters(question)
    for q in Q.QUANTITIES:
        if q.id == asked.id or dataset not in own_tables(q):
            continue
        symbol = SYMBOLS.get(q.id)
        if q.pattern().search(text) or (symbol is not None and symbol.search(text)):
            return True
    return False


def quantity_violations(
    question: str, answer: Answer, result: RetrievalResult | None = None
) -> list[str]:
    """Quantity preservation: what ``answer`` shows for a question with exactly one table
    quantity comes from that quantity's tables, or a reviewed exception (X1-X8) applies."""
    request = Q.analyse(question)
    if request is None or answer.response_type == "prose":  # X8
        return []
    if _explicit_reference(result):  # X4
        return []
    asked = request.quantity
    own = own_tables(asked)
    problems = [
        f"{asked.id}: context row from {v.dataset} (only city rows are supporting context)"
        for v in answer.context
        if v.dataset not in CONTEXT_TABLES
    ]
    shown = shown_tables(answer)
    if answer.values and all(v.dataset in own for v in answer.values):
        # X7: other tables' refusals beside the asked quantity's own value are secondary
        shown = {v.dataset for v in answer.values} | {c.dataset for c in answer.clarifications}
    for dataset in sorted(shown - own):
        if dataset == APP07 and asked.id == "r_value":
            if answer.response_type == "clarification" and _material_written(question, answer):
                continue  # X1
        elif dataset == T19_2_1:
            continue  # X5
        elif _second_quantity_written(question, dataset, asked):
            continue  # X6
        problems.append(f"{asked.id}: answer shows {dataset}")
    return problems


# the namespace an answer's interpretation item belongs to
ITEM_NAMESPACE = {"material": "material", "occupancy": "occupancy"}


def unwritten_entity_violations(
    question: str, answer: Answer, store: StructuredStore | None = None
) -> list[str]:
    """No unwritten word: an entity the answer shows as resolved (a material, an occupancy...)
    has at least one of its label words written, or none of its words merely extends a written
    word that is itself a name in another namespace («تبریز», a listed city, read as the material
    «تبریزی»). A plain derived form of a word that names nothing else («ویلا» -> the occupancy
    «ویلایی») is reviewed exception X9."""
    names = namespaces(store) if store is not None else {}
    written = set(word_keys(question))
    problems = []
    for item in answer.interpretation:
        if item.key in ("output", "climate_class", "place") or item.status != "resolved":
            continue
        words = word_keys(item.value)
        if not words or set(words) & written:
            continue
        own = ITEM_NAMESPACE.get(item.key, "row")

        def elsewhere(x: str, own: str = own) -> bool:  # a name of another namespace
            return not names or any(x in keys for n, keys in names.items() if group(n) != own)

        extended = [
            w
            for w in words
            if any(
                w != x and len(x) >= MIN_PREFIX and w.startswith(x) and elsewhere(x)
                for x in written
            )
        ]
        if extended:
            problems.append(f"{item.key}: «{item.value}» rests on unwritten {extended}")
    return problems


def substitution_violations(written_label: str, answer: Answer) -> list[str]:
    """No nearest-concept substitution: a value shown for a question naming ``written_label``
    belongs to a row whose printed label contains every name word written."""
    want = set(word_keys(written_label))
    return [
        f"value of «{v.row_label}» for «{written_label}»"
        for v in answer.values
        if not want <= set(word_keys(v.row_label))
    ]


# ---------------------------------------------------------------- generated sweeps

CITY_TEMPLATES = (
    "برای سقف یه ساختمان تو {city} حداقل مقاومت حرارتی چقدره؟",
    "حداقل مقاومت حرارتی {city} چقدره؟",
    "حداقل R دیوار در {city} چقدره؟",
    "حداکثر ضریب انتقال حرارتی پنجره در {city} چقدره؟",
    "حداکثر SHGC پنجره در {city} چقدره؟",
    "حد مجاز مصرف انرژی در {city} چقدره؟",
)
FOREIGN = "زرشکی"  # a word of no namespace (checked by the tests)
NEAR_MISS_TABLES = ((f"{MAIN}19-5-1", "element"), (f"{MAIN}19-5-5", "place_of_use"))
NEAR_MISS_MATERIALS = 25


def sweep_questions(store: StructuredStore) -> list[dict[str, str]]:
    """Single questions generated from the catalogue: every listed city x the climate-dependent
    quantity templates, every family word x every quantity, and near-miss row and material names
    (a foreign word added; the last word dropped)."""
    out = [
        {"family": "city_quantity", "question": t.format(city=e.name), "label": ""}
        for e in catalogue(store)
        for t in CITY_TEMPLATES
    ]
    out += [
        {"family": "family_quantity", "question": f"{name} {fam.shown} چقدره؟", "label": ""}
        for fam in W.FAMILIES
        for name in Q.CANONICAL.values()
    ]
    labels = [
        (Q.CANONICAL["r_value" if field == "element" else "hot_water"], label)
        for dataset, field in NEAR_MISS_TABLES
        for label in O.printed_options(store, dataset, field)
    ]
    labels += [
        (Q.CANONICAL["lambda"], str(r.get("display_name") or ""))
        for r in store.app07[:NEAR_MISS_MATERIALS]
    ]
    for head, label in labels:
        out.append(
            {
                "family": "near_miss_added",
                "question": f"{head} {label} {FOREIGN} چقدره؟",
                "label": f"{label} {FOREIGN}",
            }
        )
        words = label.split()
        if len(words) > 1:
            short = " ".join(words[:-1])
            out.append(
                {
                    "family": "near_miss_dropped",
                    "question": f"{head} {short} چقدره؟",
                    "label": short,
                }
            )
    return out
