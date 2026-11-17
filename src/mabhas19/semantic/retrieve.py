"""Offline row retrieval within one deterministically resolved quantity (option B).

Stage 1 (quantity) is :func:`mabhas19.api.quantities.analyse` - deterministic, never semantic. Stage
2 (table) follows from the quantity. Stage 3 ranks only that quantity's printed rows by the frozen
E5 model (``intfloat/multilingual-e5-small``, "query: " / "passage: ") plus a character-trigram
containment score, and turns the ranking into a decision with fixed thresholds:

* ``confirm`` - one row stands out: the user is asked «منظور شما «…» است؟»;
* ``choose`` - two or three rows are close: the user chooses;
* ``abstain`` - no row is strong enough, more than three are close, or there is no row wording.

Nothing here answers: a chosen row becomes a canonical printed question for the unchanged engine
(:mod:`mabhas19.semantic.verify`).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mabhas19.api import quantities as Q
from mabhas19.api.wording import letters
from mabhas19.generation import options as O

if TYPE_CHECKING:
    from mabhas19.retrieval.structured import StructuredStore
    from mabhas19.semantic.catalog import Catalogue, Row

Decision = Literal["confirm", "choose", "abstain"]
TOP_K = 5
MAX_CHOICES = 3

# words that carry no row meaning in a value question (colloquial verbs and connectives)
STOP = frozenset(
    {
        *Q.FILLERS,
        "چه",
        "چی",
        "چیه",
        "کدوم",
        "کدام",
        "میخواد",
        "می‌خواد",
        "میخوام",
        "داره",
        "دارد",
        "داشته",
        "باشن",
        "بشه",
        "میشه",
        "می‌شود",
        "میشود",
        "کنه",
        "کند",
        "یه",
        "یک",
        "این",
        "اون",
        "آن",
        "ما",
        "من",
        "مون",
        "مثلا",
        "معمولی",
        "معمولا",
        "نوع",
        "مدل",
        "اگه",
        "اگر",
        "هم",
        "ها",
        "های",
        "ی",
        "با",
        "بر",
        "اساس",
        "حدود",
        "خونه",
    }
)
_WORD = re.compile(r"[^\W\d_]+|[A-Za-z]+")
_DIGIT = re.compile(r"[0-9۰-۹٠-٩]")
_CODE = re.compile(r"(?<![A-Za-z0-9])[0-5۰-۵]\s*[ABCabc](?![A-Za-z])")


@dataclass(frozen=True)
class Settings:
    """Frozen with the experiment manifest; tuned on development questions only."""

    lex_weight: float = 0.3
    floor: float = 0.86  # minimum combined score of any offered row
    band: float = 0.02  # rows within this of the best are equally plausible

    def to_dict(self) -> dict[str, Any]:
        return {"lex_weight": self.lex_weight, "floor": self.floor, "band": self.band}


@dataclass
class Scored:
    row: Row
    cosine: float
    lexical: float
    score: float


@dataclass
class Proposal:
    quantity: str  # "" when stage 1 did not resolve exactly one quantity
    words: list[str]
    ranked: list[Scored] = field(default_factory=list)
    decision: Decision = "abstain"
    reason: str = ""
    offered: list[Row] = field(default_factory=list)
    code: str = ""
    cities: list[str] = field(default_factory=list)
    ms: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "words": self.words,
            "decision": self.decision,
            "reason": self.reason,
            "offered": [r.label for r in self.offered],
            "top": [
                (s.row.label, round(s.cosine, 4), round(s.lexical, 4), round(s.score, 4))
                for s in self.ranked
            ],
            "code": self.code,
            "cities": self.cities,
            "ms": round(self.ms, 1),
        }


def _key(text: str) -> str:
    return O.key(letters(text))


def _trigrams(text: str) -> set[str]:
    out: set[str] = set()
    for word in _key(text).split():
        w = f"#{word}#"
        out |= {w[i : i + 3] for i in range(len(w) - 2)}
    return out


def containment(words: str, passage: str) -> float:
    """Share of the user's row-word trigrams found in the printed passage."""
    q = _trigrams(words)
    return len(q & _trigrams(passage)) / len(q) if q else 0.0


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    def __init__(
        self, catalogue: Catalogue, store: StructuredStore, embedder: Any, settings: Settings
    ) -> None:
        self.catalogue = catalogue
        self.embedder = embedder
        self.settings = settings
        self.cities = {
            _key(str((r.get("city") or {}).get("text") or ""))
            for r in store.app02
            if (r.get("city") or {}).get("text")
        }
        passages = [r.passage() for r in catalogue.rows]
        vectors = embedder.embed_documents(passages)
        self.vectors = {r.id: v for r, v in zip(catalogue.rows, vectors, strict=True)}

    def row_words(self, question: str, request: Q.Request) -> tuple[list[str], list[str]]:
        """(the words that may name the row, the printed city names written in the question)."""
        cities = [w for w in request.leftover if _key(w) in self.cities]
        words = [
            w
            for w in request.leftover
            if w not in STOP and _key(w) not in self.cities and len(w) > 1
        ]
        return words, cities

    def propose(self, question: str) -> Proposal:
        start = time.perf_counter()
        request = Q.analyse(question)
        if request is None:
            p = Proposal("", [], reason="quantity not resolved (none, several, or conceptual)")
            p.ms = (time.perf_counter() - start) * 1000
            return p
        words, cities = self.row_words(question, request)
        p = Proposal(request.quantity.id, words, code=request.code, cities=cities)
        rows = self.catalogue.for_quantity(request.quantity.id)
        if _DIGIT.search(_CODE.sub(" ", letters(question))):
            # a number (area, density, power, a band boundary) is a printed condition only the
            # engine may check: retrieval never acts on it
            p.reason = "the question states a number"
        elif not words:
            p.reason = "no row wording"
        elif not rows:
            p.reason = "the quantity has no printed rows"
        else:
            text = " ".join(words)
            qv = self.embedder.embed_query(text)
            s = self.settings
            scored = []
            for r in rows:
                c = _cos(qv, self.vectors[r.id])
                lx = containment(text, r.passage())
                scored.append(Scored(r, c, lx, c + s.lex_weight * lx))
            scored.sort(key=lambda x: x.score, reverse=True)
            p.ranked = scored[:TOP_K]
            best = p.ranked[0].score
            close = [x for x in p.ranked if x.score >= s.floor and best - x.score <= s.band]
            if not close:
                p.reason = "no row above the floor"
            elif len(close) == 1:
                p.decision, p.offered, p.reason = "confirm", [close[0].row], "one row stands out"
            elif len(close) <= MAX_CHOICES:
                p.decision, p.offered = "choose", [x.row for x in close]
                p.reason = f"{len(close)} close rows"
            else:
                p.reason = f"{len(close)} rows equally close"
        p.ms = (time.perf_counter() - start) * 1000
        return p
