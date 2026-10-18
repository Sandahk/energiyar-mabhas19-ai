"""Section-aware prose chunks from the curated main corpus (no fixed token windows).

Only retrieval-eligible blocks are used (page furniture, TOC, excluded OCR, figures, table
placeholders and formula fragments are already marked ineligible in the curated corpus; the
logical formula ``main-p073-f01`` is eligible and indexed like any block). Table-cell text is
never part of a chunk: structured tables are served by the structured lookup layer.

Rules, in source order:

* a numbered heading, a numbered clause or an unnumbered heading starts a new chunk;
* paragraphs, list items, notes and the formula join the current chunk;
* table/figure captions (with a caption continuation) form their own ``caption`` chunk;
* footnotes are page-bottom text outside the reading flow: consecutive footnotes of one page
  form ``footnotes`` chunks. Only small-font footnotes count: on the definition pages the
  curated corpus marks the 12 pt definition bodies (English term + Persian text) as
  ``footnote``; those stay in the flow with their heading;
* a chunk is split at a block boundary when it would exceed ``MAX_WORDS`` (blocks are never
  cut); every chunk keeps its block IDs, section path and per-block pages.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from mabhas19.retrieval.models import Chunk, SectionRef

MAX_WORDS = 250
FOOTNOTE_MAX_FONT = 9.5  # real footnotes are 6-9 pt; body text is 11-12 pt
MIN_WORDS = 12  # a smaller section chunk (e.g. a lone heading) merges into the next one
BOUNDARY_KINDS = frozenset({"heading", "clause", "heading_unnumbered"})
CAPTION_KINDS = frozenset({"table_caption", "figure_caption", "caption_continuation"})
FLOW_KINDS = frozenset({"paragraph", "list_item", "note", "formula"})
# unnumbered bold lines that continue the current unit instead of opening a new one
ATTACHED_CLASSES = frozenset({"english_gloss", "caption_continuation", "toc_continuation"})


def _words(text: str) -> int:
    return len(text.split())


def _path(b: dict[str, Any]) -> tuple[SectionRef, ...]:
    return tuple(
        SectionRef(p.get("number"), p.get("title", "")) for p in b.get("section_path") or []
    )


class _Builder:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self.chunks: list[Chunk] = []
        self.current: list[dict[str, Any]] = []
        self.kind = "section"

    def flush(self) -> None:
        if not self.current:
            return
        blocks, self.current = self.current, []
        path = _path(blocks[0])
        numbered = [p for p in path if p.number]
        text = "\n".join(b["text"] for b in blocks)
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.doc_id}-c{len(self.chunks) + 1:04d}",
                doc_id=self.doc_id,
                kind=self.kind,
                block_ids=tuple(b["block_id"] for b in blocks),
                block_kinds=tuple(b["kind"] for b in blocks),
                section_number=numbered[-1].number if numbered else None,
                section_path=path,
                heading_text=numbered[-1].title if numbered else None,
                text=text,
                pdf_pages=tuple(sorted({b["pdf_page"] for b in blocks})),
                printed_pages=tuple(dict.fromkeys(b.get("printed_page") for b in blocks)),
                block_pages=tuple((b["block_id"], b["pdf_page"]) for b in blocks),
                words=_words(text),
            )
        )

    def add(self, b: dict[str, Any], kind: str) -> None:
        if self.current and (kind != self.kind or self._too_big(b)):
            self.flush()
        self.kind = kind
        self.current.append(b)

    def _too_big(self, b: dict[str, Any]) -> bool:
        size = sum(_words(x["text"]) for x in self.current)
        return size + _words(b["text"]) > MAX_WORDS


def build_chunks(blocks: Iterable[dict[str, Any]], doc_id: str) -> list[Chunk]:
    eligible = [b for b in blocks if b.get("retrieval", {}).get("include") and b.get("text")]
    builder = _Builder(doc_id)
    footnotes: list[dict[str, Any]] = []
    for b in eligible:
        kind = b["kind"]
        if kind == "footnote" and (b.get("font_size") or 0) <= FOOTNOTE_MAX_FONT:
            footnotes.append(b)
            continue
        if footnotes and footnotes[-1]["pdf_page"] != b["pdf_page"]:
            _flush_footnotes(builder, footnotes)
            footnotes = []
        if kind in CAPTION_KINDS:
            if kind != "caption_continuation" or builder.kind != "caption":
                builder.flush()
            builder.add(b, "caption")
        elif kind in BOUNDARY_KINDS and b.get("structural_class") not in ATTACHED_CLASSES:
            builder.flush()
            builder.add(b, "section")
        else:
            builder.add(b, "section")
    _flush_footnotes(builder, footnotes)
    builder.flush()
    return _renumber(_merge_small(builder.chunks), doc_id)


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Merge a tiny section chunk forward into the next chunk of the same section subtree
    (or the same front-matter page), so a lone heading travels with its content."""
    out: list[Chunk] = []
    pending: Chunk | None = None
    for chunk in chunks:
        cur = chunk
        if pending is not None:
            same_tree = cur.section_path[: len(pending.section_path)] == pending.section_path
            if cur.kind == "section" and same_tree and pending.words + cur.words <= MAX_WORDS:
                cur = _join(pending, cur)
            else:
                out.append(pending)
            pending = None
        if cur.kind == "section" and cur.words < MIN_WORDS:
            pending = cur
        else:
            out.append(cur)
    if pending is not None:
        out.append(pending)
    return out


def _join(a: Chunk, b: Chunk) -> Chunk:
    text = a.text + "\n" + b.text
    return Chunk(
        chunk_id=b.chunk_id,
        doc_id=b.doc_id,
        kind="section",
        block_ids=a.block_ids + b.block_ids,
        block_kinds=a.block_kinds + b.block_kinds,
        section_number=b.section_number,
        section_path=b.section_path,
        heading_text=b.heading_text,
        text=text,
        pdf_pages=tuple(sorted(set(a.pdf_pages) | set(b.pdf_pages))),
        printed_pages=tuple(dict.fromkeys(a.printed_pages + b.printed_pages)),
        block_pages=a.block_pages + b.block_pages,
        words=a.words + b.words,
    )


def _renumber(chunks: list[Chunk], doc_id: str) -> list[Chunk]:
    return [replace(c, chunk_id=f"{doc_id}-c{i:04d}") for i, c in enumerate(chunks, 1)]


def _flush_footnotes(builder: _Builder, footnotes: list[dict[str, Any]]) -> None:
    """Footnotes are emitted after the body chunk they interrupt, grouped by page."""
    if not footnotes:
        return
    saved, saved_kind = builder.current, builder.kind
    builder.current = []
    for f in footnotes:
        if builder.current and builder.current[-1]["pdf_page"] != f["pdf_page"]:
            builder.flush()
        builder.add(f, "footnotes")
    builder.flush()
    builder.current, builder.kind = saved, saved_kind


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    sizes = sorted(c.words for c in chunks)

    def pct(p: float) -> int:
        return sizes[min(len(sizes) - 1, int(p * len(sizes)))]

    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    return {
        "chunks": len(chunks),
        "by_kind": by_kind,
        "blocks": sum(len(c.block_ids) for c in chunks),
        "words": {
            "min": sizes[0],
            "p25": pct(0.25),
            "median": pct(0.5),
            "p75": pct(0.75),
            "p95": pct(0.95),
            "max": sizes[-1],
        },
        "multi_page": sum(len(c.pdf_pages) > 1 for c in chunks),
    }
