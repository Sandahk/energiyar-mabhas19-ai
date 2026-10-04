# Source audit (Phase 1)

Page-by-page inventory of the 10 source PDFs, produced on 2026-10-04 by

```sh
uv run python -m mabhas19.cli.audit_sources
```

with outputs in `data/interim/<doc_id>/<sha8>/` (`pages.jsonl`, `page-map-candidates.jsonl`,
`heading-candidates.jsonl`, `table-inventory.jsonl`, `document.json`, `audit-samples/`).
Everything there is a **machine suggestion**; nothing is verified. It complements a visual
review of all 224 rendered pages (recorded in
[`docs/audit/manual-table-observations.yaml`](audit/manual-table-observations.yaml), also
unverified). File identities and hashes live in [`sources/manifest.yaml`](../sources/manifest.yaml).

## Summary matrix

| Document | Pages | Text layer | Persian extractable | Tables | Likely next method |
|---|---|---|---|---|---|
| main (`mab15-v5.pdf`) | 146 | Word export; 139 DAMAGED, 7 NO_TEXT | Yes, but damaged | 11 ruled tables (auto = visual) | **Text repair** of the text layer, checked against renders; **structured table extraction** for 11 tables; **OCR** for 1 scanned page (PDF p5) |
| app01 Glossary | 8 | Print-to-PDF; 7 LATIN_DIGITS_ONLY, 1 NO_TEXT | **No** (vector outlines) | Glossary term pairs (not ruled) | **OCR/vision** for Persian column; English column from text layer; **structured extraction** of term pairs |
| app02 City → climate class | 4 | 3 LATIN_DIGITS_ONLY, 1 NO_TEXT | **No** | 1 table over 3 pages (76 rows) | **Structured table extraction**: row no. and class from text layer, city names by **OCR/vision**; **manual verification** |
| app03 Energy-use limits | 5 | 4 LATIN_DIGITS_ONLY, 1 NO_TEXT | **No** | 4 tables (classes D, C, B, A) | **Structured table extraction**: numeric cells from text layer, row labels by **OCR/vision**; **manual verification** |
| app04 LCA | 32 | 15 LATIN_DIGITS_ONLY, 17 NO_TEXT | **No** | 1 small table (p12, missed by auto); ~8 figures | **OCR/vision** for prose; figures/formulas need **vision** or manual transcription |
| app05 Prescriptive checklist | 2 | 1 LATIN_DIGITS_ONLY, 1 NO_TEXT | **No** | None — placeholder | **OCR** of one note; content "to be published" |
| app06 Energy qualification | 2 | 2 NO_TEXT | **No** | None — placeholder | **OCR** of one note; content "to be published" |
| app07 Material thermal properties | 13 | 4 LATIN_DIGITS_ONLY, 9 NO_TEXT | **No** | 1 long table over 11 pages | **Vision/OCR table extraction** — *all values are Persian digits drawn as vectors*; highest-risk table; **manual verification** of every value |
| app08 Lighting limits | 2 | 2 NO_TEXT | **No** | **None — the expected table is absent** | Placeholder; nothing to extract |
| app09 Reference standards | 10 | 1 LATIN_DIGITS_ONLY, 9 NO_TEXT | **No** | None (lists) | **OCR/vision** (Persian and most English references are vector outlines; only p9 has Latin text) |

The final OCR tool is deliberately not chosen yet (Phase 2).

## Text-layer quality

| Category | main | app01 | app02 | app03 | app04 | app05 | app06 | app07 | app08 | app09 | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GOOD | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| DAMAGED | 139 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **139** |
| LATIN_DIGITS_ONLY | 0 | 7 | 3 | 4 | 15 | 1 | 0 | 4 | 0 | 1 | **35** |
| NO_TEXT | 7 | 1 | 1 | 1 | 17 | 1 | 2 | 9 | 2 | 9 | **50** |

Classification (`src/mabhas19/ingest/audit/quality.py`): NO_TEXT < 20 letters/digits;
LATIN_DIGITS_ONLY < 10 Persian letters or < 5 % of letters; otherwise DAMAGED if any damage
signal exceeds its threshold (ZWNJ per Persian letter > 0.06, isolated-letter words > 15 %,
presentation forms > 5 %, ≥ 2 reversed lam-alef «اال» sequences), else GOOD.

## Main document — `m19-ed5-1404-main`

- **Structure.** 146 PDF pages. Printed table of contents at PDF pages 15–17. **No PDF
  bookmarks/outline and no PDF page labels.** The printed TOC is the only structural index.
  Appendices are listed on PDF p146 as published online (via QR code), not printed.
- **Text layer (all 139 text pages DAMAGED).** ZWNJ ratio 0.078–0.307 per Persian letter
  (normal Persian ≈ 0.01–0.03): ZWNJ replaces spaces; lam-alef is reversed («اسالمی»,
  «الزم‌االجرا»); words split («انرژ ی»). Digits are stored as ASCII and *displayed* as
  Persian digits. Headings sometimes use Arabic kaf «ك» (e.g. «كلیات»).
- **Section numbers** are extracted as separate fragments in visual order: «19», «‑», «2»,
  «‑1 دامنه …» (hyphens are U+2011). Body clauses use U+002D and come out fully reversed:
  «1-4-2-19» renders as ۱۹-۲-۴-۱, i.e. clause **19-2-4-1**.
- **No-text pages:** 4, 6, 8, 18, 54, 140 (blank) and **5 (full-page scanned image of the
  notification letter — needs OCR)**.
- **Raster images:** an «inbr» watermark image on every page (excluded from counts); content
  images on pages 1, 2, 5, 57, 63, 64, 66, 67, 85, 146 (logo, QR codes, letter, figures, map).
- **Page markers** (centred footer): front matter uses letters in Arabic alphabetical order
  (أ ب ت ث ج …, without پ چ ژ گ), body uses numbers. Blank pages carry *hidden* marker text
  that does not render (e.g. «110» on p140), which the detector rejects with a rendering check.
- PDF p75's page number is printed in blue (cosmetic, but it defeated a naive ink threshold).

## Appendices — `m19-ed5-1404-app01` … `app09`

- All nine were made with **Microsoft: Print To PDF**. Persian glyphs (and Persian digits)
  are **drawn as vector outlines, not text**: zero Persian characters are extractable.
  (Phase 0 attributed this to font encoding; the drawing counts — hundreds to thousands of
  paths per page — show the glyphs are outlines.) Some Latin text and ASCII digits *are*
  real text (e.g. app02 climate codes, app03 numeric cells, app01 English terms).
- Even English text is partly outlined: app09 p6–8 English references yield only bullet
  dashes; only p9 has extractable Latin text.
- Each appendix starts with an unnumbered cover page; content pages are numbered ۱, ۲, …
  (bottom centre, vector-drawn), restarting in every appendix.
- **Appendices 5, 6 and 8 are placeholders** (cover + one short note saying the content will
  be published later). **Appendix 8 contains no lighting-limit table.**
- Title discrepancies between appendix covers and the list on main p146 (e.g. appendix 6
  cover «فرآیند آموزش، سنجش و احراز صلاحیت انرژی» vs list «آیین‌نامه آموزش، ارزیابی و
  احراز صلاحیت بازرسان انرژی ساختمان»; appendix 4 list adds «… و معیارهای ارزیابی و
  رتبه‌بندی ساختمان سبز»). To be resolved during curation, not here.
- Appendix 9 page header reads «پ ۱۳-۱ استانداردها و آیین‌نامه‌های مرجع» — numbering that
  does not match «پیوست ۹»; recorded as-is.
- The PDF `Title` metadata is mojibake suggesting all appendices were printed from one
  combined file («متن کامل ویرایش پنجم مبحث 19 … (با شماره گذاری صفحات پیوست ها)»). If that
  original can be obtained it may have a real text layer (add it as a new source version).

## Page-number findings

| Document | Visible markers | Mapping (machine suggestion) | Confidence |
|---|---|---|---|
| main | p2–17 letters; p19–53, 55–139, 141–146 numbers | front matter: letter index = PDF page; p19–53 → 1–35; p55–139 → 36–120; p141–146 → 121–126; no visible marker on p1, 4, 6, 8, 18, 54, 140 | 137 high, 2 medium (p5, p7: isolated letters next to blank pages), 7 none |
| app01–09 | vector glyphs only | cover unnumbered; PDF page *n* → printed *n − 1* | all **low** (inferred from glyph presence; needs OCR/visual check) |

Cross-check: all **69 printed-TOC entries match** the printed page found at the
corresponding body heading (0 mismatches, 0 missing). Glyph-cluster counts disagree with
the inferred digit count on 7 appendix pages (app04 p24, 27–29; app07 p3, 5; app09 p7),
caused by footnote glyphs or merged digits; flagged in the candidate notes.

## Section-heading findings (main)

- 229 bold body headings + 22 regular-weight numbered clauses + 69 TOC entries.
- Chapters 19-1 (p19) … 19-8 (p141); depth distribution: 8 × two-level, 187 × three-level,
  30 × four-level, 4 × five-level.
- Every body heading increases strictly in document order (no sequence breaks).
- 160 body headings are deeper than the printed TOC lists, so the body is the fuller
  source of structure.
- Hyphen code points: U+2011 in headings (294 candidates), U+002D in numbered clauses (26).

## Table inventory

**Main** (auto-detected ruled tables; identical page set from visual review): PDF p64 (large
timing matrix, very small print), p67 (19-3-1), p68 (19-3-2), p74 (19-3-3), p84 (19-5-1),
p90 (19-5-2), p99 (19-5-3), p100 (19-5-4), p107 (19-5-5), p115 (19-5-6), p116 (19-5-7).
Auto detection also flags p66 (a 1×2 box in figure 19-3-1, false positive) and 16 pages that
only *mention* «جدول» in running text.

**Appendices** (visual review; auto detection unreliable on vector text):

| Doc | PDF pages | Content |
|---|---|---|
| app02 | 2–4 | one table, 76 rows: row no. / city / climate class |
| app03 | 2–5 | four tables P3-1…P3-4 (classes D, C, B, A): ~34 use types × 9 climate columns (0B…5B) |
| app07 | 3–13 | one table: material / dry density / thermal conductivity, 15 numbered material groups |
| app04 | 12 | ISO 14040 stage structure (text only) |
| app01 | 2–8 | glossary term pairs (table-like, not ruled) |
| app08 | — | **no table** (placeholder) |

Observation for later curation (not interpreted here): app02 assigns class «5C» to at least
one city, while the app03 tables have no 5C column.

## Replacing a source with a better original

Never overwrite an existing file. Add the new file under a new canonical name (e.g.
`m19-ed5-1404-app04-v2.pdf`), append a `v2` entry with `supersedes: v1`, set
`current_version: v2`, and record its provenance. Audit and derived outputs are keyed by
source SHA-256 (`data/interim/<doc_id>/<sha8>/`), so outputs from old and new versions never
mix; re-run the audit for the new version.
