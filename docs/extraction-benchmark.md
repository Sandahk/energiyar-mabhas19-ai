# Extraction benchmark (Phase 2A)

Measured comparison of extraction methods on 16 representative pages, run on 2026-10-04 with
`uv run python -m mabhas19.cli.extraction_benchmark` (raw outputs in `eval/runs/extraction/`,
not committed). Page set: [`configs/extraction-benchmark.yaml`](../configs/extraction-benchmark.yaml).
References: [`eval/extraction/reference/`](../eval/extraction/reference/).

## Read this first: limits of this benchmark

- **References are drafts.** About 190 items were transcribed by Claude from page renders and
  are marked `draft_pending_human_review`. Numeric cells of appendices 2 and 3 were
  cross-checked against the text layer (108 + 27 rows, 0 disagreements). Appendix 7 values
  have no independent check. Results may shift slightly once a human verifies the references.
- **Vision models were NOT run.** The adapters exist (`src/mabhas19/providers/vision.py`,
  Anthropic and Gemini, behind `--allow-external-vision`), but no external API use was
  authorized for this run. Every vision row below is `NOT_RUN`.
- **EasyOCR was NOT run.** Windows Application Control on this machine blocks PyTorch DLLs.
  The adapter and the `ocr-easyocr` dependency group are in place for another machine.
- So the offline OCR comparison covers **Tesseract 5.4 only** (`fas`+`eng`,
  `tessdata_best`). Tuning experiments (600 dpi, digit whitelist) are reported below.

## Reference corrections after the run (2026-10-06)

Human review of the appendix 7 references (pages 3, 6 and 13; 57 items now
`human_verified`) found that several stored row bounding boxes sat below the printed rows.
Measured against the vector digit glyphs of the conductivity column:

- page 13 r04-r12 were 11-14 pt low (about one row); r13-r18 6-7 pt; r01-r03 3-4 pt
- page 6 r02 and r07-r21 were 4-7 pt low

The boxes were replaced with boxes centred on the measured glyph rows (centre ±8 pt). This
corrects metadata only: no density or conductivity value changed. The `app07_rows_ocr`
results above match rows by these boxes (10 pt tolerance), so **the appendix 7 page 13 scores,
and to a lesser degree the page 6 scores, may be understated** (some OCR rows were compared
with a neighbouring reference row or with none). The benchmark has not been re-run. The
conclusion that offline OCR is unsafe for appendix 7 rests mainly on the decimal-separator and
digit errors listed under "Numeric errors", which don't depend on row matching. It should
still be confirmed by a re-run.

## Methods

| Method | Kind | Applies to | Status |
|---|---|---|---|
| `text_layer_raw` | page text | all | run |
| `repair_v1` | page text | main PDF | run |
| `tesseract_page` (300 dpi, watermark removed, psm 6) | page text | all | run |
| `easyocr_page` | page text | all | NOT_RUN (PyTorch blocked by Application Control) |
| `vision_anthropic_text` / `vision_gemini_text` | page text | all | NOT_RUN (not authorized) |
| `grid_textlayer` (ruled grid; text layer per cell, OCR only for empty cells) | structured | main table, app02, app03 | run |
| `grid_ocr` (ruled grid; Tesseract on every cell) | structured | main table, app02, app03 | run |
| `app07_rows_ocr` (rows found from vector digit glyphs; Tesseract per cell) | structured | app07 | run |
| `vision_*_json` | structured | tables | NOT_RUN |

`repair_v1` is a deterministic rebuild of the main PDF's text layer. Every step rests on
evidence from glyph IDs or geometry:

| Operation | Evidence |
|---|---|
| Space glyph → space | B Nazanin glyph 3 (the space) is mapped to U+200C. These characters occupy the visible ~3 pt word gaps. |
| Drop kashida | Glyph 47 is extracted as «ل». Rendered ink fills 16% of its box, against 55–80% for real lams (glyphs 181–184). |
| Keep ligature clusters | Lam-alef glyphs 211/212 emit «ل» plus a zero-width «ا» (glyph −1). Keeping them together preserves «لا». A genuine «ال» («الگوی») is never touched. |
| Geometric line order | Clusters are ordered by x position, as a right-to-left line with left-to-right runs (Latin, numbers). Brackets are mirrored back; guillemets aren't (they already extract logically). |
| Drop overprinted duplicates | Word fakes bold by drawing some glyphs twice at the same position (e.g. the U+2011 hyphens). |
| Reverse section numbers | Visual «1-4-2-19» becomes «19-2-4-1», applied only when the number ends in the part number 19. |
| Persian digits for display | ASCII digit codes drawn with B fonts appear as Persian digits on the page. Latin fonts keep ASCII. |
| Infer ZWNJ | A final-form glyph (learned per glyph ID from the whole document) followed directly by a letter means a ZWNJ was there. |
| Insert gap spaces | Gaps wider than 0.3 em with no space glyph get a space. This is the source of the residual errors below. |

Three representations stay separate: `raw_text` (never modified), `repaired_display_text`,
and `normalized_search_text`.

## Results: page → text

Mean over reference items. **Normalized CER / WER (%)** ignores superficial differences
(Arabic vs Persian letter forms, digit script, ZWNJ vs space, punctuation style).
Seconds are per page.

| Page | text_layer_raw | repair_v1 | tesseract_page |
|---|---|---|---|
| main p11 prose | 2.3 / 14.0 | **0.1 / 0.9** (0.02 s) | 3.3 / 7.0 (3.2 s) |
| main p21 definitions | 3.8 / 12.4 | **0.0 / 0.0** | 5.0 / 12.7 |
| main p62 clauses | 5.2 / 25.1 | **0.0 / 0.4** | 11.3 / 21.6 |
| main p84 table (as text) | 14.7 / 53.7 | **1.6 / 16.4** | 50.6 / 75.5 |
| main p5 scanned letter | 93.2 / 100 | 87.8 / 100 | **7.5 / 15.1** |
| app01 p2 glossary | 90.7 / 100 | n/a | **1.7 / 4.5** |
| app02 p2 table (as text) | 66.3 / 60.5 | n/a | 63.5 / 100 |
| app03 p2 table (as text) | 90.8 / 100 | n/a | 54.2 / 88.2 |
| app04 p3 prose | 93.9 / 100 | n/a | **3.2 / 6.6** |
| app04 p12 text table | 100 / 100 | n/a | 31.0 / 54.0 |
| app07 p3 / p6 / p13 | 83–91 / 100 | n/a | 14.4–18.9 / 27–34 |
| app09 p2 Persian list | 100 / 100 | n/a | **6.1 / 12.4** |
| app09 p7 English (outlines) | 88.5 / 100 | n/a | **0.0 / 0.0** |
| app09 p9 English (text layer) | 20.4 / 46.2 | n/a | **0.0 / 0.0** |

Grouped:

| Group | Items | text_layer_raw | repair_v1 | tesseract_page |
|---|---|---|---|---|
| Main text (p11, 21, 62) | 22 | 4.2% norm / 18.1% raw CER | **0.0% / 0.9%** (21 of 22 items exact) | 7.9% / 9.0% (5 exact) |
| Appendix prose, terms, lists | 31 | 78.5% | n/a | **1.9% / 2.7%** (23 exact) |
| Appendix tables read as plain text | 105 | 82.6% | n/a | 33.9% (23 exact) |

## Results: page or region → structured records

| Page | Method | Numeric cells exact | Labels exact | Climate codes exact | Complete rows | s/page |
|---|---|---|---|---|---|---|
| main p84 (table 19-5-1) | grid_textlayer | **36/36** | n/a | n/a | 6/6 | 0.9 |
| main p84 | grid_ocr | 0/36 | n/a | n/a | 0/6 | 9.6 |
| app02 p2 | grid_textlayer + OCR for city names | n/a | 21/27 | **27/27** | 21/27 | 4.1 |
| app02 p2 | grid_ocr | n/a | 14/27 | 18/27 | 14/27 | 10.8 |
| app03 p2 | grid_textlayer + OCR for labels | **108/108** | 5/10 | n/a | 7/12 | 22.7 |
| app03 p2 | grid_ocr | 108/108 | 5/10 | n/a | 7/12 | 58.7 |
| app07 p3 | app07_rows_ocr | 11/40 | — | n/a | 0/14 | 4.8 |
| app07 p6 | app07_rows_ocr | 17/63 | — | n/a | 0/21 | 7.6 |
| app07 p13 | app07_rows_ocr | 6/60 | — | n/a | 0/21 | 7.8 |

The numeric cells of appendix 3 are real text (Latin digits). That's why OCR also reads them
perfectly; it isn't evidence that OCR handles Persian digits.

## Numeric errors

There were **165 wrong or missing values** across the OCR-based structured methods, all in
`grid_ocr` and `app07_rows_ocr`:

| Error type | Count | Examples (expected → detected) |
|---|---|---|
| Wrong digits | 89 | 0.80 → «۸/۸۰» (8.80); 625 → «5»; 1.69 → «۱۸۳۹» |
| Missing or unparseable | 46 | 2000 → nothing; 0.23 → «VY»; 72 → «oy» |
| Decimal separator dropped | 30 | 1.96 → «۱۹۶» (196); 1.30 → «۱۳۰» (130); 1.15 → «۱۱۵» (115) |

- **All 165 end up flagged `requires_verification`.** In an earlier run one error was *not*
  flagged: Tesseract reported confidence ≥ 90 for «5» where the source prints «۶۲۵».
  **OCR-derived values are therefore capped at `medium` confidence**, so they always require
  verification. Engine confidence is not a usable safety signal for regulatory numbers.
- **The text-layer methods produced zero numeric errors** (144/144 cells).
- **Tuning didn't help.** On the 14 conductivity cells of app07 p3, Tesseract scored 0/14
  exact at 300 dpi, 0/14 at 600 dpi, and the digit whitelist emptied the output entirely.
- **The parser never guesses.** A missing leading zero (`.35`) stays `.35` at `low`
  confidence. «/» counts as a decimal separator only in numeric-cell context (dates such as
  «۱۴۰۴/۰۴/۱۹» stay unparsed). Mixed digit scripts are flagged. Commas are never resolved.

## Before/after examples (main PDF repair)

| Raw text layer | Repaired display text |
|---|---|
| `19` · `‑` · `2` · `‑` · `4 رده` (separate lines) | `۱۹-۲-۴ رده‌بندی بازدهی انرژی ساختمان‌ها` |
| `1-4-2-19 ‌بازدهی‌انرژی‌ساختمان‌ها‌بر‌مبنای…` | `۱۹-۲-۴-۱ بازدهی انرژی ساختمان‌ها بر مبنای شدت مصرف انرژی …` |
| `حداکثر‌زان یم‌مجاز‌شللدت‌مصللرف‌انرژی‌ی برا‌سللاختمانی ها` | `حداکثر میزان مجاز شدت مصرف انرژی برای ساختمان‌های` |
| `جمهوری‌اسالمی` (p62), `الزم‌االجرا` (p61) | `جمهوری اسلامی`, `لازم الاجرا` (while «الگوی» is unchanged) |
| table cell `09` + `/1` | `۱/۰۹` → numeric 1.09 (`/` read as decimal in cell context) |

Residual errors come from the gap-space rule splitting a few words at wide glyph gaps:
«گران‌ق یمت» and «خارج ی». That's about 2 words per page, and they are visible in the 0.9% raw
CER. The source typo «حافظت» (SI-0008) is reproduced faithfully.

## OCR examples (appendices, Tesseract)

| Reference | Tesseract output |
|---|---|
| ASTM E2797: Standard Practice for Building Energy Performance … | identical (0% CER, app09 p7) |
| به این ترتیب با کاهش میزان مصرف انرژی‌های فسیلی، … | … انرژی های فسیلی" … (punctuation only) |
| آبگرم مصرفی | ابگر p مصرفی |
| … مبحث ۵ مقررات ملی در خصوص … | … مبحث 5 مقررات jo در خصوص … |
| app02 table page as plain text | `وش / وی ی / ‎ET‏ / 8[ ی / …` (unusable) |

## Structured extraction examples

```json
{"row_number": 5, "city_raw": "اردبیل", "climate_class_raw": "5C", "climate_class": "5C",
 "source": {"doc_id": "m19-ed5-1404-app02", "pdf_page": 2, "method": "grid_textlayer"},
 "verification_status": "unverified"}
```

```json
{"row_number": 1, "occupancy_type_raw": "مسکونی ۳۰۰۰2 مترمربع", "energy_rating": "D",
 "climate_class": "3A", "max_eui_kwh_m2_year": {"raw_detected_value": "192",
 "normalized_value": "192", "confidence": "high", "requires_verification": false}}
```

The label above shows the SI-0003 operator misread as «2».

```json
{"material_raw": "Veo‏ اسمی: چگالی -", "density": {"kind": "range", "raw": "۵ تا ۷۲۵",
 "min": {"raw_detected_value": "5", "normalized_value": "5", "confidence": "medium",
 "requires_verification": true}}, "thermal_conductivity": {"raw_detected_value": "۵",
 "normalized_value": "5", "confidence": "low", "requires_verification": true}}
```

The source row is «چگالی اسمی: ۷۰۰ | ۶۷۵ تا ۷۲۵ | ۰٫۲۵».

Candidate schemas are in `src/mabhas19/tables/schemas.py`:
- `CityClimateRow` (appendix 2)
- `EnergyLimitCell` (appendix 3)
- `MaterialPropertyRow` (appendix 7)

The appendix 7 schema was shaped by inspecting pages 3, 6 and 13. It has:
- category (may continue from the previous page)
- material (`material_inherited` for continuation rows)
- condition (nominal density, rebar %, mix ratio)
- density as range, `more_than`, `less_than` or single
- conductivity
- units and footnote references

## Recommendations

| Document type | Recommended method | Measured accuracy | Speed | Complexity | Local/API | Likely production cost |
|---|---|---|---|---|---|---|
| **Main PDF text** (139 pages) | **`repair_v1`** deterministic text-layer repair | 0.0% normalized / 0.9% raw CER; 21/22 items exact. Tesseract: 7.9%; raw layer: 4.2% | ~0.02 s/page (+6 s one-off glyph learning) | Medium (glyph evidence, bidi); fully testable | Local | ≈ 0 |
| **Main PDF tables** (11 tables) | **`grid_textlayer`** + human verification | 36/36 numeric cells; OCR 0/36 | ~1 s/page | Low | Local | ≈ 0 + review time |
| **Main PDF scanned page** (p5 only) | OCR (Tesseract) + human proofreading, or vision once authorized | 7.5% normalized CER | 2 s | Low | Local | negligible (1 page) |
| **Appendix prose** (app01, 04, 05, 06, 08, 09) | **Provisional: Tesseract `fas+eng`**. Re-decide after a vision run. | 1.9% normalized CER on 31 items. Word-level errors remain («ملی» → «jo»), so human proofreading is needed. | 2–5 s/page | Low | Local | ≈ 0 + proofreading |
| **Appendix 2** (76 rows) | **Hybrid:** row numbers and climate codes from the text layer (27/27), city names by OCR (21/27) → **human verification of every city name** | climate 100%, cities 78% | ~4 s/page | Low | Local | ≈ 0 + ~15 min review |
| **Appendix 3** (4 tables, 136 cells each) | **Hybrid:** all numeric cells from the text layer (108/108), use-type labels curated manually (34 labels shared by the 4 tables; OCR 5/10) | numbers 100% | ~23 s/page | Low | Local | ≈ 0 + ~20 min label curation |
| **Appendix 7** (11 pages, ~200 rows) | **Not offline OCR**: 10–27% numeric accuracy, with the decimal separator often lost. **Benchmark vision structured extraction next**, then verify every value against the page (double-keyed). If vision isn't authorized, manual double-keyed transcription is feasible at this size. | Tesseract 34/163 values | 5–8 s/page | High (layout varies) | API (vision) or manual | Vision: estimated a few dollars for all 78 appendix pages at Claude Opus 5.5 rates ($4/$20 per MTok; roughly 2–4k tokens per page). Manual: ~3–4 hours double-keyed. |

Accuracy decides, not price:
- For every source whose numbers are real text (main PDF, appendix 2 codes, appendix 3
  values), the text layer is exact and cheap, so OCR should never replace it there.
- Where the digits are vector outlines (appendix 7), offline OCR is measurably unsafe.

Pending decisions for you:
1. Authorize a vision benchmark run (Anthropic and/or Gemini) on the same 16 pages, so the
   appendix prose and appendix 7 decisions can rest on measurements.
2. Have a human verify the draft references (most important: appendix 7 and appendix 1).

## Source issues found or confirmed

Recorded in [`curated/source_issues.yaml`](../curated/source_issues.yaml), all `unresolved`:

| ID | Issue |
|---|---|
| SI-0001 | Appendix 2 has a 5C row («اردبیل»). |
| SI-0002 | Appendix 3 has no 5C column. |
| SI-0003 | Appendix 3 rows 1–2 have ambiguous ≤/> operators. |
| SI-0004 | Appendices 5, 6 and 8 are placeholders; there is no lighting table. |
| SI-0005 | Appendix 9 heading is numbered «پ ۱۳-۱». |
| SI-0006 | Appendix 7 prints «فلزت». |
| SI-0007 | Appendix titles differ between the covers and the main document's list. |
| SI-0008 | Main PDF misspells «حافظت» on PDF page 20. |

---

# Phase 2B – Gemini Vision / Appendix 7

Run 2026-10-12 with `gemini-3.8-flash` (Gemini API, Standard tier), reproducible offline with:

```sh
# re-score the cached responses (no API calls); --with-tesseract adds the local OCR comparison
uv run python -m mabhas19.cli.score_app07 --with-tesseract
```

The extraction runner is `mabhas19.cli.gemini_benchmark` (explicit `--allow-external`; key from
the git-ignored `.env`). Raw responses are cached under the ignored
`data/interim/gemini-phase2b/gemini-3.8-flash/responses/` and are not committed.

## Bottom line

- **Gemini achieved 163/163 exact numeric values and 56/56 fully numerically correct rows on
  the human-verified Appendix 7 benchmark**, both from whole pages and from row crops.
- **Raw Gemini label/structure output is not yet production-ready and requires deterministic
  post-processing plus human review.**
- Only Appendix 7 was sent. Other appendices, the main document and the full corpus have not
  been run through Gemini.

## Evaluation set

- 56 `human_verified` rows on Appendix 7 PDF pages 3 (14 rows), 6 (21) and 13 (21);
  163 numeric values (56 conductivities, 107 density bounds).
- Only verified items count. Every number is **re-parsed from Gemini's raw text** with the
  project's non-guessing parser (`slash_is_decimal` in numeric cells only). The model's own
  `*_parsed` fields are stored but never credited.
- Prompts carry the numeric rule: *Never infer, repair, or guess a number from context.
  Transcribe only what is visibly present.* Raw and parsed values are kept separately.

## Methods compared

| Method | What is sent / run |
|---|---|
| Gemini whole page | one 200-dpi page image → JSON rows (raw density, raw conductivity, raw category/subgroup/material/condition, parsed values) |
| Gemini row crop | one 300-dpi crop per verified row (56 crops; crop boxes from the verified row bounding boxes) |
| Tesseract whole page | `fas+eng`, 300 dpi, words assigned to rows/columns by position |
| Tesseract per cell | `fas+eng`, cells located from the vector digit glyphs (`app07_rows_ocr`) |

## Numeric results (human-verified rows)

| Measure | Gemini whole page | Gemini row crop | Tesseract whole page | Tesseract per cell |
|---|---|---|---|---|
| Page 3 | 40/40 | 40/40 | 10/40 | 11/40 |
| Page 6 | 63/63 | 63/63 | 16/63 | 17/63 |
| Page 13 | 60/60 | 60/60 | 7/60 | 13/60 |
| **Exact numeric (all)** | **163/163** | **163/163** | 33/163 (20%) | 41/163 (25%) |
| Fully numerically correct rows | **56/56** | **56/56** | 0/56 | 0/56 |
| Decimal preserved (`٫` value correct) | **47/47** | **47/47** | 0/47 | 2/47 |
| Ranges («… تا …», both bounds) | **35/35** | **35/35** | 0/35 | 0/35 |
| Inequality qualifiers («بیش از» / «کمتر از») | **5/5** | **5/5** | 4/5 | 2/5 |
| Integer conductivities (72, 52, 56, 230 …) | **9/9** | **9/9** | 2/9 | 5/9 |
| Five-digit density «۱۱۳۴۰» (min and max) | **2/2** | **2/2** | 0/2 | 0/2 |
| Numeric mismatches | **0** | **0** | 130 | 122 |

Row detection (whole page): 14/14, 21/21 and 21/21 rows returned and matched, in order; no
missing or extra rows.

## Label / structure results (Gemini whole page, raw fields)

| Field | Strict (normalized) | Lenient (ignores list dash, trailing colon, footnote markers) |
|---|---|---|
| Category | 35/42 | 35/42 |
| Subgroup | 35/54 | 53/54 |
| Material | 31/49 | 36/49 |
| Condition | 44/56 | 44/56 |
| **All** | **145/201 (72%)** | **168/201 (84%)** |

- Expected label text present somewhere in the row's raw fields: 97/105 (92%): most
  failures are text placed in the wrong field, not misread.
- Context handling: 6 inherited materials correctly left empty; 16 of 17 values that are not
  printed on the page (page 6 continues group 2 from page 3 and a sub-group from page 5)
  correctly left empty; 1 filled with the visible line in the wrong field.
- Row crops: material label correct (lenient) on 28/30 rows that print their own label;
  page 3 r13 returned the condition line, page 6 r02 lost a wrapped second line that the
  crop box cut off.

### Known structural error classes

1. **Condition placed in the material field.** Dashed sub-lines («- چگالی اسمی: ۸۰۰»,
   «- نسبت: ۱ به ۶», the mix-ratio lines) come back as `material_raw` with `condition_raw`
   empty: page 6 r01, r03–r11; page 13 r20–r21.
2. **Footnote markers attached to headings, with text loss.** «گرماشکنی²» → «گرماشکن ۲»
   (final «ی» lost) on page 6 r15–r21 (one heading); «اتوکلاو شده¹:» → «اتوکلاو شده ۱:».
3. **Line merged across rows.** Page 13: the r19 subgroup line was appended to r18's
   material, and r19's subgroup is empty.
4. **Heading duplicated into a field.** Page 3 r01: the category heading also returned as
   the material.
5. **Formatting differences** (list dash, trailing colon, inline footnote digit): harmless
   after normalization, counted only by the strict measure.

Required before production use: deterministic post-processing (strip dashes/colons, split
footnote markers into references, apply the subgroup/condition convention and label
inheritance, re-attach cross-page context) and human review of labels. Numbers still need
verification against the page (current policy for all extracted regulatory values).

## Requests, latency, tokens and measured cost

| Requests | Count | Mean latency (min–max) | Input / output / thinking tokens | Cost |
|---|---|---|---|---|
| Whole page | 3 | 46.6 s (15.8–76.6) | 4,059 / 7,421 / 14,385 | $0.0848 (≈ $0.028/page) |
| Row crop | 56 | 9.4 s (2.6–40.6) | 65,975 / 4,782 / 51,634 | $0.2610 (≈ $0.0047/crop) |
| **Total** | **59** | | | **$0.3458** |

Prices: $0.75 input and $3.75 output (including thinking) per 1M tokens, Standard tier
through 2026-12-31 (they double from 2027-01-01). Thinking tokens are most of the cost.
All 59 successful requests completed on the first attempt (no 429, 503 or timeouts).

### Earlier free-tier behaviour (before the paid key)

- Pro models (`gemini-3.1-pro-preview`, `gemini-pro-latest`) returned `429
  RESOURCE_EXHAUSTED` (no free quota); `gemini-2.5-pro`/`-flash` are retired for new users.
- Image requests to Flash models repeatedly returned `503 UNAVAILABLE` ("high demand") for
  periods of 15 minutes to several hours, while text-only prompts succeeded.
- The free-tier daily limit for `gemini-3.8-flash` was **20 requests per project per day**
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), exhausted after probes and retries.
- One request hung for over an hour; the adapter now uses a 180 s timeout, and timeouts
  are retried like server errors.

## Phase 2B conclusion (Appendix 7)

| Question | Answer |
|---|---|
| Numbers from vector-drawn Persian digits | Gemini whole page and crops: 163/163; Tesseract unusable (20–25%). |
| Whole page or crop? | Equal numeric accuracy; whole page is cheaper per row and returns structure; crops are useful as an independent second read. |
| Production readiness | Numbers: strong evidence on this benchmark, still subject to verification. Labels/structure: not production-ready without post-processing and review. |
| Not yet benchmarked with Gemini | Appendices 1–4, 9, the main PDF scan, and plain-text transcription. |
