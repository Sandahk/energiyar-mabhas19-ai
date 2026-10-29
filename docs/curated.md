# Curated knowledge (`curated/`)

The `curated/` folder holds human-reviewed knowledge extracted from the official PDFs in `sources/`. Retrieval reads only
from here, never directly from the PDFs. Everything is keyed by document ID
(`m19-ed5-1404-main`, `m19-ed5-1404-app02`, …).

| Path | Contents |
|---|---|
| `text/` | Main text as reviewed blocks (`blocks.jsonl`), the input for prose chunks |
| `structure/` | Table of contents and section hierarchy |
| `page_maps/` | PDF page ↔ printed page numbers, used in citations |
| `tables/` | Structured tables: `table.yaml` (schema, provenance) plus `records.jsonl` and, for main-text tables, `cells.jsonl` |
| `corrections/` | Reviewed text corrections |
| `source_issues.yaml` | Known printing and text-layer problems in the sources |

Structured tables currently cover the main text (tables 19-2-1, 19-3-1 to 19-3-3 and 19-5-1 to
19-5-7), Appendix 2 (city climate classes), Appendix 3 (maximum energy use intensity and
occupancy types) and Appendix 7 (material thermal properties). Other appendices are not yet
curated.

Values are transcribed as printed, anomalies included; a suspected error is recorded in
`source_issues.yaml`, not silently fixed. Changes here alter the engine fingerprint and
invalidate frozen candidates and manifests.

This description lives in `docs/` because any file added under `curated/` changes the engine fingerprint.
