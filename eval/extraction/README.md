# Extraction benchmark references

One YAML file per benchmark page (`<page_id>.yaml`, page ids from
`configs/extraction-benchmark.yaml`). Each file:

```yaml
page_id: app02-p002
doc_id: m19-ed5-1404-app02
pdf_page: 2
status: draft_pending_human_review   # or human_verified
transcribed_by: "..."
items:
  - id: app02-p002-r05
    doc_id: m19-ed5-1404-app02        # repeated on every item
    pdf_page: 2
    type: table_row                   # see ITEM_TYPES in src/mabhas19/benchmark/references.py
    bbox: [x0, y0, x1, y1]            # optional, PDF points
    expected_text: "اردبیل 5C"        # as printed
    expected_fields: {...}            # optional, structured ground truth
    notes: "..."                      # optional
```

Rules: transcribe **as printed** (typos and anomalies included; see
`curated/source_issues.yaml`), Persian digits where the page shows Persian digits, numbers
in `expected_fields` as ASCII decimal strings. All current files are
`draft_pending_human_review`: transcribed by Claude from page renders, with numeric cells
cross-checked against the text layer where one exists (appendix 2 and 3). A human must
check them before they are marked `human_verified`.

Run: `uv run python -m mabhas19.cli.extraction_benchmark` (outputs in `eval/runs/`, not
committed).
