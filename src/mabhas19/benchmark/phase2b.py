"""Phase 2B: experimental Gemini vision benchmark on the fixed benchmark pages only.

Ground truth: only reference items whose status (item or page) is ``human_verified`` count
towards authoritative scores; draft items are scored separately and never drive
recommendations. Numbers are always re-parsed from the model's *raw* transcription with the
project's own non-guessing parser; the model's own parsed values are scored separately.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pymupdf

from mabhas19.benchmark import extractors as ex
from mabhas19.benchmark.metrics import label_match, score_text
from mabhas19.benchmark.references import VERIFIED, BenchmarkPage, PageReference, ReferenceItem
from mabhas19.providers.ocr import OcrEngine
from mabhas19.providers.vision import NUMERIC_RULE, TEXT_PROMPT, VisionResult
from mabhas19.tables.schemas import DensitySpec, NumericField
from mabhas19.text.normalize import fold_hyphens, to_ascii_digits
from mabhas19.text.numbers import parse_numeric, same_value

PAGE_DPI = 200
CROP_DPI = 300
MIN_INTERVAL_S = 7.0  # stay well under free-tier per-minute limits
MAX_ATTEMPTS = 4
SERVER_BACKOFF_S = (20, 60, 120)
MAX_RETRY_DELAY_S = 120
ERROR_MESSAGE_CHARS = 3000  # long enough to keep quota ids and limits from 429 responses
DECIMAL_MARKS = frozenset({".", "٫", "/", ",", "،"})

APP07_ROW_FIELDS = (
    "category_raw", "subgroup_raw", "material_raw", "condition_raw", "density_raw",
    "conductivity_raw", "density_min_parsed", "density_max_parsed", "conductivity_parsed",
)  # fmt: skip
APP07_PAGE_PROMPT = (
    "This page is part of a table of building materials (columns, right to left: material "
    "description; dry density [kg/m3]; effective thermal conductivity [W/m.K]). Return JSON "
    '{"rows": [...]} with ONE entry per printed row that has a conductivity value, in '
    "top-to-bottom order. Each entry has exactly these keys: " + ", ".join(APP07_ROW_FIELDS) + ". "
    "*_raw fields: the text exactly as printed in that row (Persian digits, the decimal "
    "separator and words like «تا», «بیش از», «کمتر از» kept), or null if nothing is printed "
    "in that cell of that row (e.g. a continuation row without its own material text). "
    "category_raw/subgroup_raw: the nearest numbered group heading / unnumbered heading "
    "above the row, as printed. *_parsed fields: ASCII decimal strings using '.', or null "
    "when you are not certain. " + NUMERIC_RULE
)
APP07_CROP_PROMPT = (
    "This image is ONE row cut from a table of building materials (columns, right to left: "
    "material description; dry density [kg/m3]; effective thermal conductivity [W/m.K]). "
    "Return JSON with exactly these keys: material_raw, density_raw, conductivity_raw, "
    "density_min_parsed, density_max_parsed, conductivity_parsed. *_raw: exactly as "
    "printed (Persian digits, decimal separator, «تا», «بیش از», «کمتر از»), null if empty. "
    "*_parsed: ASCII decimal strings using '.', or null when not certain. " + NUMERIC_RULE
)
APP02_PROMPT = (
    "This page holds a table of Iranian cities and climate classes (columns, right to left: "
    'row number, city name, climate class). Return JSON {"rows": [{"row_number_raw": ..., '
    '"city_raw": ..., "climate_class_raw": ...}]} for every table row, top to bottom, '
    "copied exactly as printed. Do not normalise or correct any climate class. " + NUMERIC_RULE
)
APP03_PROMPT = (
    "This page holds a table of maximum energy-use intensity values (kWh/m2.yr). Return "
    'JSON {"table_caption_raw": ..., "energy_rating_raw": ..., "rows": [{"row_number_raw": '
    '..., "occupancy_type_raw": ..., "values_raw": {<climate code exactly as printed in the '
    "header>: <cell text exactly as printed>}}]} for every data row, top to bottom. " + NUMERIC_RULE
)


class AskingVision(Protocol):
    model: str

    def ask(self, image: Path, prompt: str, json_mode: bool = False) -> VisionResult: ...


@dataclass
class Request:
    id: str  # stable cache key
    kind: str  # "text" | "app07_page" | "app07_crop" | "app02" | "app03"
    page_id: str
    doc_id: str
    pdf_page: int
    prompt: str
    json_mode: bool
    clip: tuple[float, float, float, float] | None = None
    item_id: str | None = None


@dataclass
class Response:
    request_id: str
    kind: str
    page_id: str
    item_id: str | None
    model_requested: str
    status: str  # "ok" | "error" | "skipped_quota"
    model_version: str | None = None
    latency_s: float | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    text: str | None = None
    received_at: str | None = None  # UTC ISO time of the successful answer (newer runs only)


# --------------------------------------------------------------------------- requests


def is_verified(item: ReferenceItem, page: PageReference) -> bool:
    return item.status == VERIFIED or (item.status is None and page.status == VERIFIED)


def build_requests(pages: list[BenchmarkPage], refs: dict[str, PageReference]) -> list[Request]:
    """All Phase 2B requests in priority order (appendix 7 first)."""
    by_doc = {p.id: p for p in pages}
    out: list[Request] = []
    app07 = [p for p in pages if p.category == "table_materials"]
    for p in app07:
        out.append(Request(f"{p.id}.page", "app07_page", p.id, p.doc_id, p.pdf_page,
                           APP07_PAGE_PROMPT, True))  # fmt: skip
    for p in app07:
        for item in refs[p.id].items:
            if item.type == "material_row" and item.bbox is not None:
                _, y0, _, y1 = item.bbox
                clip = (115.0, y0 - 3, 545.0, y1 + 3)
                out.append(Request(f"{item.id}.crop", "app07_crop", p.id, p.doc_id, p.pdf_page,
                                   APP07_CROP_PROMPT, True, clip, item.id))  # fmt: skip
    for p in pages:
        if p.category == "table_city_climate":
            out.append(
                Request(f"{p.id}.app02", "app02", p.id, p.doc_id, p.pdf_page, APP02_PROMPT, True)
            )
        if p.category == "table_energy_limits":
            out.append(
                Request(f"{p.id}.app03", "app03", p.id, p.doc_id, p.pdf_page, APP03_PROMPT, True)
            )
    for p in pages:
        out.append(Request(f"{p.id}.text", "text", p.id, p.doc_id, p.pdf_page, TEXT_PROMPT, False))
    assert len({r.id for r in out}) == len(out)
    del by_doc
    return out


# --------------------------------------------------------------------------- calling


def scrub(text: str) -> str:
    """Remove the API key (if it ever appears) from anything that will be stored."""
    key = os.environ.get("GEMINI_API_KEY")
    return text.replace(key, "[REDACTED]") if key else text


def classify_error(exc: Exception) -> tuple[str, float | None]:
    """('rate_limit' | 'quota_daily' | 'server' | 'client', retry-after seconds or None)."""
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        m = re.search(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s", msg)
        daily = bool(re.search(r"PerDay|per_day|PerProjectPerDay", msg))
        return ("quota_daily" if daily else "rate_limit"), (float(m.group(1)) if m else None)
    if any(code in msg for code in ("500", "502", "503", "504", "UNAVAILABLE", "INTERNAL")):
        return "server", None
    if (
        isinstance(exc, TimeoutError)
        or "timed out" in msg.lower()
        or "Timeout" in type(exc).__name__
    ):
        return "server", None
    return "client", None


@dataclass
class Caller:
    vision: AskingVision
    cache_dir: Path
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    min_interval_s: float = MIN_INTERVAL_S
    _last: float = field(default=-1e9, init=False)
    quota_exhausted: bool = field(default=False, init=False)
    requests_sent: int = field(default=0, init=False)

    def _cache_path(self, req: Request) -> Path:
        return self.cache_dir / f"{req.id}.json"

    def cached(self, req: Request) -> Response | None:
        path = self._cache_path(req)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") == "ok":
                return Response(**data)
        return None

    def call(self, req: Request, image: Path) -> Response:
        hit = self.cached(req)
        if hit is not None:
            return hit
        resp = Response(req.id, req.kind, req.page_id, req.item_id, self.vision.model, "error")
        if self.quota_exhausted:
            resp.status = "skipped_quota"
            return self._store(req, resp)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            wait = self.min_interval_s - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = self.clock()
            self.requests_sent += 1
            try:
                result = self.vision.ask(image, req.prompt, req.json_mode)
            except Exception as exc:  # recorded per attempt
                kind, delay = classify_error(exc)
                message = scrub(str(exc))[:ERROR_MESSAGE_CHARS]
                resp.attempts.append({"attempt": attempt, "error": kind, "message": message})
                if kind == "quota_daily":
                    self.quota_exhausted = True
                    break
                if kind == "client" or attempt == MAX_ATTEMPTS:
                    break
                if kind == "rate_limit":
                    self.sleep(min((delay or 30.0) + 2, MAX_RETRY_DELAY_S))
                else:
                    self.sleep(SERVER_BACKOFF_S[min(attempt - 1, len(SERVER_BACKOFF_S) - 1)])
                continue
            resp.status = "ok"
            resp.model_version = result.model
            resp.latency_s = round(result.seconds, 2)
            resp.usage = dict(result.usage)
            resp.text = scrub(result.text)
            resp.received_at = datetime.now(UTC).isoformat(timespec="seconds")
            resp.attempts.append({"attempt": attempt, "error": None})
            break
        return self._store(req, resp)

    def _store(self, req: Request, resp: Response) -> Response:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(req).write_text(
            json.dumps(asdict(resp), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return resp


def render_request(req: Request, sources_dir: Path, out: Path) -> Path:
    kind = "main" if req.doc_id.endswith("-main") else "appendices"
    pdf = pymupdf.open(sources_dir / "mabhas19" / "ed5-1404" / kind / f"{req.doc_id}-v1.pdf")
    page = pdf[req.pdf_page - 1]
    clip = pymupdf.Rect(req.clip) if req.clip else None
    path = ex.render_clean(page, out / f"{req.id}.png", clip=clip,
                           dpi=CROP_DPI if clip else PAGE_DPI)  # fmt: skip
    pdf.close()
    return path


# --------------------------------------------------------------------------- parsing


def parse_json(text: str | None) -> Any:
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def schema_ok(kind: str, data: Any) -> bool:
    """Minimal structural validity of a model's JSON answer."""
    if kind == "app07_crop":
        return isinstance(data, dict) and {"density_raw", "conductivity_raw"} <= set(data)
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return False
    rows = data["rows"]
    need = {"app07_page": {"density_raw", "conductivity_raw"},
            "app02": {"city_raw", "climate_class_raw"},
            "app03": {"occupancy_type_raw", "values_raw"}}[kind]  # fmt: skip
    return all(isinstance(r, dict) and need <= set(r) for r in rows)


def _s(value: Any) -> str | None:
    return None if value is None else str(value)


# --------------------------------------------------------------------------- app07 rows


@dataclass(frozen=True)
class ExtractedRow:
    """One extracted appendix 7 row, whatever the method."""

    density_raw: str | None
    conductivity_raw: str | None
    material_raw: str | None = None
    density_min_parsed: str | None = None  # model-provided (not trusted)
    density_max_parsed: str | None = None
    conductivity_parsed: str | None = None


def rows_from_json(data: Any) -> list[ExtractedRow]:
    items = data.get("rows", []) if isinstance(data, dict) else []
    return [ExtractedRow(_s(r.get("density_raw")), _s(r.get("conductivity_raw")),
                         _s(r.get("material_raw")), _s(r.get("density_min_parsed")),
                         _s(r.get("density_max_parsed")), _s(r.get("conductivity_parsed")))
            for r in items if isinstance(r, dict)]  # fmt: skip


def _density_key(text: str | None) -> str:
    return re.sub(r"\s+", "", fold_hyphens(to_ascii_digits(text or "")))


def align_rows(refs: list[ReferenceItem], rows: list[ExtractedRow]) -> dict[str, ExtractedRow]:
    """Order-preserving alignment of extracted rows to reference rows (tolerates skips).

    Score: density text equal (2) + conductivity value equal (1). Unmatched rows stay out.
    """
    n, m = len(refs), len(rows)

    def sim(r: ReferenceItem, x: ExtractedRow) -> int:
        f = r.expected_fields
        score = 0
        if _density_key(r.expected_text) == _density_key(x.density_raw):
            score += 2
        k = parse_numeric(x.conductivity_raw or "", slash_is_decimal=True).normalized_value
        if same_value(f.get("conductivity"), k):
            score += 1
        return score

    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            s = sim(refs[i], rows[j])
            take = best[i + 1][j + 1] + s if s > 0 else -1
            best[i][j] = max(take, best[i + 1][j], best[i][j + 1])
    out: dict[str, ExtractedRow] = {}
    i = j = 0
    while i < n and j < m:
        s = sim(refs[i], rows[j])
        if s > 0 and best[i][j] == best[i + 1][j + 1] + s:
            out[refs[i].id] = rows[j]
            i += 1
            j += 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return out


def decimal_separator_ok(expected: str, raw: str | None) -> bool | None:
    """For an expected decimal value: is exactly one separator present with the right
    number of digits after it? None when the expected value has no decimals."""
    if "." not in expected:
        return None
    if not raw:
        return False
    body = to_ascii_digits("".join(c for c in raw if not c.isspace()))
    marks = [i for i, c in enumerate(body) if c in DECIMAL_MARKS]
    if len(marks) != 1:
        return False
    after = body[marks[0] + 1 :]
    return after.isdigit() and len(after) == len(expected.split(".")[1])


@dataclass
class NumericReport:
    method: str
    total: int = 0
    exact: int = 0
    model_parsed_total: int = 0
    model_parsed_exact: int = 0
    decimal_total: int = 0
    decimal_ok: int = 0
    rows_total: int = 0
    rows_complete: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def rate(self, a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "numeric_exact": f"{self.exact}/{self.total}",
            "numeric_accuracy": self.rate(self.exact, self.total),
            "model_parsed_accuracy": self.rate(self.model_parsed_exact, self.model_parsed_total),
            "decimal_separator_accuracy": self.rate(self.decimal_ok, self.decimal_total),
            "row_completeness": self.rate(self.rows_complete, self.rows_total),
            "rows": f"{self.rows_complete}/{self.rows_total}",
        }


def _error(item: str, name: str, method: str, seen: tuple[str | None, str | None, str | None]
           ) -> dict[str, Any]:  # fmt: skip
    """seen = (source value, extracted raw text, extracted parsed value)."""
    source, raw, value = seen
    return {"item": item, "field": name, "source": source, "extracted_raw": raw,
            "extracted": value, "method": method}  # fmt: skip


def score_app07(
    method: str, refs: list[ReferenceItem], found: dict[str, ExtractedRow]
) -> NumericReport:
    """Exact numeric accuracy of density bounds and conductivity per verified reference row."""
    rep = NumericReport(method)
    for ref in refs:
        f = ref.expected_fields
        row = found.get(ref.id)
        dens: DensitySpec | None = ex.parse_density(row.density_raw or "", "high") if row else None
        k = parse_numeric(row.conductivity_raw or "", slash_is_decimal=True) if row else None
        checks = [("conductivity", f["conductivity"], k.normalized_value if k else None,
                   row.conductivity_raw if row else None,
                   row.conductivity_parsed if row else None)]  # fmt: skip
        for bound in ("min", "max"):
            got: NumericField | None = getattr(dens, bound) if dens else None
            parsed = getattr(row, f"density_{bound}_parsed") if row else None
            checks.append((f"density_{bound}", f[f"density_{bound}"],
                           got.normalized_value if got else None,
                           row.density_raw if row else None, parsed))  # fmt: skip
        complete = True
        for name, expected, value, raw, model_parsed in checks:
            if expected is None:
                ok = value is None
                if not ok:
                    complete = False
                    rep.errors.append(_error(ref.id, name, method, (None, raw, value)))
                continue
            rep.total += 1
            ok = same_value(expected, value)
            rep.exact += ok
            if model_parsed is not None or row is not None:
                rep.model_parsed_total += 1
                rep.model_parsed_exact += same_value(expected, model_parsed)
            if name == "conductivity":
                sep = decimal_separator_ok(expected, raw)
                if sep is not None:
                    rep.decimal_total += 1
                    rep.decimal_ok += sep
            if not ok:
                complete = False
                rep.errors.append(_error(ref.id, name, method, (expected, raw, value)))
        rep.rows_total += 1
        rep.rows_complete += complete
    return rep


# ------------------------------------------------------------- tesseract comparisons


def tesseract_page_rows(page: pymupdf.Page, engine: OcrEngine, workdir: Path,
                        refs: list[ReferenceItem]) -> dict[str, ExtractedRow]:  # fmt: skip
    """Whole-page Tesseract; words assigned to reference rows/columns by position."""
    img = ex.render_clean(page, workdir / f"tess-page-{page.number + 1}.png")
    res = engine.recognize(img, ex.LANGS, psm=6)
    scale = 72 / ex.OCR_DPI
    words = [(w.text, (w.x0 + w.x1) / 2 * scale, (w.y0 + w.y1) / 2 * scale) for w in res.words]
    cx0, cx1 = ex.APP07_COLUMNS["conductivity"]
    dx0, dx1 = ex.APP07_COLUMNS["density"]
    out = {}
    for ref in refs:
        assert ref.bbox is not None
        yc = (ref.bbox[1] + ref.bbox[3]) / 2
        near = [w for w in words if abs(w[2] - yc) <= 7]
        cond = " ".join(t for t, x, _ in sorted(near, key=lambda w: -w[1]) if cx0 < x < cx1)
        dens = " ".join(t for t, x, _ in sorted(near, key=lambda w: -w[1]) if dx0 < x < dx1)
        out[ref.id] = ExtractedRow(dens or None, cond or None)
    return out


def tesseract_cell_rows(page: pymupdf.Page, engine: OcrEngine, doc_id: str,
                        refs: list[ReferenceItem]) -> dict[str, ExtractedRow]:  # fmt: skip
    """Per-cell Tesseract (rows located from vector glyphs), matched by verified bbox."""
    recs = ex.app07_rows_ocr(page, engine, doc_id)
    out = {}
    for ref in refs:
        assert ref.bbox is not None
        yc = (ref.bbox[1] + ref.bbox[3]) / 2
        rec = min(recs, key=lambda r: abs((r.source.row_y or 0) - yc), default=None)
        if rec is None or abs((rec.source.row_y or 0) - yc) > 8:
            continue
        dens = rec.density.raw if rec.density else None
        k = rec.thermal_conductivity.raw_detected_value if rec.thermal_conductivity else None
        out[ref.id] = ExtractedRow(dens, k)
    return out


# ------------------------------------------------------------------ other scoring


def score_text_requests(resp: Response, ref: PageReference) -> dict[str, Any]:
    """CER of a plain transcription, split into verified and draft items."""
    groups: dict[str, list[dict[str, Any]]] = {"verified": [], "draft": []}
    for item in ref.items:
        s = asdict(score_text(item.expected_text, resp.text or ""))
        s["item"] = item.id
        groups["verified" if is_verified(item, ref) else "draft"].append(s)
    out: dict[str, Any] = {}
    for name, scores in groups.items():
        if scores:
            out[name] = {
                "items": len(scores),
                "mean_cer_normalized": round(
                    statistics.mean(s["cer_normalized"] for s in scores), 4
                ),
                "exact_normalized": sum(s["cer_normalized"] == 0 for s in scores),
            }
    return out


def score_app02(data: Any, ref: PageReference) -> dict[str, Any]:
    rows = data.get("rows", []) if isinstance(data, dict) else []
    by_num = {}
    for r in rows:
        n = to_ascii_digits(str(r.get("row_number_raw") or "")).strip()
        if n.isdigit():
            by_num[int(n)] = r
    city = climate = complete = total = 0
    for item in ref.items:
        f = item.expected_fields
        r = by_num.get(f["row_number"])
        c_ok = r is not None and label_match(f["city"], _s(r.get("city_raw")))
        k_ok = r is not None and str(r.get("climate_class_raw") or "").strip() == f["climate_class"]
        total += 1
        city += c_ok
        climate += k_ok
        complete += c_ok and k_ok
    return {"authoritative": False, "rows": total, "city_accuracy": round(city / total, 4),
            "climate_accuracy": round(climate / total, 4),
            "row_completeness": round(complete / total, 4)}  # fmt: skip


def score_app03(data: Any, ref: PageReference) -> dict[str, Any]:
    rows = data.get("rows", []) if isinstance(data, dict) else []
    by_num = {}
    for r in rows:
        n = to_ascii_digits(str(r.get("row_number_raw") or "")).strip()
        if n.isdigit():
            by_num[int(n)] = r
    num_total = num_ok = lab_total = lab_ok = complete = 0
    for item in ref.items:
        f = item.expected_fields
        r = by_num.get(f["row_number"]) or {}
        values = r.get("values_raw") or {}
        row_ok = True
        for climate, expected in f["values"].items():
            raw = values.get(climate)
            ok = same_value(expected, parse_numeric(str(raw or "")).normalized_value)
            num_total += 1
            num_ok += ok
            row_ok &= ok
        if f.get("label_check", True):
            ok = label_match(f["occupancy_type"], _s(r.get("occupancy_type_raw")))
            lab_total += 1
            lab_ok += ok
            row_ok &= ok
        complete += row_ok
    row_count = len(ref.items)
    return {"authoritative": False, "numeric_accuracy": round(num_ok / num_total, 4),
            "numeric_exact": f"{num_ok}/{num_total}",
            "occupancy_label_accuracy": round(lab_ok / lab_total, 4) if lab_total else None,
            "row_completeness": round(complete / row_count, 4)}  # fmt: skip
