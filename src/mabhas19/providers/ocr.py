"""OCR engine interface and adapters (offline engines only)."""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_TESSERACT_WINDOWS = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float  # 0-100, engine-reported
    x0: float
    y0: float
    x1: float
    y1: float  # pixel coordinates in the input image


@dataclass(frozen=True)
class OcrResult:
    text: str
    words: list[OcrWord] = field(default_factory=list)
    seconds: float = 0.0


class OcrEngine(Protocol):
    name: str

    def availability(self) -> tuple[bool, str]:
        """(available, reason-if-not)."""
        ...

    def recognize(self, image: Path, languages: tuple[str, ...], psm: int = 6) -> OcrResult: ...


class TesseractCli:
    """Tesseract via its command line (no Python binding needed)."""

    name = "tesseract"

    def __init__(self, executable: Path | None = None, tessdata_dir: Path | None = None) -> None:
        found = shutil.which("tesseract")
        env = os.environ.get("TESSERACT_CMD")
        self.executable = executable or (
            Path(env) if env else Path(found) if found else DEFAULT_TESSERACT_WINDOWS
        )
        self.tessdata_dir = tessdata_dir

    def _base_cmd(self) -> list[str]:
        cmd = [str(self.executable)]
        if self.tessdata_dir is not None:
            cmd += ["--tessdata-dir", str(self.tessdata_dir)]
        return cmd

    def availability(self) -> tuple[bool, str]:
        if not self.executable.exists():
            return False, f"tesseract not found at {self.executable}"
        out = subprocess.run(
            [*self._base_cmd(), "--list-langs"], capture_output=True, text=True, check=False
        )
        langs = set(out.stdout.split()[1:]) if out.returncode == 0 else set()
        if "fas" not in langs:
            return False, "Persian traineddata (fas) not available"
        return True, ""

    def version(self) -> str:
        out = subprocess.run(
            [str(self.executable), "--version"], capture_output=True, text=True, check=False
        )
        return (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else ""

    def recognize(self, image: Path, languages: tuple[str, ...], psm: int = 6) -> OcrResult:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "out"
            cmd = [*self._base_cmd(), str(image), str(base), "-l", "+".join(languages),
                   "--psm", str(psm), "-c", "tessedit_create_txt=1",
                   "-c", "tessedit_create_tsv=1"]  # fmt: skip
            subprocess.run(cmd, capture_output=True, check=True)
            text = base.with_suffix(".txt").read_text(encoding="utf-8")
            tsv = base.with_suffix(".tsv").read_text(encoding="utf-8")
        words = []
        for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
            if row.get("level") == "5" and (row.get("text") or "").strip():
                left, top = float(row["left"]), float(row["top"])
                words.append(
                    OcrWord(row["text"], float(row["conf"]), left, top,
                            left + float(row["width"]), top + float(row["height"]))
                )  # fmt: skip
        return OcrResult(text, words, time.perf_counter() - start)


class EasyOcrEngine:
    """EasyOCR (optional dependency group ``ocr-easyocr``; needs PyTorch)."""

    name = "easyocr"

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir
        self._reader: Any = None

    def availability(self) -> tuple[bool, str]:
        try:
            import easyocr  # noqa: F401, PLC0415
        except ImportError:
            return False, "easyocr not installed (uv sync --group ocr-easyocr)"
        except OSError as exc:  # e.g. Windows Application Control blocking torch DLLs
            return False, f"easyocr/torch failed to load: {exc}"
        return True, ""

    def recognize(self, image: Path, languages: tuple[str, ...], psm: int = 6) -> OcrResult:
        import easyocr  # noqa: PLC0415

        start = time.perf_counter()
        langs = ["fa" if lang == "fas" else "en" if lang == "eng" else lang for lang in languages]
        if self._reader is None:
            self._reader = easyocr.Reader(
                langs, gpu=False, model_storage_directory=str(self.model_dir or "")
            )
        results = self._reader.readtext(str(image))
        words = [
            OcrWord(text, float(conf) * 100, box[0][0], box[0][1], box[2][0], box[2][1])
            for box, text, conf in results
        ]
        return OcrResult("\n".join(w.text for w in words), words, time.perf_counter() - start)
