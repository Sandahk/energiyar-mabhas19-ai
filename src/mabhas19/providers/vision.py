"""Vision-model extraction interface and adapters.

External APIs are never called implicitly: an adapter reports itself unavailable unless
its credentials exist AND the caller passed ``allow_external=True`` (an explicit,
user-approved opt-in). Each provider SDK lives in an optional dependency group and is
imported lazily.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

TEXT_PROMPT = (
    "Transcribe all text on this page of an official Persian building regulation exactly "
    "as printed. Keep Persian digits as printed. Do not correct, translate, summarise or "
    "complete anything; write [illegible] for anything you cannot read."
)
JSON_PROMPT = (
    "Extract the table on this page as JSON matching this schema. Copy labels and numbers "
    "exactly as printed (keep the printed digits and decimal separators in the *_raw "
    "fields). Never infer missing values: use null. Schema:\n{schema}"
)


@dataclass(frozen=True)
class VisionResult:
    text: str
    seconds: float
    model: str


class VisionExtractor(Protocol):
    name: str

    def availability(self) -> tuple[bool, str]: ...

    def extract_text(self, image: Path) -> VisionResult: ...

    def extract_json(self, image: Path, schema: dict[str, Any]) -> VisionResult: ...


def _png_b64(image: Path) -> str:
    return base64.standard_b64encode(image.read_bytes()).decode("ascii")


class AnthropicVision:
    """Claude via the official ``anthropic`` SDK (optional group ``vision-anthropic``)."""

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", allow_external: bool = False) -> None:
        self.model = model
        self.allow_external = allow_external

    def availability(self) -> tuple[bool, str]:
        if not self.allow_external:
            return False, "external vision API not authorized for this run"
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False, "no Anthropic credentials in the environment"
        try:
            import anthropic  # noqa: F401, PLC0415
        except ImportError:
            return False, "anthropic SDK not installed (uv sync --group vision-anthropic)"
        return True, ""

    def _ask(self, image: Path, prompt: str) -> VisionResult:
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic()
        start = time.perf_counter()
        response = client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(image)}},
                {"type": "text", "text": prompt},
            ]}],
        )  # fmt: skip
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model declined: {response.stop_details}")
        text = "".join(b.text for b in response.content if b.type == "text")
        return VisionResult(text, time.perf_counter() - start, response.model)

    def extract_text(self, image: Path) -> VisionResult:
        return self._ask(image, TEXT_PROMPT)

    def extract_json(self, image: Path, schema: dict[str, Any]) -> VisionResult:
        return self._ask(image, JSON_PROMPT.format(schema=json.dumps(schema, ensure_ascii=False)))


class GeminiVision:
    """Gemini via the ``google-genai`` SDK (optional group ``vision-gemini``)."""

    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-pro", allow_external: bool = False) -> None:
        self.model = model
        self.allow_external = allow_external

    def availability(self) -> tuple[bool, str]:
        if not self.allow_external:
            return False, "external vision API not authorized for this run"
        if not os.environ.get("GEMINI_API_KEY"):
            return False, "no GEMINI_API_KEY in the environment"
        try:
            installed = importlib.util.find_spec("google.genai") is not None
        except ModuleNotFoundError:
            installed = False
        if not installed:
            return False, "google-genai SDK not installed (uv sync --group vision-gemini)"
        return True, ""

    def _ask(self, image: Path, prompt: str) -> VisionResult:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client()
        start = time.perf_counter()
        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image.read_bytes(), mime_type="image/png"),
                prompt,
            ],
        )
        return VisionResult(response.text or "", time.perf_counter() - start, self.model)

    def extract_text(self, image: Path) -> VisionResult:
        return self._ask(image, TEXT_PROMPT)

    def extract_json(self, image: Path, schema: dict[str, Any]) -> VisionResult:
        return self._ask(image, JSON_PROMPT.format(schema=json.dumps(schema, ensure_ascii=False)))
