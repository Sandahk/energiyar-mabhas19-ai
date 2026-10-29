"""Record/replay data structures for proposer calls (no provider is called in SH0).

A :class:`ReplayKey` pins everything a proposal depends on - the normalized query, the
candidate context, the provider and exact model ID, the instruction text and JSON Schema, the
settings and the engine (frozen retrieval modules, curated data, shadow versions) - and its
digest names one immutable :class:`ReplayRecord`: the raw provider response, the parsed
proposal, schema errors, the verifier's field decisions, the translated plan, the counterfactual
result and token/latency metadata. Evaluation replays records only; a missing key is reported,
never fetched silently, and a record is never overwritten with different content.

The request a provider would receive (:func:`render_request`) holds the instructions, the
provider projection of the proposal schema and the redacted candidate context - no document,
corpus text, regulatory value, credential or project metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from mabhas19.shadow.context import CONTEXT_VERSION, CandidateContext, canonical_json
from mabhas19.shadow.proposal import PROPOSAL_SCHEMA, canonical_schema, canonical_schema_sha256
from mabhas19.shadow.schema_projection import PROJECTIONS, project_anthropic, project_gemini
from mabhas19.shadow.schema_projection import sha256 as projection_sha256
from mabhas19.shadow.translate import TRANSLATION_VERSION
from mabhas19.shadow.verifier import VERIFIER_VERSION
from mabhas19.text.normalize import normalize_for_search

REPLAY_SCHEMA = "sh0-replay-1"
PROMPT_VERSION = "sh0-prompt-1"
INSTRUCTIONS = """\
You label the semantic structure of one Persian question about the Iranian national building \
code, Mabhas 19. You have no authority over regulatory content: a deterministic system answers \
from printed tables and checks every label you give. Return one JSON object that matches the \
supplied JSON Schema, and nothing else.

Rules:
- Label every supplied item exactly once, echoing its index and offsets (and text where the \
schema asks for it) from the context. Never invent a span, an index or an identifier.
- clauses: lookup (asks for a printed value), verify_request (asks to check a stated claim), \
background (context the user states, not a request), explanation (asks for an explanation), \
unsupported (anything else).
- entities: identity only when the span names that printed row; grammatical when it is a \
function word or a homograph used grammatically (a preposition, a possessive, a numeral); \
contextual when it is context, not the thing asked about; ambiguous when you cannot tell. \
candidate is one of the span's supplied candidate IDs, "none" or "ambiguous".
- values: lookup_constraint (restricts the requested lookup), verify_request (a claim the user \
asks to check), background (stated context, not a constraint), excluded, corrected_away; clause \
is the index of the clause holding the value.
- polarity: keep every supplied exclusion and rejected reading (kept: true).
- operation: "none" for a plain lookup, a calculation kind when the user asks for a computed \
quantity, "ambiguous" when you cannot tell.
- requests: only the supplied tables the user actually asks about, with intent, the requested \
output name from that table's outputs, and the index of a unit expression the user asks the \
answer in (or "none").
- coordination: list every supplied member of every coordinated table.
- Never write a number, a value, a record, a citation or a name of your own. When unsure, say \
"ambiguous": an ambiguous label is always safe.
"""


ANTHROPIC = "anthropic"
GEMINI = "gemini"
PILOT_MODEL = "claude-opus-5-5"  # the approved Anthropic model; changes need explicit approval
GEMINI_PILOT_MODEL = "gemini-3.8-flash"  # the approved Gemini model (stable ID, not an alias)
APPROVED_MODELS = {ANTHROPIC: PILOT_MODEL, GEMINI: GEMINI_PILOT_MODEL}
RENDER_VERSION = "sh0-render-1"  # the user turn: canonical JSON of {render, context}


@dataclass(frozen=True)
class ProposerSettings:
    """How a proposal is requested (every field is part of the replay key)."""

    provider: str = ANTHROPIC
    model_id: str = PILOT_MODEL  # the exact requested model (never an alias)
    # the provider's reasoning-effort control: Anthropic ``output_config.effort`` (thinking is
    # the model's adaptive default), Gemini ``thinkingConfig.thinkingLevel``
    effort: str = "medium"
    structured_output: bool = True  # constrained decoding with the provider schema projection
    max_output_tokens: int = 16000  # thinking tokens included
    timeout_s: float = 120.0
    max_retries: int = 1  # retries of transport / transient API errors only

    @classmethod
    def gemini(cls) -> ProposerSettings:
        return cls(provider=GEMINI, model_id=GEMINI_PILOT_MODEL)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def provider_schema(provider: str = ANTHROPIC) -> dict[str, Any]:
    """The projection of the canonical schema a provider's constrained decoding receives."""
    if provider == ANTHROPIC:
        return project_anthropic(canonical_schema())
    if provider == GEMINI:
        return project_gemini(canonical_schema())
    raise ValueError(f"no schema projection for provider {provider!r}")


def projection_version(provider: str = ANTHROPIC) -> str:
    return PROJECTIONS[provider].version


def prompt_sha256() -> str:
    """The system instructions, the render format and the schema contract, as sent."""
    doc = {
        "prompt_version": PROMPT_VERSION,
        "instructions": INSTRUCTIONS,
        "render_version": RENDER_VERSION,
        "canonical_schema": [PROPOSAL_SCHEMA, canonical_schema_sha256()],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def render_user(ctx: CandidateContext) -> str:
    """The user turn: the (redacted) candidate context, byte-stable."""
    return canonical_json({"render": RENDER_VERSION, "context": ctx.to_dict()})


def render_request(
    ctx: CandidateContext, settings: ProposerSettings | None = None
) -> dict[str, Any]:
    """The exact request body a provider call would send (nothing else is sent: no document,
    corpus text, credential or project metadata). The instructions and the user turn are the
    same for every provider; only the request envelope differs."""
    s = settings or ProposerSettings()
    if s.provider == GEMINI:
        return _render_gemini(ctx, s)
    if s.provider != ANTHROPIC:
        raise ValueError(f"no request format for provider {s.provider!r}")
    output_config: dict[str, Any] = {"effort": s.effort}
    if s.structured_output:
        output_config["format"] = {"type": "json_schema", "schema": provider_schema(s.provider)}
    return {
        "model": s.model_id,
        "max_tokens": s.max_output_tokens,
        "system": INSTRUCTIONS,
        "messages": [{"role": "user", "content": render_user(ctx)}],
        "output_config": output_config,
    }


def _render_gemini(ctx: CandidateContext, s: ProposerSettings) -> dict[str, Any]:
    """A Gemini ``generateContent`` REST body (the model is in the URL). The same body, with
    the model name added, is what ``countTokens`` measures as its ``generateContentRequest``.
    No ``temperature`` / ``topP`` / ``topK`` / ``seed`` and no safety-setting override."""
    config: dict[str, Any] = {
        "maxOutputTokens": s.max_output_tokens,
        "thinkingConfig": {"thinkingLevel": s.effort.upper()},
    }
    if s.structured_output:
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = provider_schema(GEMINI)
    return {
        "systemInstruction": {"parts": [{"text": INSTRUCTIONS}]},
        "contents": [{"role": "user", "parts": [{"text": render_user(ctx)}]}],
        "generationConfig": config,
    }


def user_turn(body: dict[str, Any]) -> str:
    """The rendered user turn of a request body, whatever the provider envelope."""
    if "messages" in body:
        return str(body["messages"][0]["content"])
    return str(body["contents"][0]["parts"][0]["text"])


def query_sha256(query: str) -> str:
    return hashlib.sha256(normalize_for_search(query).encode("utf-8")).hexdigest()


@cache
def engine_fingerprint() -> str:
    """The frozen deterministic engine and the shadow layer: every pinned retrieval module, the
    curated data tree and the shadow versions (any change makes new keys)."""
    from mabhas19.core.paths import CURATED_DIR, REPO_ROOT  # noqa: PLC0415
    from mabhas19.retrieval import candidate  # noqa: PLC0415

    shadow = sorted((REPO_ROOT / "src" / "mabhas19" / "shadow").glob("*.py"))
    doc = {
        "modules": {m: candidate._sha(REPO_ROOT / m) for m in candidate.MODULES},
        "shadow": {p.name: candidate._sha(p) for p in shadow},
        "curated": candidate._tree_digest(CURATED_DIR),
        "versions": [CONTEXT_VERSION, PROPOSAL_SCHEMA, VERIFIER_VERSION, TRANSLATION_VERSION],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplayKey:
    query_sha256: str
    context_sha256: str
    provider: str
    model_id: str
    prompt_sha256: str
    schema_version: str  # the canonical contract
    canonical_schema_sha256: str
    provider_schema_sha256: str
    projection_version: str
    settings: tuple[tuple[str, Any], ...]  # sorted (name, value) pairs
    engine: str

    @classmethod
    def of(
        cls, query: str, ctx: CandidateContext, settings: ProposerSettings, engine: str
    ) -> ReplayKey:
        rest = {k: v for k, v in settings.to_dict().items() if k not in ("provider", "model_id")}
        return cls(
            query_sha256=query_sha256(query),
            context_sha256=ctx.sha256(),
            provider=settings.provider,
            model_id=settings.model_id,
            prompt_sha256=prompt_sha256(),
            schema_version=PROPOSAL_SCHEMA,
            canonical_schema_sha256=canonical_schema_sha256(),
            provider_schema_sha256=projection_sha256(provider_schema(settings.provider)),
            projection_version=projection_version(settings.provider),
            settings=tuple(sorted(rest.items())),
            engine=engine,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["settings"] = dict(self.settings)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReplayKey:
        return cls(**{**d, "settings": tuple(sorted(d["settings"].items()))})

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass
class ReplayRecord:
    key: ReplayKey
    timestamp: str  # ISO 8601 UTC, given by the recorder
    raw_response: str | None  # exactly as returned (None: no response)
    parsed_proposal: dict[str, Any] | None
    schema_errors: list[str] = field(default_factory=list)
    verifier: dict[str, Any] = field(default_factory=dict)  # Verification.to_dict()
    translated_plan: dict[str, Any] | None = None
    counterfactual: dict[str, Any] | None = None
    # input_tokens, output_tokens, latency_ms, stop_reason, request_id, error (never a secret)
    usage: dict[str, Any] = field(default_factory=dict)
    response_model: str | None = None  # the model the API reports having run
    # False when the record cannot count in the experiment (another model ran, no response…)
    experiment_valid: bool = True
    invalid_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key.to_dict()
        d["key_digest"] = self.key.digest()
        d["replay_schema"] = REPLAY_SCHEMA
        return d

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> ReplayRecord:
        d = json.loads(text)
        if d.pop("replay_schema", None) != REPLAY_SCHEMA:
            raise ValueError("not a sh0 replay record")
        digest = d.pop("key_digest")
        key = ReplayKey.from_dict(d.pop("key"))
        if key.digest() != digest:
            raise ValueError("replay record key digest mismatch")
        return cls(key=key, **d)


class ReplayConflictError(RuntimeError):
    pass


class ReplayStore:
    """One immutable JSON file per key digest (``<root>/<2 hex>/<digest>.json``)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: ReplayKey) -> Path:
        digest = key.digest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: ReplayKey) -> ReplayRecord | None:
        path = self.path(key)
        if not path.is_file():
            return None
        return ReplayRecord.from_json(path.read_text(encoding="utf-8"))

    def put(self, record: ReplayRecord) -> Path:
        path = self.path(record.key)
        text = record.to_json()
        if path.is_file():
            if path.read_text(encoding="utf-8") != text:
                raise ReplayConflictError(f"{path.name} exists with different content")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def missing(self, keys: list[ReplayKey]) -> list[ReplayKey]:
        """Keys with no record (reported by a replay run, never fetched)."""
        return [k for k in keys if not self.path(k).is_file()]
