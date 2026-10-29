"""The text-only SH0 proposers (Anthropic, Gemini) and the external-call guards.

Both adapters send the same pinned instructions and the same rendered candidate context; only
the request envelope and the schema projection differ (:func:`replay.render_request`). Every
response is parsed by the canonical parser and checked by the verifier, whatever the provider.

**Anthropic** - one Messages API request per question: the pinned instructions as the system
prompt, the redacted candidate context as the user turn, structured output with the provider
projection of the proposal schema, ``output_config.effort`` and adaptive thinking. No
``temperature`` (the model rejects sampling parameters), no server-side model fallback, no beta.
The SDK retries transport and transient API errors once (``max_retries=1``); a returned
proposal is never regenerated - a schema-invalid or verifier-rejected proposal is a result.

The actual ``response.model`` is recorded; a response from any model other than the requested
one is marked invalid for the experiment.

**Gemini** - one ``generateContent`` REST request per question (``v1beta``, API key in the
``x-goog-api-key`` header, never in the URL): the instructions as ``systemInstruction``, the
context as the user turn, ``responseJsonSchema`` with the Gemini projection,
``thinkingConfig.thinkingLevel``; no sampling parameter, no safety-setting override, no model
alias. ``countTokens`` measures the identical body as its ``generateContentRequest`` (the SDK
cannot: in Developer-API mode it refuses a system instruction or generation config). One retry
of transport errors and 429 / 5xx answers, recorded in the usage metadata (attempts, whether a
retry occurred, the first error's category) of the one record; an HTTP 400 rejects the request
shape itself and stops the pilot (:class:`ExperimentConfigError`); anything else is recorded as
the result. The reported ``modelVersion`` must equal the requested model.

No call is possible unless every guard holds (:func:`authorize`): ``allow_external`` explicitly
given, the provider's credential present in the environment (never read, printed or stored
beyond a presence check, except that the Gemini adapter puts its key in the request header), a
frozen pilot manifest verified against the live engine, and a dollar and token budget, which
:class:`BudgetMeter` enforces before every call. A model whose pinned price has expired is
refused until the price is re-pinned.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from mabhas19.shadow.replay import (
    ANTHROPIC,
    APPROVED_MODELS,
    GEMINI,
    GEMINI_PILOT_MODEL,
    PILOT_MODEL,
    ProposerSettings,
    render_request,
)

if TYPE_CHECKING:
    from mabhas19.shadow.context import CandidateContext

PROVIDER_VERSION = "sh0-anthropic-proposer-1"
GEMINI_PROVIDER_VERSION = "sh0-gemini-proposer-1"
# list prices, $ per million tokens (input, output); thinking bills as output for both.
# Gemini 3.8 Flash standard tier: $0.75 / $3.75 through 2026-12-31 ($1.50 / $7.50 from
# 2027-01-01), ai.google.dev/gemini-api/docs/pricing, read 2026-10-29.
PRICES = {PILOT_MODEL: (4.00, 20.00), GEMINI_PILOT_MODEL: (0.75, 3.75)}
PRICE_VALID_UNTIL = {GEMINI_PILOT_MODEL: date(2026, 12, 31)}
CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
CREDENTIALS = {ANTHROPIC: CREDENTIAL_VARS, GEMINI: ("GEMINI_API_KEY",)}
GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class ExternalCallRefusedError(RuntimeError):
    """An external call was attempted without every guard holding."""


class BudgetExceededError(RuntimeError):
    pass


class ExperimentConfigError(RuntimeError):
    """The provider rejected the request shape itself (HTTP 400: configuration or response
    schema); every later case would send the same shape, so the pilot stops. A refusal, an
    invalid proposal or a verifier rejection is a result, not this."""


@dataclass(frozen=True)
class Budget:
    max_usd: float
    max_tokens: int  # input + output (incl. thinking) tokens over the whole run


def credentials_present(env: Mapping[str, str] | None = None, provider: str = ANTHROPIC) -> bool:
    """Whether the provider's credential is set (its value is never read beyond this)."""
    e = os.environ if env is None else env
    return any(bool(e.get(k)) for k in CREDENTIALS[provider])


def authorize(
    *,
    allow_external: bool,
    manifest_frozen: bool,
    budget: Budget | None,
    env: Mapping[str, str] | None = None,
    provider: str = ANTHROPIC,
) -> None:
    """Raise unless an external call is explicitly allowed, credentialed, tied to a frozen
    pilot manifest and budgeted."""
    if not allow_external:
        raise ExternalCallRefusedError("external calls need --allow-external")
    if not credentials_present(env, provider):
        name = "Anthropic credentials" if provider == ANTHROPIC else "GEMINI_API_KEY"
        raise ExternalCallRefusedError(f"no {name} in the environment")
    if not manifest_frozen:
        raise ExternalCallRefusedError("the pilot manifest is not frozen and verified")
    if budget is None or budget.max_usd <= 0 or budget.max_tokens <= 0:
        raise ExternalCallRefusedError("a positive dollar and token budget is required")


@dataclass
class BudgetMeter:
    budget: Budget
    model: str = PILOT_MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        p_in, p_out = PRICES[self.model]
        return (input_tokens * p_in + output_tokens * p_out) / 1_000_000

    @property
    def spent_usd(self) -> float:
        return self.cost(self.input_tokens, self.output_tokens)

    def admit(self, input_tokens: int, max_output_tokens: int) -> None:
        """Refuse a call whose worst case (all output tokens used) would exceed the budget."""
        tokens = self.input_tokens + self.output_tokens + input_tokens + max_output_tokens
        usd = self.spent_usd + self.cost(input_tokens, max_output_tokens)
        if tokens > self.budget.max_tokens or usd > self.budget.max_usd:
            raise BudgetExceededError(
                f"the next call could reach {tokens} tokens / ${usd:.2f} "
                f"(cap {self.budget.max_tokens} / ${self.budget.max_usd:.2f})"
            )

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1


@dataclass
class ProviderResponse:
    raw_text: str | None  # the text block as returned (None: no proposal)
    response_model: str | None
    stop_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    experiment_valid: bool = True
    invalid_reason: str = ""


def input_token_bound(body: dict[str, Any]) -> int:
    """A conservative upper bound of a request's input tokens (one per character) for budget
    admission when no measured count is available."""
    from mabhas19.shadow.context import canonical_json  # noqa: PLC0415

    return len(canonical_json(body))


class AnthropicProposer:
    """Text-only proposals from ``claude-opus-5-5``; constructed only when every guard holds."""

    def __init__(  # noqa: PLR0913
        self,
        settings: ProposerSettings | None = None,
        *,
        allow_external: bool = False,
        manifest_frozen: bool = False,
        budget: Budget | None = None,
        env: Mapping[str, str] | None = None,
        client: Any = None,
    ) -> None:
        self.settings = settings or ProposerSettings()
        if self.settings.model_id != PILOT_MODEL or self.settings.provider != "anthropic":
            raise ExternalCallRefusedError(
                f"model {self.settings.model_id!r} is not the approved pilot model {PILOT_MODEL}"
            )
        authorize(
            allow_external=allow_external, manifest_frozen=manifest_frozen, budget=budget, env=env
        )
        assert budget is not None
        self.meter = BudgetMeter(budget, self.settings.model_id)
        self._client = client if client is not None else self._sdk_client()

    def _sdk_client(self) -> Any:
        import anthropic  # noqa: PLC0415

        return anthropic.Anthropic(
            timeout=self.settings.timeout_s, max_retries=self.settings.max_retries
        )

    def request(self, ctx: CandidateContext) -> dict[str, Any]:
        return render_request(ctx, self.settings)

    def count_tokens(self, ctx: CandidateContext) -> int:
        """Measured input tokens of the exact request (the token-counting endpoint; free, but
        an external call under the same guards)."""
        body = self.request(ctx)
        resp = self._client.messages.count_tokens(
            model=body["model"],
            system=body["system"],
            messages=body["messages"],
            output_config=body["output_config"],
        )
        return int(resp.input_tokens)

    def propose(self, ctx: CandidateContext) -> ProviderResponse:
        body = self.request(ctx)
        self.meter.admit(input_token_bound(body), self.settings.max_output_tokens)
        start = time.perf_counter()
        try:
            resp = self._client.messages.create(**body)
        except Exception as exc:  # recorded as a failed call, never retried here
            self.meter.add(0, 0)
            return ProviderResponse(
                None,
                None,
                None,
                {"error": type(exc).__name__, "latency_ms": _ms(start)},
                False,
                f"call failed: {type(exc).__name__}",
            )
        usage = getattr(resp, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        self.meter.add(tokens_in, tokens_out)
        stop = getattr(resp, "stop_reason", None)
        text = next(
            (
                b.text
                for b in getattr(resp, "content", []) or []
                if getattr(b, "type", "") == "text"
            ),
            None,
        )
        out = ProviderResponse(
            text if stop != "refusal" else None,
            getattr(resp, "model", None),
            stop,
            {
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "latency_ms": _ms(start),
                "stop_reason": stop,
                "request_id": getattr(resp, "_request_id", None),
            },
        )
        if out.response_model != self.settings.model_id:
            out.experiment_valid = False
            out.invalid_reason = f"response model {out.response_model!r} is not the requested one"
        return out


class GeminiProposer:
    """Text-only proposals from ``gemini-3.8-flash`` over REST; constructed only when every
    guard holds. ``client`` is anything with an httpx-style ``post(url, json=, headers=,
    timeout=)`` (tests inject an in-process fake)."""

    def __init__(  # noqa: PLR0913
        self,
        settings: ProposerSettings | None = None,
        *,
        allow_external: bool = False,
        manifest_frozen: bool = False,
        budget: Budget | None = None,
        env: Mapping[str, str] | None = None,
        client: Any = None,
        today: date | None = None,
    ) -> None:
        self.settings = settings or ProposerSettings.gemini()
        if self.settings.provider != GEMINI or self.settings.model_id != GEMINI_PILOT_MODEL:
            raise ExternalCallRefusedError(
                f"model {self.settings.model_id!r} is not the approved pilot model "
                f"{GEMINI_PILOT_MODEL}"
            )
        if self.settings.effort not in ("low", "medium", "high"):
            raise ExternalCallRefusedError(f"thinking level {self.settings.effort!r}")
        authorize(
            allow_external=allow_external,
            manifest_frozen=manifest_frozen,
            budget=budget,
            env=env,
            provider=GEMINI,
        )
        day = today or datetime.now(UTC).date()
        if day > PRICE_VALID_UNTIL[GEMINI_PILOT_MODEL]:
            raise ExternalCallRefusedError(
                f"the pinned {GEMINI_PILOT_MODEL} price expired on "
                f"{PRICE_VALID_UNTIL[GEMINI_PILOT_MODEL]}; re-pin it before any call"
            )
        assert budget is not None
        self.meter = BudgetMeter(budget, self.settings.model_id)
        self._env = os.environ if env is None else env
        self._transient: tuple[type[BaseException], ...] = (OSError,)
        self.attempts: dict[str, Any] = {}
        self._client = client if client is not None else self._http_client()

    def _http_client(self) -> Any:
        import httpx  # noqa: PLC0415

        self._transient = (OSError, httpx.TransportError)
        return httpx.Client(timeout=self.settings.timeout_s)

    def request(self, ctx: CandidateContext) -> dict[str, Any]:
        return render_request(ctx, self.settings)

    def _post(self, method: str, body: dict[str, Any]) -> Any:
        """One POST with at most ``max_retries`` retries of transport errors and 429 / 5xx.
        ``self.attempts`` records the attempts made and the first retried error's category
        (status or exception class only: never headers or credentials)."""
        url = f"{GEMINI_API}/models/{self.settings.model_id}:{method}"
        headers = {"x-goog-api-key": self._env["GEMINI_API_KEY"]}
        self.attempts = {"attempts": 0, "retried": False, "first_error": None}
        for attempt in range(self.settings.max_retries + 1):
            last = attempt == self.settings.max_retries
            self.attempts["attempts"] = attempt + 1
            try:
                resp = self._client.post(
                    url, json=body, headers=headers, timeout=self.settings.timeout_s
                )
            except self._transient as exc:
                self.attempts["first_error"] = self.attempts["first_error"] or (
                    f"transport:{type(exc).__name__}"
                )
                if last:
                    raise
                self.attempts["retried"] = True
                continue
            if resp.status_code in _RETRY_STATUS and not last:
                self.attempts["first_error"] = self.attempts["first_error"] or (
                    f"HTTP {resp.status_code}"
                )
                self.attempts["retried"] = True
                continue
            return resp
        raise AssertionError("unreachable")

    def count_tokens(self, ctx: CandidateContext) -> int:
        """Measured input tokens of the exact generation body (``countTokens`` with a
        ``generateContentRequest``: system instruction, contents and generation config incl.
        the response schema). No output is generated."""
        body = {"model": f"models/{self.settings.model_id}", **self.request(ctx)}
        resp = self._post("countTokens", {"generateContentRequest": body})
        if resp.status_code != 200:
            raise RuntimeError(f"countTokens failed: HTTP {resp.status_code}")
        return int(resp.json()["totalTokens"])

    def propose(self, ctx: CandidateContext) -> ProviderResponse:
        body = self.request(ctx)
        self.meter.admit(input_token_bound(body), self.settings.max_output_tokens)
        start = time.perf_counter()
        try:
            resp = self._post("generateContent", body)
        except Exception as exc:  # recorded as a failed call, never retried here
            self.meter.add(0, 0)
            return _failed(type(exc).__name__, start, self.attempts)
        if resp.status_code == 400:  # the request shape itself: same for every case
            self.meter.add(0, 0)
            raise ExperimentConfigError(f"HTTP 400 {_api_error(resp)}")
        if resp.status_code != 200:
            self.meter.add(0, 0)
            return _failed(f"HTTP {resp.status_code}", start, self.attempts)
        doc = resp.json()
        u = doc.get("usageMetadata") or {}
        tokens_in = int(u.get("promptTokenCount") or 0)
        thoughts = int(u.get("thoughtsTokenCount") or 0)
        tokens_out = int(u.get("candidatesTokenCount") or 0) + thoughts  # both bill as output
        self.meter.add(tokens_in, tokens_out)
        blocked = (doc.get("promptFeedback") or {}).get("blockReason")
        cand = (doc.get("candidates") or [{}])[0]
        stop = blocked or cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
        out = ProviderResponse(
            None if blocked or stop in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST") else text,
            doc.get("modelVersion"),
            stop,
            {
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "thinking_tokens": thoughts,
                "latency_ms": _ms(start),
                "stop_reason": stop,
                "request_id": doc.get("responseId"),
                **self.attempts,
            },
        )
        if out.response_model != self.settings.model_id:
            out.experiment_valid = False
            out.invalid_reason = f"response model {out.response_model!r} is not the requested one"
        return out


def make_proposer(  # noqa: PLR0913
    settings: ProposerSettings,
    *,
    allow_external: bool,
    manifest_frozen: bool,
    budget: Budget | None,
    env: Mapping[str, str] | None = None,
    client: Any = None,
) -> AnthropicProposer | GeminiProposer:
    """The adapter for the settings' provider (never another provider or model)."""
    if APPROVED_MODELS.get(settings.provider) != settings.model_id:
        raise ExternalCallRefusedError(
            f"{settings.provider}/{settings.model_id} is not an approved pilot model"
        )
    cls = GeminiProposer if settings.provider == GEMINI else AnthropicProposer
    return cls(
        settings,
        allow_external=allow_external,
        manifest_frozen=manifest_frozen,
        budget=budget,
        env=env,
        client=client,
    )


def _failed(
    error: str, start: float, attempts: Mapping[str, Any] | None = None
) -> ProviderResponse:
    return ProviderResponse(
        None,
        None,
        None,
        {"error": error, "latency_ms": _ms(start), **(attempts or {})},
        False,
        f"call failed: {error}",
    )


def _api_error(resp: Any) -> str:
    """The API's own error status and message (body only; never request or response headers)."""
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        return ""
    return f"{err.get('status', '')}: {str(err.get('message', ''))[:300]}"


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)
