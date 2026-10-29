"""The text-only Anthropic proposer (SH0 pilot adapter) and the external-call guards.

One Messages API request per question: the pinned instructions as the system prompt, the
redacted candidate context as the user turn, structured output with the provider projection of
the proposal schema, ``output_config.effort`` and the model's default adaptive thinking. No
``temperature`` (the model rejects sampling parameters), no server-side model fallback, no beta.
The SDK retries transport and transient API errors once (``max_retries=1``); a returned
proposal is never regenerated - a schema-invalid or verifier-rejected proposal is a result.

The actual ``response.model`` is recorded; a response from any model other than the requested
one is marked invalid for the experiment.

No call is possible unless every guard holds (:func:`authorize`): ``allow_external`` explicitly
given, Anthropic credentials present in the environment (never read, printed or stored beyond a
presence check), a frozen pilot manifest verified against the live engine, and a dollar and
token budget, which :class:`BudgetMeter` enforces before every call.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mabhas19.shadow.replay import PILOT_MODEL, ProposerSettings, render_request

if TYPE_CHECKING:
    from mabhas19.shadow.context import CandidateContext

PROVIDER_VERSION = "sh0-anthropic-proposer-1"
# first-party list prices, $ per million tokens (input, output); thinking bills as output
PRICES = {PILOT_MODEL: (4.00, 20.00)}
CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class ExternalCallRefusedError(RuntimeError):
    """An external call was attempted without every guard holding."""


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    max_usd: float
    max_tokens: int  # input + output (incl. thinking) tokens over the whole run


def credentials_present(env: Mapping[str, str] | None = None) -> bool:
    """Whether an Anthropic credential is set (its value is never read beyond this)."""
    e = os.environ if env is None else env
    return any(bool(e.get(k)) for k in CREDENTIAL_VARS)


def authorize(
    *,
    allow_external: bool,
    manifest_frozen: bool,
    budget: Budget | None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Raise unless an external call is explicitly allowed, credentialed, tied to a frozen
    pilot manifest and budgeted."""
    if not allow_external:
        raise ExternalCallRefusedError("external calls need --allow-external")
    if not credentials_present(env):
        raise ExternalCallRefusedError("no Anthropic credentials in the environment")
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


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)
