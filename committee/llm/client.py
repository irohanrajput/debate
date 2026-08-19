from __future__ import annotations

from typing import Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from committee.config import settings
from committee.models import Tier, Usage

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    async def structured(
        self, *, schema: type[T], system: str, user: str, tier: Tier, max_tokens: int, kind: str
    ) -> tuple[T, Usage]: ...


class SchemaError(Exception):
    pass


def _model_name(tier: Tier) -> str:
    return settings.pro_model if tier == Tier.PRO else settings.flash_model


def _usage_from(raw: object, model: str, kind: str) -> Usage:
    meta = getattr(raw, "usage_metadata", None) or {}
    return Usage(
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
        model=model,
        kind=kind,
    )


class GeminiProvider:
    def __init__(self) -> None:
        self._cache: dict[str, ChatGoogleGenerativeAI] = {}

    def _llm(self, tier: Tier, max_tokens: int) -> ChatGoogleGenerativeAI:
        thinking = settings.pro_thinking_budget if tier == Tier.PRO else settings.flash_thinking_budget
        return ChatGoogleGenerativeAI(
            model=_model_name(tier),
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=max_tokens + thinking,
            thinking_budget=thinking,
            timeout=settings.llm_timeout_s,
        )

    async def structured(
        self, *, schema: type[T], system: str, user: str, tier: Tier, max_tokens: int, kind: str
    ) -> tuple[T, Usage]:
        llm = self._llm(tier, max_tokens).with_structured_output(schema, include_raw=True)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        total = Usage(model=_model_name(tier), kind=kind)
        for attempt in range(settings.repair_retries + 1):
            result = await llm.ainvoke(messages)
            usage = _usage_from(result.get("raw"), _model_name(tier), kind)
            total.input_tokens += usage.input_tokens
            total.output_tokens += usage.output_tokens
            parsed, err = result.get("parsed"), result.get("parsing_error")
            if parsed is not None:
                return parsed, total
            messages.append(
                HumanMessage(content=f"Your previous output failed validation: {err}. Return valid output matching the schema.")
            )
        raise SchemaError(f"structured output failed after retries ({kind})")
