from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from committee.models import Tier, Usage

T = TypeVar("T", bound=BaseModel)


class FakeProvider:
    """Scripted provider: pops responses per kind (FIFO). No network."""

    def __init__(self, scripts: dict[str, list[BaseModel]], tokens_per_call: int = 100) -> None:
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._tokens = tokens_per_call
        self.calls: list[dict] = []

    async def structured(
        self, *, schema: type[T], system: str, user: str, tier: Tier, max_tokens: int, kind: str
    ) -> tuple[T, Usage]:
        self.calls.append({"kind": kind, "tier": tier, "max_tokens": max_tokens})
        queue = self._scripts.get(kind) or self._scripts.get("*")
        if not queue:
            raise AssertionError(f"FakeProvider has no script for kind={kind}")
        obj = queue.pop(0)
        assert isinstance(obj, schema), f"scripted {type(obj).__name__} != expected {schema.__name__}"
        usage = Usage(input_tokens=self._tokens, output_tokens=self._tokens, model="fake", kind=kind)
        return obj, usage


class FakeEmbedder:
    """Deterministic hash-based vectors; only pairwise consistency matters in tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str, dim: int = 16) -> list[float]:
        seed = [float((hash(text) >> i) % 97) for i in range(0, dim * 3, 3)]
        norm = sum(x * x for x in seed) ** 0.5 or 1.0
        return [x / norm for x in seed]
