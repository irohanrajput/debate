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

    # pop the next scripted object for this call kind; fails loudly if unscripted
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
    """One-hot md5 vectors: identical texts -> cosine 1.0, different -> ~0."""

    DIM = 64

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    @classmethod
    def _vec(cls, text: str) -> list[float]:
        import hashlib

        idx = int(hashlib.md5(text.encode()).hexdigest(), 16) % cls.DIM
        return [1.0 if i == idx else 0.0 for i in range(cls.DIM)]
