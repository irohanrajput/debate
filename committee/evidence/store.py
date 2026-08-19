from __future__ import annotations

from committee.config import settings
from committee.models import Evidence


class EvidenceStore:
    """Run-scoped registry of everything fetched, with dedupe and provenance."""

    def __init__(self) -> None:
        self._by_id: dict[str, Evidence] = {}
        self._by_key: dict[tuple[str, str], str] = {}

    def register(self, *, source: str, ref: str, snippet: str, as_of: str | None = None,
                 reliability: float | None = None, fetched_by: str | None = None) -> Evidence:
        key = (source, ref)
        if key in self._by_key:
            return self._by_id[self._by_key[key]]
        eid = f"EV-{len(self._by_id) + 1}"
        ev = Evidence(id=eid, source=source, ref=ref,
                      snippet=snippet[: settings.evidence_snippet_char_cap],
                      as_of=as_of, reliability=reliability, fetched_by=fetched_by)
        self._by_id[eid] = ev
        self._by_key[key] = eid
        return ev

    def get(self, eid: str) -> Evidence | None:
        return self._by_id.get(eid)

    def known_ids(self) -> set[str]:
        return set(self._by_id)

    def index(self) -> list[Evidence]:
        return list(self._by_id.values())
