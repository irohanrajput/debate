from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from committee.config import settings
from committee.llm.embeddings import get_embedder as _embedder


# best-effort: which of the sequence's entities does this fact mention
def _entity_for(content: str, entities: list[str]) -> str | None:
    return next((e for e in entities if e.split()[0].lower() in content.lower()), None)


# normalized filter key so 'NovaTech' matches 'NovaTech Inc.'
def entity_key(entity: str | None) -> str:
    return entity.split()[0].lower() if entity else ""


# 1 fact = 1 chunk, header-prefixed, with entity/source/reliability/date metadata
def load_facts_dataset(path: Path) -> list[Document]:
    data = json.loads(path.read_text())
    docs: list[Document] = []
    for seq_name, seq in data.items():
        if not isinstance(seq, dict) or "facts" not in seq:
            continue
        entities = seq.get("entities", [])
        for fact in seq["facts"]:
            reliability = settings.reliability_map.get(str(fact.get("source_reliability", "")).lower())
            entity = _entity_for(fact["content"], entities)
            header = f"[{entity or 'unknown'} | {fact.get('source', 'unknown')} | {fact.get('timestamp', '')}]"
            docs.append(Document(
                page_content=f"{header} {fact['content']}",
                metadata={"kind": "fact", "record_id": fact["id"], "sequence": seq_name,
                          "entity": entity or "", "entity_key": entity_key(entity), "source": fact.get("source", ""),
                          "reliability": reliability if reliability is not None else 0.0,
                          "timestamp": fact.get("timestamp", "")},
            ))
    return docs


# 1 past investment decision = 1 chunk (rationale + outcome)
def load_decisions_dataset(path: Path) -> list[Document]:
    data = json.loads(path.read_text())
    docs: list[Document] = []
    for d in data.get("decisions", []):
        header = f"[{d['entity']} | Ridgeline {d['decision']} decision | {d['timestamp']}]"
        outcome = f" Outcome: {d['outcome']}" if d.get("outcome") else ""
        docs.append(Document(
            page_content=f"{header} {d['rationale']}{outcome}",
            metadata={"kind": "decision", "record_id": d["id"], "entity": d["entity"],
                      "entity_key": entity_key(d["entity"]),
                      "decision": d["decision"], "timestamp": d["timestamp"], "reliability": 1.0},
        ))
    return docs


# free-form md/txt files get real splitting (~2000 chars, 200 overlap)
def load_text_docs(directory: Path) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_chars,
                                              chunk_overlap=settings.chunk_overlap_chars)
    docs: list[Document] = []
    for path in sorted(directory.glob("**/*")):
        if path.suffix not in (".md", ".txt"):
            continue
        for i, chunk in enumerate(splitter.split_text(path.read_text())):
            docs.append(Document(page_content=f"[{path.stem}] {chunk}",
                                 metadata={"kind": "doc", "record_id": f"{path.stem}:{i}", "entity": "",
                                           "entity_key": "", "source": path.name, "reliability": 0.0}))
    return docs


# ingest everything in data/ into the persistent Chroma index; returns chunk count
def build_corpus(data_dir: Path) -> int:
    from langchain_chroma import Chroma

    docs: list[Document] = []
    for path in sorted(data_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        if "decisions" in raw:
            docs.extend(load_decisions_dataset(path))
        else:
            docs.extend(load_facts_dataset(path))
    docs.extend(load_text_docs(data_dir))
    if not docs:
        return 0
    Chroma.from_documents(docs, _embedder(), collection_name=settings.corpus_collection,
                          persist_directory=settings.chroma_dir)
    return len(docs)


class CorpusIndex:
    def __init__(self) -> None:
        from langchain_chroma import Chroma

        self._store = Chroma(collection_name=settings.corpus_collection,
                             persist_directory=settings.chroma_dir, embedding_function=_embedder())

    # embed the query, nearest-neighbor with entity filter, then reliability filter
    def search(self, query: str, entity: str | None = None, min_reliability: float | None = None,
               k: int | None = None) -> list[Document]:
        flt = {"entity_key": entity_key(entity)} if entity else None
        results = self._store.similarity_search(query, k=k or settings.retrieval_k, filter=flt)
        if min_reliability is not None:
            results = [d for d in results if d.metadata.get("reliability", 0.0) >= min_reliability]
        return results


    def timeline(self, entity: str) -> list[Document]:
        """Every stored record for an entity, oldest first. The datasets' signal
        is in sequences (claim, denial, restatement); similarity search hides that."""
        raw = self._store.get(where={"entity_key": entity_key(entity)})
        docs = [Document(page_content=text, metadata=meta)
                for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or [])]
        seen: set[str] = set()
        unique = []
        for d in sorted(docs, key=lambda d: str(d.metadata.get("timestamp", ""))):
            rid = str(d.metadata.get("record_id"))
            if rid not in seen:
                seen.add(rid)
                unique.append(d)
        return unique


class NullCorpus:
    """Used when no corpus has been ingested; tools degrade gracefully."""

    def search(self, query: str, entity: str | None = None, min_reliability: float | None = None,
               k: int | None = None) -> list[Document]:
        return []

    def timeline(self, entity: str) -> list[Document]:
        return []
