from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from committee.config import settings


def _embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=settings.gemini_api_key)


def _entity_for(content: str, entities: list[str]) -> str | None:
    return next((e for e in entities if e.split()[0].lower() in content.lower()), None)


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
                          "entity": entity or "", "source": fact.get("source", ""),
                          "reliability": reliability if reliability is not None else 0.0,
                          "timestamp": fact.get("timestamp", "")},
            ))
    return docs


def load_decisions_dataset(path: Path) -> list[Document]:
    data = json.loads(path.read_text())
    docs: list[Document] = []
    for d in data.get("decisions", []):
        header = f"[{d['entity']} | Ridgeline {d['decision']} decision | {d['timestamp']}]"
        outcome = f" Outcome: {d['outcome']}" if d.get("outcome") else ""
        docs.append(Document(
            page_content=f"{header} {d['rationale']}{outcome}",
            metadata={"kind": "decision", "record_id": d["id"], "entity": d["entity"],
                      "decision": d["decision"], "timestamp": d["timestamp"], "reliability": 1.0},
        ))
    return docs


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
                                 metadata={"kind": "doc", "record_id": f"{path.stem}:{i}",
                                           "entity": "", "source": path.name, "reliability": 0.0}))
    return docs


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

    def search(self, query: str, entity: str | None = None, min_reliability: float | None = None,
               k: int | None = None) -> list[Document]:
        flt = {"entity": entity} if entity else None
        results = self._store.similarity_search(query, k=k or settings.retrieval_k, filter=flt)
        if min_reliability is not None:
            results = [d for d in results if d.metadata.get("reliability", 0.0) >= min_reliability]
        return results


class NullCorpus:
    """Used when no corpus has been ingested; tools degrade gracefully."""

    def search(self, query: str, entity: str | None = None, min_reliability: float | None = None,
               k: int | None = None) -> list[Document]:
        return []
