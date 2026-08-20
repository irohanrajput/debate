from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from committee.config import settings


# single place that knows which embedding model and key to use
def get_embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=settings.gemini_api_key)
