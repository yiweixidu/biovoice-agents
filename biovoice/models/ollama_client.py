"""
biovoice/models/base.py + openai_client.py + ollama_client.py
Unified model abstraction layer.
All LLM calls go through ModelClient so the orchestrator stays
model-agnostic.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

OPENAI_BASE_URL = "https://api.openai.com/v1"


# ── Base ──────────────────────────────────────────────────────────────────────

class ModelClient(ABC):
    """Abstract base for all model backends."""

    @abstractmethod
    def chat(
        self,
        system: str,
        human: str,
        temperature: float = 0.1,
    ) -> str:
        """Single-turn chat completion. Returns the assistant message text."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return dense embeddings for a list of text strings."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier string."""

# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaClient(ModelClient):
    """
    Wraps ChatOllama for local LLM inference.
    Embeddings fall back to HuggingFaceEmbeddings (BAAI/bge-large-en-v1.5).
    """

    def __init__(
        self,
        llm_model:   str = "llama3.2:3b",
        embed_model: str = "BAAI/bge-large-en-v1.5",
        temperature: float = 0.1,
        device:      str = "cpu",
    ):
        from langchain_community.chat_models import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings

        self._llm = ChatOllama(model=llm_model, temperature=temperature)
        self._embedder = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._llm_model   = llm_model
        self._embed_model = embed_model

    @property
    def model_name(self) -> str:
        return f"ollama/{self._llm_model}"

    def chat(self, system: str, human: str, temperature: float = 0.1) -> str:
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        resp = self._llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed_documents(texts)

    @property
    def langchain_llm(self):
        return self._llm
