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


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIClient(ModelClient):
    """
    Wraps ChatOpenAI for LLM calls and OpenAIEmbeddings for vector ops.
    Forces openai_api_base to prevent accidental Anthropic routing.
    """

    def __init__(
        self,
        llm_model:   str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        temperature: float = 0.1,
        api_key:     Optional[str] = None,
    ):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        key = api_key or os.environ["OPENAI_API_KEY"]

        self._llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=key,
            openai_api_base=OPENAI_BASE_URL,
        )
        self._embedder = OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=key,
            openai_api_base=OPENAI_BASE_URL,
        )
        self._llm_model   = llm_model
        self._embed_model = embed_model

    @property
    def model_name(self) -> str:
        return f"openai/{self._llm_model}"

    def chat(self, system: str, human: str, temperature: float = 0.1) -> str:
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        resp = self._llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed_documents(texts)

    @property
    def langchain_llm(self):
        """Expose the raw LangChain LLM for modules that need it directly."""
        return self._llm