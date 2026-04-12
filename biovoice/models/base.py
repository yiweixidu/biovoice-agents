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


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model_client(cfg: Dict) -> ModelClient:
    """
    Build a ModelClient from a config dict:
      {
        "llm_type":    "openai" | "ollama",
        "llm_model":   "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
        "temperature": 0.1,
      }
    """
    from biovoice.models.openai_client import OpenAIClient
    from biovoice.models.ollama_client import OllamaClient

    llm_type = cfg.get("llm_type", "openai")
    if llm_type == "openai":
        return OpenAIClient(
            llm_model=cfg.get("llm_model", "gpt-4o-mini"),
            embed_model=cfg.get("embed_model", "text-embedding-3-small"),
            temperature=float(cfg.get("temperature", 0.1)),
        )
    elif llm_type == "ollama":
        return OllamaClient(
            llm_model=cfg.get("llm_model", "llama3.2:3b"),
            embed_model=cfg.get("embed_model", "BAAI/bge-large-en-v1.5"),
            temperature=float(cfg.get("temperature", 0.1)),
        )
    else:
        raise ValueError(f"Unknown llm_type: {llm_type!r}")