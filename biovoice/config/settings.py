"""
biovoice/core/task.py     — Task state machine
biovoice/config/settings.py — Pydantic settings loaded from .env
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ── Settings ──────────────────────────────────────────────────────────────────

class BioVoiceSettings(BaseSettings):
    """
    All configuration in one place.
    Values are read from environment variables or .env file.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key:  str  = Field(default="", env="OPENAI_API_KEY")
    llm_type:        str  = Field(default="openai", env="LLM_TYPE")
    llm_model:       str  = Field(default="gpt-4o-mini", env="LLM_MODEL")
    embed_model:     str  = Field(default="text-embedding-3-small", env="EMBED_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")

    # ── PubMed ────────────────────────────────────────────────────────────────
    pubmed_api_key:  str  = Field(default="", env="PUBMED_API_KEY")
    email:           str  = Field(default="biovoice@example.com", env="EMAIL")

    # ── RAG ───────────────────────────────────────────────────────────────────
    collection_name: str  = Field(default="biovoice", env="COLLECTION_NAME")
    persist_dir:     str  = Field(default="./data/vector_db", env="PERSIST_DIR")

    # ── Cache & output ────────────────────────────────────────────────────────
    cache_file:      str  = Field(default="data/cache/articles.json", env="CACHE_FILE")
    output_dir:      str  = Field(default="./output", env="OUTPUT_DIR")
    ppt_template:    str  = Field(default="./templates/lab_template.pptx", env="PPT_TEMPLATE")

    # ── Per-agent defaults ────────────────────────────────────────────────────
    max_papers_per_agent: int  = Field(default=100, env="MAX_PAPERS_PER_AGENT")
    pmc_delay:            float = Field(default=0.5, env="PMC_DELAY")
    unpaywall_delay:      float = Field(default=0.5, env="UNPAYWALL_DELAY")

    class Config:
        env_file      = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def to_orchestrator_config(self) -> Dict[str, Any]:
        """Convert settings to the dict format expected by BioVoiceOrchestrator."""
        return {
            "llm_type":             self.llm_type,
            "llm_model":            self.llm_model,
            "embed_model":          self.embed_model,
            "temperature":          self.llm_temperature,
            "collection_name":      self.collection_name,
            "persist_dir":          self.persist_dir,
            "output_dir":           self.output_dir,
            "ppt_template":         self.ppt_template,
            "max_papers_per_agent": self.max_papers_per_agent,
            "agents": {
                "pubmed": {
                    "enabled":         True,
                    "api_key":         self.pubmed_api_key or None,
                    "email":           self.email,
                    "pmc_delay":       self.pmc_delay,
                    "unpaywall_delay": self.unpaywall_delay,
                    "fetch_fulltext":  True,
                },
                "europe_pmc": {
                    "enabled": True,
                    "email":   self.email,
                    "delay":   self.pmc_delay,
                },
                "pdb":            {"enabled": True},
                "uniprot":        {"enabled": True},
                "clinicaltrials": {"enabled": True},
                "chembl":         {"enabled": True},
                "local_data":     {"enabled": True},
            },
        }