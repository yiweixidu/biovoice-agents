"""
biovoice/agents/base.py
BaseAgent ABC and FetchResult data model.
Every data source agent inherits from BaseAgent and must implement
fetch(), get_default_prompt(), and get_capabilities().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Per-agent configuration loaded from config.yaml or env vars."""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    prompt_template: Optional[str] = None  # overrides get_default_prompt()
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class FetchResult(BaseModel):
    """Unified return type from every agent's fetch() call."""
    source: str                      # agent name, e.g. "pubmed"
    items: List[Dict[str, Any]]      # raw structured records
    metadata: Dict[str, Any]         # e.g. {"total": 150, "query_time_s": 2.1}
    prompt_context: str              # pre-formatted text ready for LLM consumption

    @property
    def count(self) -> int:
        return len(self.items)


class BaseAgent(ABC):
    """
    Abstract base for all BioVoice data source agents.

    Subclasses are discovered automatically via the 'biovoice.agents'
    entry_point group declared in pyproject.toml.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    async def fetch(self, query: str, limit: int = 50, **kwargs) -> FetchResult:
        """
        Retrieve and structure data for `query`.
        Must be async-safe; use httpx or aiohttp for I/O-bound agents.
        Synchronous agents should wrap blocking calls in asyncio.to_thread().
        """

    @abstractmethod
    def get_default_prompt(self) -> str:
        """
        Return the default prompt template for this agent.
        Supports {topic}, {query}, {count} placeholders.
        Can be overridden per-user via AgentConfig.prompt_template.
        """

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Declare what kind of data this agent provides.
        Used by the orchestrator to select relevant agents for a task.
        Examples: ['literature', 'structure', 'sequence', 'clinical', 'chemical']
        """

    # ── Convenience helpers ───────────────────────────────────────────────────

    def effective_prompt(self) -> str:
        """Return user-overridden prompt if set, otherwise the default."""
        return self.config.prompt_template or self.get_default_prompt()

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} enabled={self.config.enabled}>"