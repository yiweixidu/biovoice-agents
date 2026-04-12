"""
biovoice/agents/registry.py
Plugin-based agent registry.
Agents are discovered from the 'biovoice.agents' entry_point group,
so third-party packages can add new agents without touching this file.
"""

from __future__ import annotations

import importlib.metadata
from typing import Dict, List, Optional, Type

from .base import AgentConfig, BaseAgent


class AgentRegistry:
    """
    Singleton registry that maps agent names → agent classes.
    Populated at startup by load_plugins() which reads pyproject entry_points.
    """

    _agents: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str, agent_cls: Type[BaseAgent]) -> None:
        """Manually register an agent class (useful for testing)."""
        cls._agents[name] = agent_cls
        print(f"[Registry] Registered agent: {name} → {agent_cls.__name__}")

    @classmethod
    def load_plugins(cls) -> None:
        """
        Discover and register all agents declared under the
        'biovoice.agents' entry_point group in any installed package.
        """
        try:
            eps = importlib.metadata.entry_points(group="biovoice.agents")
        except TypeError:
            # Python < 3.12 fallback
            all_eps = importlib.metadata.entry_points()
            eps = all_eps.get("biovoice.agents", [])

        for ep in eps:
            try:
                agent_cls = ep.load()
                cls.register(ep.name, agent_cls)
            except Exception as exc:
                print(f"[Registry] Failed to load agent {ep.name!r}: {exc}")

    @classmethod
    def get(cls, name: str, config: Optional[AgentConfig] = None) -> BaseAgent:
        """
        Instantiate and return an agent by name.
        If config is None, a default AgentConfig(name=name) is used.
        """
        if name not in cls._agents:
            raise KeyError(
                f"Agent {name!r} not found. "
                f"Available: {list(cls._agents.keys())}"
            )
        cfg = config or AgentConfig(name=name)
        return cls._agents[name](cfg)

    @classmethod
    def list_agents(cls) -> List[str]:
        return sorted(cls._agents.keys())

    @classmethod
    def available(cls) -> Dict[str, Type[BaseAgent]]:
        return dict(cls._agents)

    @classmethod
    def build_from_config(cls, agents_config: Dict) -> List[BaseAgent]:
        """
        Build a list of enabled agent instances from a config dict like:
          {
            "pubmed":  {"enabled": true, "prompt_template": "..."},
            "pdb":     {"enabled": false},
          }
        """
        instances = []
        for name, raw in agents_config.items():
            if not raw.get("enabled", True):
                continue
            cfg = AgentConfig(name=name, **raw)
            try:
                instances.append(cls.get(name, cfg))
            except KeyError:
                print(f"[Registry] Warning: agent {name!r} not installed — skipping.")
        return instances