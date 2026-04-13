"""
biovoice/_skill_shim.py
Local SkillManifest shim for standalone use of biovoice-agents.

This provides the SkillManifest dataclass when the flubroad framework
package is NOT installed. Once flubroad is published and installed as a
dependency, this file becomes dead code — skill.py will import directly
from flubroad.skill.

This shim must stay in sync with flubroad/skill.py in the flubroad repo.
When flubroad updates the SkillManifest interface (new optional fields,
changed signatures), update this file to match and bump the Skill version.
"""

# Re-export the full SkillManifest implementation inline.
# The canonical source of truth is flubroad/flubroad/skill.py.
# Keep this file minimal — just enough to satisfy biovoice/skill.py imports.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillManifest:
    """Local shim — mirrors flubroad.skill.SkillManifest v1.0."""

    # Required
    name:                  str
    version:               str
    display_name:          str
    agents:                List[str]
    section_queries:       Dict[str, str]
    section_instructions:  Dict[str, str]
    extraction_schema:     Dict[str, Any]

    # Recommended
    description:           str          = ""
    topic_keywords:        List[str]    = field(default_factory=list)
    system_prompt:         str          = ""

    # Optional
    default_agents:        List[str]    = field(default_factory=list)
    grant_templates:       Dict[str, str] = field(default_factory=dict)
    knowledge_graph_config: Dict[str, Any] = field(default_factory=dict)
    output_section_order:  List[str]    = field(default_factory=list)
    output_section_titles: Dict[str, str] = field(default_factory=dict)
    finetuning_config:     Dict[str, Any] = field(default_factory=dict)
    chart_config:          Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.default_agents:
            self.default_agents = list(self.agents)
        if not self.output_section_order:
            self.output_section_order = list(self.section_queries.keys())
        if not self.system_prompt:
            self.system_prompt = (
                f"You are an expert {self.display_name} research assistant. "
                "Write accurate, well-cited biomedical synthesis sections. "
                "Cite every factual claim with a PMID."
            )

    @property
    def sections(self) -> List[str]:
        return self.output_section_order

    def section_title(self, key: str) -> str:
        return self.output_section_titles.get(key, key.replace("_", " ").title())

    def is_relevant(self, title: str, abstract: str, min_hits: int = 1) -> bool:
        if not self.topic_keywords:
            return True
        text = (title + " " + abstract).lower()
        hits = sum(1 for kw in self.topic_keywords if kw.lower() in text)
        return hits >= min_hits

    def summary(self) -> str:
        return (
            f"Skill({self.name} v{self.version}) | "
            f"{len(self.agents)} agents | "
            f"{len(self.section_queries)} sections"
        )
