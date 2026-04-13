"""
biovoice/core/grant_config.py
Shared data model for the Virology Grant Copilot pipeline.

GrantConfig     — injectable config that drives orchestrator section queries + prompts
Citation        — a single verified (or flagged) citation
GrantSection    — one rendered section of the grant (text + citations)
GrantOutput     — complete grant run result; consumed by both PPT and Word renderers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Citation:
    pmid: str
    title: str = ""
    authors: str = ""          # "Smith J et al."
    journal: str = ""
    year: str = ""
    doi: str = ""
    suspicious: bool = False   # True when Jaccard < threshold
    warning: str = ""          # human-readable reason for suspicion


@dataclass
class GrantSection:
    key: str                   # e.g. "significance"
    title: str                 # e.g. "Significance"
    text: str                  # rendered prose with [N] inline citation markers
    citations: List[Citation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GrantOutput:
    research_question: str
    sections: List[GrantSection] = field(default_factory=list)
    all_citations: List[Citation] = field(default_factory=list)
    citation_warnings: List[str] = field(default_factory=list)
    ppt_file: Optional[str] = None
    word_file: Optional[str] = None

    def section(self, key: str) -> Optional[GrantSection]:
        for s in self.sections:
            if s.key == key:
                return s
        return None

    @property
    def has_warnings(self) -> bool:
        return bool(self.citation_warnings)

    @property
    def warning_count(self) -> int:
        return len(self.citation_warnings)


@dataclass
class GrantConfig:
    """
    Injectable configuration for grant mode.

    Pass a GrantConfig to BioVoiceOrchestrator.__init__ to activate grant mode.
    The orchestrator replaces its default SECTION_QUERIES and SECTION_INSTRUCTIONS
    with the values from this config and skips RAG construction.

    Defaults load from domain/virology/prompts/pmrc_templates.py so callers
    only need to override research_question.
    """

    research_question: str

    # Override section ordering and which sections to produce.
    # Defaults to the standard NIH grant flow.
    section_keys: List[str] = field(
        default_factory=lambda: [
            "specific_aims",
            "significance",
            "innovation",
            "background",
        ]
    )

    # Section-level search queries (keyed by section_keys entries).
    # Leave None to load defaults from pmrc_templates.SECTION_QUERIES.
    section_queries: Optional[Dict[str, str]] = None

    # Section-level synthesis instructions.
    # Leave None to load defaults from pmrc_templates.SECTION_INSTRUCTIONS.
    section_instructions: Optional[Dict[str, str]] = None

    # Max papers to pull per agent fetch (abstracts only, no full text).
    max_papers_per_agent: int = 20

    # Max combined papers after dedup and ranking.
    max_ranked_papers: int = 30

    # Jaccard similarity threshold below which a citation is flagged suspicious.
    # Computed against title + abstract tokens (after biomedical stoplist removal).
    # 0.08 catches true hallucinations (Jaccard 0.00–0.04) without penalizing
    # legitimate synthesis prose that paraphrases the source at a higher level
    # of abstraction. Tighter prompts push scores higher naturally.
    jaccard_threshold: float = 0.08

    # Output directory for .docx and .pptx files.
    output_dir: str = "./output"

    def __post_init__(self):
        from domain.virology.prompts.pmrc_templates import (
            SECTION_QUERIES as DEFAULT_QUERIES,
            SECTION_INSTRUCTIONS as DEFAULT_INSTRUCTIONS,
        )
        if self.section_queries is None:
            self.section_queries = DEFAULT_QUERIES
        if self.section_instructions is None:
            self.section_instructions = DEFAULT_INSTRUCTIONS
