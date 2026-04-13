"""
biovoice/core/orchestrator.py
Multi-agent orchestrator.
Coordinates agent selection, parallel data fetching, RAG construction,
section-based review generation, and output pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import warnings
from typing import Callable, Dict, List, Optional, Tuple

from biovoice.agents.base import FetchResult
from biovoice.agents.registry import AgentRegistry
from biovoice.core.task import Task, TaskStatus

# Re-use production-grade RAG and output modules from FluBroad-Voice
from core.presentation.ppt_generator import PPTGenerator
from core.knowledge_graph import BioVoiceGraph
from domain.virology.schemas.antibody_schema import antibody_schema

from biovoice.models.base import ModelClient, build_model_client

# ── Section queries (same as FluBroad-Voice, kept here for reference) ─────────
SECTION_QUERIES = {
    "problem":    "antigenic drift shift influenza vaccine mismatch pandemic",
    "motivation": "conserved epitope HA stem broadly neutralizing antibody strategy",
    "results":    "broadly neutralizing antibody influenza neutralization spectrum IC50",
    "mechanisms": "Fc effector function ADCC broadly neutralizing antibody mechanism",
    "challenges": "immunodominance germline targeting epitope accessibility challenge",
    "future":     "mRNA vaccine immunogen design deep mutational scanning universal",
}

SECTION_INSTRUCTIONS = {
    "problem": (
        "Write the 'Problem' section of a virology literature review.\n"
        "- Describe the clinical challenge: antigenic drift, vaccine mismatch, pandemic risk.\n"
        "- Quantify the problem with specific statistics from the provided context.\n"
        "- Every factual claim MUST include a PMID in parentheses.\n"
        "- Length: 150-200 words. No section header."
    ),
    "motivation": (
        "Write the 'Motivation' section.\n"
        "- Explain WHY targeting conserved epitopes is promising.\n"
        "- Critically compare to strain-specific vaccination.\n"
        "- Every claim MUST include a PMID.\n"
        "- Length: 150-200 words. No header."
    ),
    "results": (
        "Write the 'Key Results' section.\n"
        "- Compare at least 4-5 specific broadly neutralizing antibodies.\n"
        "- Include: name, epitope, IGHV gene, neutralization spectrum, IC50, clinical stage.\n"
        "- Synthesize patterns; do NOT just list.\n"
        "- Every claim MUST include a PMID. Length: 300-400 words. No header."
    ),
    "mechanisms": (
        "Write the 'Mechanisms' section.\n"
        "- Describe Fc effector functions, ADCC, ADCP, complement.\n"
        "- Cite Fc-knockout experiments.\n"
        "- Note controversies. Every claim MUST include a PMID.\n"
        "- Length: 150-200 words. No header."
    ),
    "challenges": (
        "Write the 'Technical Challenges' section.\n"
        "- At least 3 challenges: immunodominance, stem accessibility, germline requirements, escape.\n"
        "- For each: cite evidence AND describe approaches to overcome it.\n"
        "- Which remain unsolved? Every claim MUST include a PMID.\n"
        "- Length: 200-250 words. No header."
    ),
    "future": (
        "Write the 'Future Directions' section.\n"
        "- 3-4 SPECIFIC suggestions NOT quoted from papers.\n"
        "- Each must be mechanistically justified.\n"
        "- No vague 'further research is needed'.\n"
        "- Length: 200-250 words. No header."
    ),
}

TOPIC_KEYWORDS = [
    "influenza", "hemagglutinin", "neuraminidase", "bnab", "broadly neutralizing",
    "sars-cov-2", "coronavirus", "antibody", "vaccine", "epitope", "stem",
]


def _is_relevant(article: Dict, min_hits: int = 1) -> bool:
    text = (
        (article.get("title") or "") + " " + (article.get("abstract") or "")
    ).lower()
    return sum(1 for kw in TOPIC_KEYWORDS if kw in text) >= min_hits


# ── Citation verification helpers ─────────────────────────────────────────────

def _extract_cite_pmids(text: str) -> List[str]:
    """Return all PMID strings found in [CITE:PMID] markers."""
    return re.findall(r'\[CITE:(\d+)\]', text)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _token_set(text: str, stoplist: set) -> set:
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return {t for t in tokens if t not in stoplist and len(t) > 1}


def verify_citations(
    synthesis_text: str,
    ranked_papers: List[Dict],
    jaccard_threshold: float = 0.15,
    stoplist: Optional[set] = None,
) -> Tuple[str, List[str]]:
    """
    Check every [CITE:PMID] in synthesis_text against ranked_papers.

    Returns (verified_text, citation_warnings).
    - verified_text has [CITE:PMID] markers replaced with numbered [N] markers.
    - citation_warnings lists suspicious or missing citations for user review.
    """
    if stoplist is None:
        from domain.virology.prompts.pmrc_templates import BIOMEDICAL_STOPLIST
        stoplist = BIOMEDICAL_STOPLIST

    pmid_to_paper: Dict[str, Dict] = {}
    for paper in ranked_papers:
        pmid = str(paper.get("pmid") or paper.get("PMID") or "")
        if pmid:
            pmid_to_paper[pmid] = paper

    cite_pmids = _extract_cite_pmids(synthesis_text)
    unique_pmids = list(dict.fromkeys(cite_pmids))  # preserve order, deduplicate

    pmid_to_number: Dict[str, int] = {}
    warnings_list: List[str] = []

    for pmid in unique_pmids:
        n = len(pmid_to_number) + 1
        pmid_to_number[pmid] = n

        if pmid not in pmid_to_paper:
            warnings_list.append(
                f"[{n}] PMID {pmid}: not found in fetched results — may be hallucinated"
            )
            continue

        # Jaccard check: find sentences in synthesis that cite this PMID,
        # compare against title + abstract (titles carry antibody names and
        # epitope vocabulary that claims share with their sources).
        title    = pmid_to_paper[pmid].get("title") or ""
        abstract = pmid_to_paper[pmid].get("abstract") or ""
        source_tokens = _token_set(title + " " + abstract, stoplist)

        sentences = re.split(r'(?<=[.!?])\s+', synthesis_text)
        for sentence in sentences:
            if f"[CITE:{pmid}]" in sentence:
                claim_tokens = _token_set(sentence, stoplist)
                score = _jaccard(claim_tokens, source_tokens)
                if score < jaccard_threshold:
                    warnings_list.append(
                        f"[{n}] PMID {pmid}: claim may not match source "
                        f"(Jaccard={score:.2f} < {jaccard_threshold}). "
                        f"Review: \"{sentence[:120].strip()}...\""
                    )
                    break  # one warning per paper is enough

    # Replace [CITE:PMID] with [N] in text
    def replace_cite(m: re.Match) -> str:
        pmid = m.group(1)
        n = pmid_to_number.get(pmid, 0)
        return f"[{n}]" if n else f"[CITE:{pmid}]"

    verified_text = re.sub(r'\[CITE:(\d+)\]', replace_cite, synthesis_text)

    if not cite_pmids:
        warnings_list.append(
            "WARNING: synthesis output contains no [CITE:PMID] markers — "
            "all claims are unsupported. Check the synthesis prompt."
        )

    return verified_text, warnings_list


def _format_authors(paper: Dict) -> str:
    """Return 'First Author et al.' style string from a paper dict."""
    authors = paper.get("authors") or paper.get("author_list") or []
    if isinstance(authors, list) and authors:
        first = authors[0]
        if isinstance(first, dict):
            last = first.get("last_name") or first.get("name") or ""
        else:
            last = str(first).split(",")[0].strip()
        suffix = " et al." if len(authors) > 1 else ""
        return last + suffix
    if isinstance(authors, str) and authors:
        parts = authors.split(";")
        first = parts[0].split(",")[0].strip()
        suffix = " et al." if len(parts) > 1 else ""
        return first + suffix
    return ""


class BioVoiceOrchestrator:
    """
    Top-level coordinator for BioVoice-Agents.

    Usage:
        orch = BioVoiceOrchestrator(config)
        result = asyncio.run(orch.run(
            query="broadly neutralizing antibodies influenza",
            agent_names=["pubmed", "pdb"],
            output_types=["review", "ppt"],
        ))
    """

    def __init__(
        self,
        config: Dict,
        use_rag: bool = True,
        grant_config=None,   # Optional[GrantConfig] — avoids circular import at module level
    ):
        self.config       = config
        self.grant_config = grant_config
        self.model        = build_model_client(config)
        self.ppt_gen = PPTGenerator(template_path=config.get("ppt_template"))
        self.tts     = None  # lazy-loaded when video output is requested

        if use_rag:
            from core.rag.vector_store import FluBroadRAG
            self.rag = FluBroadRAG(
                collection_name=config.get("collection_name", "biovoice"),
                persist_directory=config.get("persist_dir", "./data/vector_db"),
            )
        else:
            self.rag = None

        # Load agent plugins once at init
        AgentRegistry.load_plugins()

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        query:        str,
        agent_names:  List[str],
        output_types: List[str] = ("review", "ppt"),
        topic:        str       = "flu_bnabs",
        progress_cb:  Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict:
        """
        Run the full pipeline.

        Parameters
        ----------
        query        : natural language search query
        agent_names  : list of agent keys, e.g. ["pubmed", "pdb", "uniprot"]
        output_types : subset of ["review", "ppt", "video"]
        topic        : prompt template key (default "flu_bnabs")
        progress_cb  : optional callback(stage, current, total)
        """
        task = Task(query=query, agents=agent_names, output_types=output_types)
        task.set_status(TaskStatus.RUNNING)

        try:
            # 1. Parallel agent fetch ─────────────────────────────────────────
            task.set_status(TaskStatus.FETCHING)
            fetch_results = await self._fetch_all(
                query, agent_names, progress_cb
            )

            # 2. Collect + filter articles ────────────────────────────────────
            all_articles = []
            for fr in fetch_results:
                for item in fr.items:
                    if _is_relevant(item):
                        all_articles.append(item)

            print(
                f"[Orchestrator] {len(all_articles)} relevant items from "
                f"{len(fetch_results)} agents"
            )

            # 3. Build RAG (skipped in grant mode) ────────────────────────────
            task.set_status(TaskStatus.INDEXING)
            if self.rag is not None:
                self.rag.build(all_articles)

            # 4. Generate review ──────────────────────────────────────────────
            task.set_status(TaskStatus.GENERATING)
            review, review_sections = self._generate_review(fetch_results)

            # 5. Extract antibodies ───────────────────────────────────────────
            antibodies = self._extract_antibodies(review)

            # 5.5 Build knowledge graph ───────────────────────────────────────
            graph = BioVoiceGraph()
            graph.build_from_corpus(all_articles)
            graph.build_from_antibodies(antibodies)

            # 6. Build outputs ────────────────────────────────────────────────
            task.set_status(TaskStatus.OUTPUT)
            outputs: Dict = {"review": review, "antibodies": antibodies}

            if "word" in output_types:
                import os
                from biovoice.output.word import render_review_word_doc
                os.makedirs(self.config.get("output_dir", "./output"), exist_ok=True)
                word_path = os.path.join(
                    self.config.get("output_dir", "./output"), "review.docx"
                )
                section_order  = list(SECTION_QUERIES.keys())
                section_titles = {
                    "problem":    "Problem",
                    "motivation": "Motivation & Rationale",
                    "results":    "Key Results & Broadly Neutralizing Antibodies",
                    "mechanisms": "Mechanisms of Action",
                    "challenges": "Technical Challenges",
                    "future":     "Future Directions",
                }
                outputs["word_file"] = render_review_word_doc(
                    query=query,
                    sections=review_sections,
                    section_order=section_order,
                    section_titles=section_titles,
                    output_path=word_path,
                )

            # Export knowledge graph
            output_dir = self.config.get("output_dir", "./output")
            os.makedirs(output_dir, exist_ok=True)
            graph_stats = graph.statistics()
            print(
                f"[Graph] {graph_stats['nodes']} nodes, "
                f"{graph_stats['edges']} edges"
            )
            graph_graphml = os.path.join(output_dir, "knowledge_graph.graphml")
            graph_json    = os.path.join(output_dir, "knowledge_graph.json")
            try:
                graph.to_graphml(graph_graphml)
                graph.to_json(graph_json)
                outputs["graph_graphml"] = graph_graphml
                outputs["graph_json"]    = graph_json
                outputs["graph_stats"]   = graph_stats
            except Exception as e:
                print(f"[Graph] Export failed: {e}")

            if "ppt" in output_types:
                outputs["ppt_file"] = self._build_ppt(
                    review, antibodies, all_articles, query,
                    review_sections=review_sections,
                    graph=graph,
                )

            if "video" in output_types and outputs.get("ppt_file"):
                slide_imgs = self._export_ppt_to_images(
                    outputs["ppt_file"],
                    self.config.get("output_dir", "./output"),
                )
                if slide_imgs:
                    outputs["video_file"] = await self._build_video(
                        review, slide_imgs
                    )

            task.set_status(TaskStatus.DONE)
            outputs["task"] = task.to_dict()
            return outputs

        except Exception as exc:
            task.set_status(TaskStatus.FAILED, error=str(exc))
            raise

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _fetch_all(
        self,
        query:       str,
        agent_names: List[str],
        progress_cb: Optional[Callable],
    ) -> List[FetchResult]:
        """Fetch from all requested agents concurrently."""
        agents_cfg = self.config.get("agents", {})
        agents = AgentRegistry.build_from_config(
            {name: agents_cfg.get(name, {"enabled": True}) for name in agent_names}
        )
        limit = self.config.get("max_papers_per_agent", 100)

        async def _one(agent):
            print(f"[Orchestrator] Fetching: {agent.name}")
            result = await agent.fetch(query, limit=limit)
            print(f"[Orchestrator] {agent.name}: {result.count} items")
            if progress_cb:
                progress_cb(agent.name, 1, len(agents))
            return result

        results = await asyncio.gather(*[_one(a) for a in agents], return_exceptions=True)
        return [r for r in results if isinstance(r, FetchResult)]

    def _generate_review(self, fetch_results: List[FetchResult]):
        """Section-by-section RAG-driven review generation (standard mode).

        Returns (review_str, sections_dict) where review_str is the full
        markdown text and sections_dict maps section keys to prose strings.
        """
        if self.rag is None:
            raise RuntimeError(
                "_generate_review requires RAG. Use _generate_grant_sections for grant mode."
            )
        sections: Dict[str, str] = {}
        for section_name, section_query in SECTION_QUERIES.items():
            print(f"  [Review] Generating: {section_name}")
            docs = self.rag.similarity_search(section_query, k=8)
            context = "\n\n".join(
                f"[PMID: {d.metadata.get('pmid','N/A')}] {d.page_content}"
                for d in docs
            )
            instruction = SECTION_INSTRUCTIONS.get(section_name, "")
            prior = ""
            if sections:
                recent = list(sections.items())[-2:]
                prior = "\n\n".join(
                    f"[{k}]:\n{v[:400]}..." for k, v in recent
                )

            system = (
                "You are a senior biomedical researcher writing a peer-reviewed "
                "literature review. Be analytical, compare studies, identify "
                "contradictions, and cite PMIDs for every factual claim."
            )
            human_parts = [f"TASK:\n{instruction}\n"]
            if prior:
                human_parts.append(f"PRIOR SECTIONS:\n{prior}\n")
            human_parts.append(
                f"LITERATURE CONTEXT:\n{context}\n\nWrite the section now:"
            )
            sections[section_name] = self.model.chat(
                system, "\n".join(human_parts)
            )

        order = {
            "problem":    "## Problem",
            "motivation": "## Motivation & Rationale",
            "results":    "## Key Results & Broadly Neutralizing Antibodies",
            "mechanisms": "## Mechanisms of Action",
            "challenges": "## Technical Challenges",
            "future":     "## Future Directions",
        }
        review_str = "\n\n---\n\n".join(
            f"{heading}\n\n{sections[key]}"
            for key, heading in order.items()
            if key in sections
        )
        return review_str, sections

    def _generate_grant_sections(
        self,
        all_articles: List[Dict],
    ) -> "GrantOutput":
        """
        Grant mode: synthesize NIH grant sections from abstracts only.
        Each section is citation-verified before being added to GrantOutput.
        """
        from biovoice.core.grant_config import GrantConfig, GrantOutput, GrantSection, Citation
        from domain.virology.prompts.pmrc_templates import (
            GRANT_SYSTEM_PROMPT,
            GRANT_SYNTHESIS_PROMPT,
            SECTION_TITLES,
        )

        gc: GrantConfig = self.grant_config
        output = GrantOutput(research_question=gc.research_question)

        # Build a flat context string per paper (abstract only, max 250 tokens approx)
        def paper_context(p: Dict) -> str:
            pmid  = p.get("pmid") or p.get("PMID") or "N/A"
            title = (p.get("title") or "")[:120]
            abst  = (p.get("abstract") or "")[:1200]
            return f"[PMID:{pmid}] {title}\n{abst}"

        all_pmids = {str(p.get("pmid") or p.get("PMID") or "") for p in all_articles}

        for key in gc.section_keys:
            print(f"  [Grant] Generating section: {key}")
            section_instruction = (gc.section_instructions or {}).get(key, "")
            title = SECTION_TITLES.get(key, key.replace("_", " ").title())

            # Pull papers relevant to this section (simple keyword filter)
            section_query = (gc.section_queries or {}).get(key, gc.research_question)
            query_tokens = set(section_query.lower().split())
            relevant = [
                p for p in all_articles
                if query_tokens & _token_set(
                    (p.get("title") or "") + " " + (p.get("abstract") or ""),
                    set()
                )
            ][:20] or all_articles[:20]

            context = "\n\n".join(paper_context(p) for p in relevant)

            prompt = GRANT_SYNTHESIS_PROMPT.format(
                section_title=title,
                research_question=gc.research_question,
                section_instruction=section_instruction,
                context=context,
            )

            raw_text = self.model.chat(GRANT_SYSTEM_PROMPT, prompt)

            verified_text, cite_warnings = verify_citations(
                raw_text,
                all_articles,
                jaccard_threshold=gc.jaccard_threshold,
            )

            # Build Citation objects for papers cited in this section
            cited_pmids = re.findall(r'\[(\d+)\]', verified_text)
            # Map back: build pmid→number from verify_citations output
            # (re-extract to build Citation list)
            pmid_map: Dict[str, Dict] = {
                str(p.get("pmid") or p.get("PMID") or ""): p
                for p in all_articles
            }
            section_citations: List[Citation] = []
            seen: set = set()
            for pmid_raw in _extract_cite_pmids(raw_text):
                if pmid_raw in seen:
                    continue
                seen.add(pmid_raw)
                paper = pmid_map.get(pmid_raw, {})
                susp = any(pmid_raw in w for w in cite_warnings)
                section_citations.append(Citation(
                    pmid=pmid_raw,
                    title=(paper.get("title") or "")[:100],
                    authors=_format_authors(paper),
                    journal=paper.get("journal") or paper.get("source") or "",
                    year=str(paper.get("year") or paper.get("pub_year") or ""),
                    doi=paper.get("doi") or "",
                    suspicious=susp,
                    warning=next((w for w in cite_warnings if pmid_raw in w), ""),
                ))

            section = GrantSection(
                key=key,
                title=title,
                text=verified_text,
                citations=section_citations,
                warnings=cite_warnings,
            )
            output.sections.append(section)
            output.citation_warnings.extend(cite_warnings)

            if cite_warnings:
                print(f"  [Grant] {len(cite_warnings)} citation warning(s) in {key}")

        # Deduplicated full citation list across all sections
        seen_pmids: set = set()
        for section in output.sections:
            for c in section.citations:
                if c.pmid not in seen_pmids:
                    output.all_citations.append(c)
                    seen_pmids.add(c.pmid)

        return output

    def _extract_antibodies(self, text: str) -> List[Dict]:
        schema_str = json.dumps(antibody_schema, indent=2)
        system = (
            "You are an expert in virology and antibody research. "
            "Return ONLY valid JSON — no markdown fences, no preamble."
        )
        human = "\n".join([
            "Extract all broadly neutralizing antibodies mentioned in the text.",
            'Return JSON: {"antibodies": [...]}',
            "Schema:", schema_str, "", "Text:", text, "",
            "Return only the JSON object:",
        ])
        raw = self.model.chat(system, human)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data = json.loads(raw)
            result = [
                ab for ab in data.get("antibodies", [])
                if isinstance(ab, dict) and ab.get("antibody_name")
            ]
            if not result and "antibodies" in data:
                print("[Orchestrator] _extract_antibodies: JSON parsed but antibodies list is empty")
            return result
        except json.JSONDecodeError as exc:
            print(
                f"[Orchestrator] _extract_antibodies: malformed JSON from LLM "
                f"({exc}). Raw response starts with: {raw[:120]!r}"
            )
            return []

    def _build_ppt(
        self,
        review:          str,
        antibodies:      List[Dict],
        articles:        List[Dict],
        query:           str,
        review_sections: Optional[Dict] = None,
        graph:           Optional["BioVoiceGraph"] = None,
    ) -> str:
        from core.presentation.ppt_generator import PPTGenerator

        gen = PPTGenerator()
        ft  = sum(1 for a in articles if a.get("fulltext_available"))

        # Slide 1 — Title
        gen.add_title_slide(
            "BioVoice-Agents\nLiterature Review",
            query[:120],
        )

        # Slide 2 — Corpus overview
        source_counts = {}
        for a in articles:
            src = a.get("_agent_source") or a.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        gen.add_content_slide(
            "Literature Corpus",
            [
                f"Articles indexed: {len(articles)}",
                f"Full-text available: {ft}",
                f"Abstract only: {len(articles) - ft}",
            ] + [f"{src}: {n}" for src, n in
                 sorted(source_counts.items(), key=lambda x: -x[1])[:5]],
            subtitle="Source breakdown",
        )

        # Slide 3 — Strain surveillance chart (from FluNet agent if present)
        try:
            from core.presentation.visualizer import create_strain_bar_chart
            flunet_items = [a for a in articles
                            if a.get("_agent_source") == "flunet"
                            or a.get("source") == "flunet"]
            if flunet_items:
                # Aggregate subtype counts across all FluNet records
                from collections import defaultdict
                subtypes: dict = defaultdict(int)
                key_map = {
                    "H1N1pdm09": ["ah1n1_2009"],
                    "H3N2":      ["ah3"],
                    "H5":        ["ah5"],
                    "B/Victoria":["b_victoria"],
                    "B/Yamagata":["b_yamagata"],
                }
                for item in flunet_items:
                    for label, keys in key_map.items():
                        for k in keys:
                            try:
                                subtypes[label] += int(item.get(k) or 0)
                            except (ValueError, TypeError):
                                pass
                nonempty = {k: v for k, v in subtypes.items() if v > 0}
                if nonempty:
                    img = create_strain_bar_chart(
                        nonempty,
                        title="Influenza Strain Distribution",
                        subtitle="WHO FluNet global surveillance",
                    )
                    gen.add_chart_slide(
                        "Circulating Strain Landscape",
                        img,
                        caption="Source: WHO FluNet — aggregated strain surveillance data",
                    )
        except Exception as e:
            print(f"[PPT] Strain chart skipped: {e}")

        # Slides 3–8 — one prose slide per review section
        section_order = ["problem", "motivation", "results",
                         "mechanisms", "challenges", "future"]
        section_titles = {
            "problem":    "Problem",
            "motivation": "Motivation & Rationale",
            "results":    "Key Results",
            "mechanisms": "Mechanisms of Action",
            "challenges": "Technical Challenges",
            "future":     "Future Directions",
        }
        sections = review_sections or {}
        for key in section_order:
            text = sections.get(key, "")
            if not text:
                continue
            title = section_titles.get(key, key.title())
            # Split long section across two slides if needed
            chunks = [text[i:i+900] for i in range(0, len(text), 900)]
            for idx, chunk in enumerate(chunks[:2]):
                slide_title = title if idx == 0 else f"{title} (cont.)"
                gen.add_prose_slide(slide_title, chunk)

        # Publication trend chart
        try:
            from core.presentation.visualizer import create_publication_trend
            from collections import defaultdict
            year_counts: dict = defaultdict(int)
            for a in articles:
                yr = str(a.get("year", "")).strip()
                if yr.isdigit() and 1990 <= int(yr) <= 2030:
                    year_counts[int(yr)] += 1
            if len(year_counts) >= 3:
                img = create_publication_trend(
                    dict(year_counts),
                    title="Publication Volume by Year",
                    highlight_years=[2009, 2020],  # H1N1 pandemic, COVID
                )
                gen.add_chart_slide(
                    "Research Momentum",
                    img,
                    caption="Publications per year across indexed databases",
                )
        except Exception as e:
            print(f"[PPT] Trend chart skipped: {e}")

        # Knowledge graph summary slide
        if graph is not None:
            try:
                stats = graph.statistics()
                top_epitopes = graph.query_top_targeted_epitopes(n=5)
                bullets = [
                    f"Nodes: {stats['nodes']}  |  Edges: {stats['edges']}",
                    f"Antibody nodes: {stats.get('antibody_nodes', 0)}",
                    f"Virus/subtype nodes: {stats.get('virus_nodes', 0)}",
                    f"Publication nodes: {stats.get('publication_nodes', 0)}",
                ]
                if top_epitopes:
                    bullets.append(
                        "Top epitopes: "
                        + ", ".join(
                            f"{e['epitope']} ({e['antibody_count']})"
                            for e in top_epitopes[:3]
                        )
                    )
                gen.add_content_slide(
                    "Knowledge Graph",
                    bullets,
                    subtitle="Antibody-antigen-publication network",
                )
                # Subgraph visualisation (if networkx + matplotlib available)
                img_bytes = graph.plot_subgraph(max_nodes=40)  # entity_name=None → top ab
                if img_bytes:
                    gen.add_chart_slide(
                        "Knowledge Graph — Top Antibody Network",
                        img_bytes,
                        caption=(
                            f"{stats['nodes']} nodes, {stats['edges']} edges. "
                            "Exported to knowledge_graph.graphml"
                        ),
                    )
            except Exception as e:
                print(f"[PPT] Knowledge graph slide skipped: {e}")

        # Antibody table slide
        if antibodies:
            valid = [ab for ab in antibodies if isinstance(ab, dict)][:14]
            if valid:
                headers = ["Antibody", "Target", "Epitope", "Gene", "Breadth", "Phase"]
                rows = [[
                    ab.get("antibody_name", ""),
                    ab.get("target_protein", ""),
                    ab.get("epitope_region", "")[:30],
                    ab.get("gene_usage", ""),
                    ab.get("neutralization_spectrum", "")[:30],
                    ab.get("clinical_phase", ""),
                ] for ab in valid]
                gen.add_table_slide("Broadly Neutralizing Antibodies", headers, rows)

        output_dir = self.config.get("output_dir", "./output")
        ppt_file   = os.path.join(output_dir, "biovoice_output.pptx")
        os.makedirs(output_dir, exist_ok=True)
        gen.save(ppt_file)
        print(f"[PPT] Saved: {ppt_file}")
        return ppt_file

    def _export_ppt_to_images(self, pptx_path: str, output_dir: str) -> List[str]:
        import glob
        import subprocess
        import tempfile
        if not os.path.exists(pptx_path):
            return []
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "pdf",
                     "--outdir", tmpdir, pptx_path],
                    check=True, capture_output=True, text=True,
                )
                pdfs = glob.glob(os.path.join(tmpdir, "*.pdf"))
                if not pdfs:
                    return []
                from pdf2image import convert_from_path
                images = convert_from_path(pdfs[0], dpi=200)
                os.makedirs(output_dir, exist_ok=True)
                paths = []
                for i, img in enumerate(images):
                    p = os.path.join(output_dir, f"slide_{i+1:03d}.png")
                    img.save(p, "PNG")
                    paths.append(p)
                return paths
            except Exception as e:
                print(f"[Orchestrator] PPT→images failed: {e}")
                return []

    async def run_grant(self, grant_config) -> "GrantOutput":
        """
        Grant mode entry point.

        Runs parallel agent fetch (pubmed + europe_pmc + uniprot),
        ranks and deduplicates papers, synthesizes NIH grant sections,
        verifies citations, and renders Word + PPT outputs.

        Parameters
        ----------
        grant_config : GrantConfig
        """
        from biovoice.core.grant_config import GrantConfig
        from biovoice.output.word import render_word_doc
        from core.presentation.ppt_generator import render_grant_ppt

        gc: GrantConfig = grant_config
        self.grant_config = gc

        agent_names = ["pubmed", "europe_pmc", "uniprot"]
        fetch_results = await self._fetch_all(gc.research_question, agent_names, None)

        # Merge + deduplicate by PMID; tag each item with its agent source
        seen: set = set()
        merged: List[Dict] = []
        for fr in fetch_results:
            for item in fr.items:
                pmid = str(item.get("pmid") or item.get("PMID") or "")
                key  = pmid or (item.get("doi") or item.get("title") or id(item))
                if key and key not in seen:
                    seen.add(key)
                    item = dict(item)          # don't mutate the agent's original
                    item.setdefault("_agent_source", fr.source)
                    merged.append(item)

        # Rank: recency + citation count + domain relevance
        from datetime import date
        current_year = date.today().year
        max_cites = max((int(p.get("citation_count") or 0) for p in merged), default=1) or 1

        query_tokens = _token_set(gc.research_question, set())
        from domain.virology.schemas.antibody_schema import antibody_schema
        schema_tokens = _token_set(" ".join(antibody_schema.keys()), set())
        domain_vocab  = query_tokens | schema_tokens

        def rank_score(p: Dict) -> float:
            year = int(p.get("year") or p.get("pub_year") or current_year)
            recency  = max(0.0, 1.0 - (current_year - year) / 20.0)
            cites    = int(p.get("citation_count") or 0) / max_cites
            paper_tokens = _token_set(
                (p.get("title") or "") + " " + (p.get("abstract") or ""), set()
            )
            relevance = _jaccard(paper_tokens, domain_vocab)
            return 0.5 * recency + 0.3 * cites + 0.2 * relevance

        # Cap UniProt entries at 3 so protein-DB records don't crowd out papers.
        # Strategy: sort all, then walk the list filling a capped bucket.
        MAX_UNIPROT = 3
        all_sorted = sorted(merged, key=rank_score, reverse=True)
        ranked: List[Dict] = []
        uniprot_count = 0
        for p in all_sorted:
            if len(ranked) >= gc.max_ranked_papers:
                break
            if p.get("_agent_source") == "uniprot":
                if uniprot_count >= MAX_UNIPROT:
                    continue
                uniprot_count += 1
            ranked.append(p)

        print(
            f"[Grant] {len(merged)} papers after dedup, "
            f"{len(ranked)} after ranking (top {gc.max_ranked_papers})"
        )

        grant_output = self._generate_grant_sections(ranked)

        # Render outputs
        os.makedirs(gc.output_dir, exist_ok=True)
        word_path = os.path.join(gc.output_dir, "grant_specific_aims.docx")
        ppt_path  = os.path.join(gc.output_dir, "grant_slides.pptx")

        render_word_doc(grant_output, word_path)
        render_grant_ppt(grant_output, ppt_path)

        grant_output.word_file = word_path
        grant_output.ppt_file  = ppt_path

        return grant_output

    async def _build_video(
        self, review: str, slide_images: List[str]
    ) -> Optional[str]:
        from moviepy.editor import AudioFileClip
        from core.presentation.speech_synthesizer import SpeechSynthesizer
        if self.tts is None:
            self.tts = SpeechSynthesizer()
        script = review.replace("\n", " ").strip()
        if not script or not slide_images:
            return None
        output_dir  = self.config.get("output_dir", "./output")
        audio_path  = os.path.join(output_dir, "audio.mp3")
        video_path  = os.path.join(output_dir, "biovoice_presentation.mp4")
        os.makedirs(output_dir, exist_ok=True)
        await self.tts.synthesize(script, audio_path)
        try:
            audio    = AudioFileClip(audio_path)
            duration = audio.duration
            audio.close()
        except Exception as e:
            print(f"[Orchestrator] Audio read failed: {e}")
            return None
        n       = len(slide_images)
        timings = [i * duration / n for i in range(n)]
        self.tts.create_video(
            audio_path=audio_path,
            slide_images=slide_images,
            timings=timings,
            output_video=video_path,
        )
        return video_path