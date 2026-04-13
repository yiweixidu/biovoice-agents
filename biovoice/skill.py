"""
biovoice/skill.py
FluBroad Skill manifest for the BioVoice Virology package.

This file registers the Virology Skill with the FluBroad framework.
When ``flubroad-skill-virology`` is installed, the framework discovers
this manifest via the ``flubroad.skills`` entry point and uses it to
configure the full pipeline (agents, section prompts, extraction schema,
knowledge graph patterns) for virology and bnAb research.

Entry point (pyproject.toml):
    [project.entry-points."flubroad.skills"]
    virology = "biovoice.skill:manifest"

Usage (once flubroad framework is installed):
    from flubroad.skill import SkillLoader
    skill = SkillLoader.load("virology")
    print(skill.display_name)

Standalone usage (current mode, before framework extraction):
    from biovoice.skill import manifest
    print(manifest.summary())
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Graceful import: SkillManifest lives in the flubroad framework package once
# it is extracted. Until then we fall back to a local shim so biovoice-agents
# continues to work as a standalone repo.
# ---------------------------------------------------------------------------
try:
    from flubroad.skill import SkillManifest          # future: framework package
except ImportError:
    from biovoice._skill_shim import SkillManifest    # current: local shim

from domain.virology.schemas.antibody_schema import antibody_schema

# ── Section queries ────────────────────────────────────────────────────────────
# Keyword-dense strings sent to the RAG vector store, one per output section.
# Optimised for semantic similarity against virology / influenza bnAb abstracts.

SECTION_QUERIES = {
    "problem": (
        "antigenic drift shift influenza vaccine mismatch pandemic"
    ),
    "motivation": (
        "conserved epitope HA stem stalk broadly neutralizing antibody strategy"
    ),
    "results": (
        "broadly neutralizing antibody influenza neutralization spectrum IC50 breadth"
    ),
    "mechanisms": (
        "Fc effector function ADCC ADCP complement broadly neutralizing antibody mechanism"
    ),
    "challenges": (
        "immunodominance germline targeting epitope accessibility escape challenge"
    ),
    "future": (
        "mRNA vaccine immunogen design deep mutational scanning universal influenza"
    ),
}

# ── Section instructions ───────────────────────────────────────────────────────
# LLM synthesis prompts. Every instruction mandates PMID citations and
# specifies a word-count target. Keys must match SECTION_QUERIES.

SECTION_INSTRUCTIONS = {
    "problem": (
        "Write the 'Problem' section of a virology literature review.\n"
        "- Describe the clinical challenge: antigenic drift, vaccine mismatch, pandemic risk.\n"
        "- Quantify with specific statistics from the provided context (mortality, incidence).\n"
        "- Contrast strain-specific immunity with the need for broad protection.\n"
        "- Every factual claim MUST include a PMID in parentheses.\n"
        "- Length: 150-200 words. No section header."
    ),
    "motivation": (
        "Write the 'Motivation' section.\n"
        "- Explain WHY targeting conserved epitopes (HA stalk, stem, RBS) is promising.\n"
        "- Critically compare to strain-specific vaccination approaches.\n"
        "- Reference germline gene usage patterns (e.g. IGHV1-69) where relevant.\n"
        "- Every claim MUST include a PMID.\n"
        "- Length: 150-200 words. No header."
    ),
    "results": (
        "Write the 'Key Results' section.\n"
        "- Compare at least 4-5 specific broadly neutralizing antibodies by name.\n"
        "- For each include: name, epitope, IGHV gene, neutralization breadth, IC50, clinical stage.\n"
        "- Synthesise patterns across antibodies — do NOT just list them.\n"
        "- Note structural data (PDB IDs) where available.\n"
        "- Every claim MUST include a PMID. Length: 300-400 words. No header."
    ),
    "mechanisms": (
        "Write the 'Mechanisms' section.\n"
        "- Describe Fc effector functions: ADCC, ADCP, complement activation.\n"
        "- Cite specific Fc-knockout or Fc-engineering experiments.\n"
        "- Note any controversies about the relative contribution of neutralisation vs. Fc.\n"
        "- Every claim MUST include a PMID.\n"
        "- Length: 150-200 words. No header."
    ),
    "challenges": (
        "Write the 'Technical Challenges' section.\n"
        "- Address at least 3: immunodominance, stem accessibility, germline requirements, "
        "mutational escape, manufacturing, delivery.\n"
        "- For each: cite evidence AND describe current approaches to overcome it.\n"
        "- Be explicit about which challenges remain unsolved.\n"
        "- Every claim MUST include a PMID.\n"
        "- Length: 200-250 words. No header."
    ),
    "future": (
        "Write the 'Future Directions' section.\n"
        "- Propose 3-4 SPECIFIC, mechanistically justified directions.\n"
        "- Ground each in a gap identified earlier in the review.\n"
        "- Do NOT write 'further research is needed' — give concrete next steps.\n"
        "- Cite supporting evidence for feasibility where available.\n"
        "- Length: 200-250 words. No header."
    ),
}

# ── Grant mode section templates ───────────────────────────────────────────────

GRANT_TEMPLATES = {
    "specific_aims": (
        "Write a 1-page NIH Specific Aims section for a grant studying broadly "
        "neutralizing antibodies against influenza.\n"
        "Structure:\n"
        "  Para 1 (2-3 sentences): clinical problem and current gap.\n"
        "  Para 2 (1-2 sentences): central hypothesis.\n"
        "  Aim 1: [title] — hypothesis, approach, expected outcome.\n"
        "  Aim 2: [title] — hypothesis, approach, expected outcome.\n"
        "  Aim 3 (optional): [title] — hypothesis, approach, expected outcome.\n"
        "  Closing (1-2 sentences): impact and innovation.\n"
        "Cite every factual claim with [PMID]. Target: ~450 words."
    ),
    "significance": (
        "Write the 'Significance' section of an NIH Research Strategy.\n"
        "- Open with the clinical and scientific gap.\n"
        "- Quantify the problem (incidence, mortality, economic burden).\n"
        "- Explain how this work advances state of knowledge.\n"
        "- Cite every claim with [PMID]. ~400 words."
    ),
    "innovation": (
        "Write the 'Innovation' section of an NIH Research Strategy.\n"
        "- State clearly what is novel vs. prior art.\n"
        "- Reference specific existing antibodies/approaches this work improves upon.\n"
        "- Avoid marketing language. Be precise. Cite [PMID]. ~250 words."
    ),
    "approach": (
        "Write the 'Approach' section for Aim 1 of an NIH Research Strategy.\n"
        "- Describe experimental design, methods, and controls.\n"
        "- Anticipate potential pitfalls and propose alternatives.\n"
        "- Ground every methodological choice in cited precedent [PMID]. ~600 words."
    ),
}

# ── Topic keywords ─────────────────────────────────────────────────────────────
# Used to score corpus item relevance (Jaccard bag-of-words filter).
# Covers influenza virology, bnAbs, structural immunology, and related areas.

TOPIC_KEYWORDS = [
    # Pathogens
    "influenza", "influenza a", "influenza b", "h1n1", "h3n2", "h5n1",
    "sars-cov-2", "coronavirus", "rsv", "hiv",
    # Proteins
    "hemagglutinin", "neuraminidase", "ha stalk", "ha stem", "ha head",
    "nucleoprotein", "matrix protein",
    # Antibody biology
    "bnab", "broadly neutralizing", "broadly reactive", "antibody",
    "monoclonal antibody", "igg", "fab", "fc", "cdr3",
    "ighv1-69", "ighv", "iglv", "germline",
    # Immunology concepts
    "epitope", "stem", "stalk", "receptor binding site", "fusion peptide",
    "neutralization", "neutralisation", "adcc", "adcp",
    "effector function", "protection",
    # Drug/vaccine development
    "vaccine", "immunogen", "adjuvant", "clinical trial",
    "phase i", "phase ii", "prophylaxis",
]

# ── Knowledge graph entity patterns ───────────────────────────────────────────
# Regex patterns for FluBroadGraph entity extraction from abstracts.

KNOWLEDGE_GRAPH_CONFIG = {
    "node_types": {
        "Antibody":    "Antibody",
        "Epitope":     "Epitope",
        "Virus":       "Virus",
        "Publication": "Publication",
        "Target":      "Target",
    },
    "edge_types": {
        "TARGETS":    "Antibody → Target",
        "BINDS":      "Antibody → Epitope",
        "NEUTRALISES":"Antibody → Virus",
        "CITED_IN":   "Antibody → Publication",
    },
    "entity_patterns": {
        # Named bnAbs (expand as new antibodies are published)
        "antibody": (
            r"\b(CR6261|CR9114|FI6[v\d]*|MEDI8852|VIS410|CT-P27|"
            r"[A-Z]{2,4}-[0-9]{3,6}|[A-Z]{1,4}\d{3,5}[a-zA-Z]?\d*)\b"
        ),
        # Influenza subtypes and lineages
        "virus": (
            r"\b(H\d{1,2}N\d{1,2}|H1N1pdm09|H5N1|H3N2|"
            r"B/Victoria|B/Yamagata|influenza [AB])\b"
        ),
        # Structural / functional epitope regions
        "epitope": (
            r"\b(HA stalk|HA stem|HA head|RBS|receptor.binding.site|"
            r"fusion peptide|trimer interface|group [12] conserved|"
            r"[Ss]ite [IVX]+)\b"
        ),
    },
}

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior virologist and structural immunologist specialising in "
    "broadly neutralising antibodies against influenza and related respiratory "
    "pathogens. Write rigorous, analytical synthesis sections for a peer-reviewed "
    "literature review. Compare studies, identify contradictions, and cite a PMID "
    "for every factual claim. Do not speculate beyond the provided evidence."
)

# ── Fine-tuning defaults ───────────────────────────────────────────────────────

FINETUNING_CONFIG = {
    "base_model":             "unsloth/llama-3-8b-Instruct",
    "lora_rank":              16,
    "lora_alpha":             16,
    "feedback_dir":           "data/feedback",
    "min_examples_to_train":  50,
    "export_format":          "sharegpt",
    "hf_repo":                "flubroad-skill-virology/virology-lora",
}

# ── Chart config ───────────────────────────────────────────────────────────────

CHART_CONFIG = {
    "strain_chart":          True,    # FluNet strain bar chart if data present
    "trend_chart":           True,    # publication volume over time
    "heatmap_source_agent":  "",      # no heatmap source configured by default
}

# ── SkillManifest instance ─────────────────────────────────────────────────────

manifest = SkillManifest(
    # Identity
    name="virology",
    version="1.0.0",
    display_name="BioVoice: Virology & Broadly Neutralizing Antibodies",
    description=(
        "A FluBroad Skill for influenza virology, broadly neutralising antibody "
        "(bnAb) research, and universal vaccine design. Fetches from 17 databases "
        "including PubMed, Europe PMC, PDB, UniProt, AlphaFold, SAbDab, WHO FluNet, "
        "IEDB, bioRxiv, and private BCR-seq data (AIRR format). Synthesises "
        "6-section literature reviews and NIH grant sections with verified citations."
    ),

    # Agents — full list; default_agents is the fast subset
    agents=[
        "pubmed",
        "europe_pmc",
        "semantic_scholar",
        "biorxiv",
        "crossref",
        "iedb",
        "opentargets",
        "uniprot",
        "alphafold",
        "pdb",
        "pubchem",
        "flunet",
        "chembl",
        "clinicaltrials",
        "local_data",
        "maad",
        "airr",
    ],
    default_agents=[
        "pubmed",
        "europe_pmc",
        "semantic_scholar",
        "iedb",
        "uniprot",
        "pdb",
        "flunet",
        "maad",
    ],

    # Pipeline configuration
    section_queries=SECTION_QUERIES,
    section_instructions=SECTION_INSTRUCTIONS,
    extraction_schema=antibody_schema,
    topic_keywords=TOPIC_KEYWORDS,
    system_prompt=SYSTEM_PROMPT,

    # Section display
    output_section_order=["problem", "motivation", "results",
                          "mechanisms", "challenges", "future"],
    output_section_titles={
        "problem":    "Problem",
        "motivation": "Motivation & Rationale",
        "results":    "Key Results & Broadly Neutralizing Antibodies",
        "mechanisms": "Mechanisms of Action",
        "challenges": "Technical Challenges",
        "future":     "Future Directions",
    },

    # Optional features
    grant_templates=GRANT_TEMPLATES,
    knowledge_graph_config=KNOWLEDGE_GRAPH_CONFIG,
    finetuning_config=FINETUNING_CONFIG,
    chart_config=CHART_CONFIG,
)
