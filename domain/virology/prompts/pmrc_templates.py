"""
domain/virology/prompts/pmrc_templates.py
NIH grant synthesis prompt templates for the Virology Grant Copilot.

Every template enforces [CITE:PMID] markers so the citation verifier
in the grant pipeline can check every claim against the source abstract.
"""

# ── System persona ─────────────────────────────────────────────────────────────

GRANT_SYSTEM_PROMPT = (
    "You are a senior virologist and expert NIH grant writer helping a postdoc "
    "prepare the Research Strategy section of an R01 application. "
    "Write in formal scientific prose appropriate for peer review. "
    "Every factual claim MUST be followed by a citation marker in the form "
    "[CITE:PMID] where PMID is the exact PubMed ID from the provided literature. "
    "Do NOT invent PMIDs. Do NOT make claims without a citation marker. "
    "A claim with no [CITE:PMID] will be flagged as unsupported and removed."
)


# ── Section-level synthesis prompt ────────────────────────────────────────────

GRANT_SYNTHESIS_PROMPT = """\
You are writing the {section_title} section of an NIH R01 grant application
on the following research question:

  {research_question}

Use ONLY the literature provided below. For every factual claim, append a
[CITE:PMID] marker using the exact PMID shown in the source entry. Do not
combine multiple PMIDs into one marker — use separate markers per claim.
Do not cite a paper for a claim it does not support.

{section_instruction}

--- LITERATURE ---
{context}
--- END LITERATURE ---

Write the section now (no section header, no preamble):"""


# ── Per-section instructions ───────────────────────────────────────────────────

SECTION_INSTRUCTIONS = {
    "significance": (
        "Write the Significance section (150-200 words).\n"
        "- Open with the clinical or scientific problem. Quantify it (incidence, "
        "burden, unmet need) with specific numbers from the literature [CITE:PMID].\n"
        "- Explain why existing approaches fall short, citing limitations of "
        "current vaccines or therapies [CITE:PMID].\n"
        "- State why the proposed approach is a meaningful advance.\n"
        "- Every factual sentence needs a [CITE:PMID]. No vague generalities."
    ),
    "innovation": (
        "Write the Innovation section (100-150 words).\n"
        "- Identify 2-3 specific ways the proposed work is novel compared to "
        "published literature [CITE:PMID].\n"
        "- Do NOT claim novelty for things that already exist in the papers.\n"
        "- Innovation can be: new epitope target, new structural insight, new "
        "engineering approach, or new combination of existing tools.\n"
        "- Keep it to what the evidence actually supports."
    ),
    "specific_aims": (
        "Write the Specific Aims section (200-250 words) structured as:\n"
        "  1. Opening paragraph: problem statement and significance [CITE:PMID].\n"
        "  2. Overall hypothesis (1 sentence, no citation needed).\n"
        "  3. Aim 1 (one sentence, bold): [brief mechanistic rationale [CITE:PMID]].\n"
        "  4. Aim 2 (one sentence, bold): [brief mechanistic rationale [CITE:PMID]].\n"
        "  5. Aim 3 optional (one sentence, bold) if evidence warrants it.\n"
        "  6. Closing sentence: expected impact.\n"
        "Keep aims tight and mechanistically grounded in the literature. "
        "Use NIH Specific Aims page style — no sub-bullets, flowing prose."
    ),
    "background": (
        "Write the Background and Preliminary Data section (300-400 words).\n"
        "- Organize around the key concepts the aims build on.\n"
        "- Compare 3-5 specific antibodies or studies: include name, epitope, "
        "neutralization breadth, IC50 range, and clinical stage where reported.\n"
        "- Note where results conflict or where gaps exist — this motivates the aims.\n"
        "- Every comparative claim needs [CITE:PMID].\n"
        "- Do NOT list papers — synthesize them into a narrative."
    ),
}


# ── Section queries (used to pull relevant abstracts per section) ──────────────

SECTION_QUERIES = {
    "significance": (
        "influenza pandemic burden vaccine mismatch unmet clinical need mortality"
    ),
    "innovation": (
        "broadly neutralizing antibody novel epitope approach engineering germline"
    ),
    "specific_aims": (
        "broadly neutralizing antibody influenza hemagglutinin stem neutralization "
        "spectrum preclinical clinical"
    ),
    "background": (
        "broadly neutralizing antibody IC50 neutralization breadth IGHV gene "
        "effector function clinical stage"
    ),
}


# ── Section display titles (used in rendered output) ──────────────────────────

SECTION_TITLES = {
    "significance":   "Significance",
    "innovation":     "Innovation",
    "specific_aims":  "Specific Aims",
    "background":     "Background and Preliminary Data",
}

# Biomedical stoplist — filtered out before Jaccard citation verification.
# These words appear in almost every abstract and would inflate similarity
# scores for unrelated claims.
BIOMEDICAL_STOPLIST = {
    "the", "a", "an", "and", "or", "in", "of", "to", "with", "for", "by",
    "was", "were", "is", "are", "be", "been", "being", "has", "have", "had",
    "that", "this", "these", "those", "which", "from", "as", "on", "at",
    "we", "our", "their", "its",
    # biomedical near-universals
    "cells", "protein", "proteins", "antibody", "antibodies", "virus", "viral",
    "human", "mouse", "mice", "study", "studies", "using", "results", "showed",
    "significant", "data", "analysis", "expression", "activity", "found",
    "increased", "decreased", "response", "responses", "model", "models",
    "against", "may", "can", "also", "both", "however", "therefore",
    "suggest", "suggests", "suggested", "show", "shows", "shown",
}
