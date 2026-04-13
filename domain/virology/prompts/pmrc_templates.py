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
        "\n"
        "STRICT RULES — violating any of these will cause the section to be rejected:\n"
        "- Do NOT open with 'Influenza remains...', 'X is a significant threat...', "
        "or any generic epidemiology sentence. Open instead with a specific named "
        "problem: a gap, a failure, or a mechanism that is unresolved.\n"
        "- Include at least one specific quantified statistic (deaths, cases, IC50, "
        "efficacy percentage) pulled verbatim or closely paraphrased from an abstract, "
        "with [CITE:PMID] using the PMID of the paper that contains that number.\n"
        "- Name at least one specific bnAb (e.g. CR9114, FI6v3, MEDI8852) and state "
        "exactly why it falls short — resistance mutation, manufacturing limitation, "
        "narrow breadth — with [CITE:PMID].\n"
        "- State one specific mechanistic reason the proposed approach addresses that gap.\n"
        "- Every factual sentence needs [CITE:PMID]. No generalities. No vague 'studies show'."
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
        "Write the Specific Aims section (200-250 words) in NIH R01 style.\n"
        "\n"
        "STRICT RULES — violating any of these will cause the section to be rejected:\n"
        "- Do NOT open with 'Influenza remains a significant threat' or any generic "
        "sentence. Open with a specific unresolved mechanistic or clinical problem "
        "grounded in the literature.\n"
        "- The opening paragraph MUST name at least one specific bnAb (e.g. CR9114, "
        "FI6v3, CR6261) or one specific epitope region (HA stem, receptor binding site, "
        "fusion peptide) and state the precise gap that motivates this work [CITE:PMID].\n"
        "- Overall hypothesis: one sentence naming the specific molecule, epitope, or "
        "mechanism being tested — not a generic 'broadly neutralizing antibodies will work'.\n"
        "- Each Aim: one bold sentence naming the specific experimental system, molecule, "
        "or readout (IC50, neutralization breadth, ADCC activity, crystal structure) "
        "and its mechanistic rationale [CITE:PMID]. No 'we will investigate X'.\n"
        "- Closing sentence: expected impact stated in terms of the specific gap identified "
        "in the opening.\n"
        "Use flowing prose, no sub-bullets."
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
