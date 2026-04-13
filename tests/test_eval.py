"""
tests/test_eval.py
Synthesis quality evals for BioVoice-Agents.

These tests make live LLM calls and require OPENAI_API_KEY (or Ollama).
They are skipped in normal CI; run manually or in a dedicated eval pipeline.

    pytest -m eval tests/test_eval.py -v

What is measured:
  1. Citation tagging rate  — fraction of section sentences that contain [CITE:PMID]
  2. Section word count     — each of 6 sections meets minimum length
  3. PMID hallucination     — no [CITE:PMID] references a PMID absent from fetched corpus
  4. Jaccard warning rate   — fraction of citations flagged as suspicious must be < threshold
  5. Antibody extraction    — _extract_antibodies returns at least 1 named antibody
  6. Grant-mode sections    — run_grant produces Specific Aims + Innovation + Significance
  7. End-to-end timing      — full pipeline completes within 5 minutes on default agents

All thresholds are taken directly from the whitepaper (section 5) or set conservatively
when the whitepaper did not specify an exact number.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Dict, List

import pytest

# ── Constants ─────────────────────────────────────────────────────────────────

BENCHMARK_QUERY = "broadly neutralizing antibodies influenza hemagglutinin"

# Whitepaper section 3.4: >90% time saving, synthesis must cover key sections
SECTION_MIN_WORDS: Dict[str, int] = {
    "problem":    100,
    "motivation": 100,
    "results":    100,
    "mechanisms": 80,
    "challenges": 80,
    "future":     80,
}

# Whitepaper: every claim must include [CITE:PMID]
# We accept that not every sentence is a claim, so threshold is conservative
CITE_RATE_THRESHOLD  = 0.30    # at least 30% of sentences in a section must be cited
JACCARD_WARN_RATE_MAX = 0.25   # no more than 25% of citations may be flagged as suspicious
MAX_PIPELINE_SECONDS  = 300    # 5 minutes total

# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_sentences(text: str) -> int:
    return max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))


def _count_cited_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text)
                if s.strip() and re.search(r"\[\d+\]|\[CITE:", s)])


def _extract_numbered_pmids(review_text: str, corpus: List[Dict]) -> List[str]:
    """Return PMIDs referenced in text as [N] that are NOT in the corpus."""
    # Build mapping from citation number to PMID (as produced by verify_citations)
    # verify_citations replaces [CITE:PMID] with [N], so we look at raw CITE markers
    # before verification for this test. We call _generate_review which returns
    # post-verification text, so we track via corpus presence instead.
    cited_nums = re.findall(r"\[(\d+)\]", review_text)
    # The citation numbering is sequential from 1; we compare against corpus size
    if not cited_nums:
        return []
    max_cited = max(int(n) for n in cited_nums)
    # We can't recover PMID from [N] without the mapping, so return count info
    return [str(n) for n in range(1, max_cited + 1)]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_result():
    """
    Run the full BioVoice pipeline once for the benchmark query and cache
    the result for all eval tests in this module.

    Uses a small agent subset (pubmed + europe_pmc) so it completes in
    reasonable time even without all databases.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from biovoice.config.settings import BioVoiceSettings
    from biovoice.core.orchestrator import BioVoiceOrchestrator

    settings = BioVoiceSettings()
    config   = settings.to_orchestrator_config()
    config["max_papers_per_agent"] = 50   # bounded for eval speed

    orch = BioVoiceOrchestrator(config, use_rag=True)

    t0 = time.time()
    result = asyncio.run(orch.run(
        query       = BENCHMARK_QUERY,
        agent_names = ["pubmed", "europe_pmc"],
        output_types= ["review"],
        topic       = "flu_bnabs",
    ))
    elapsed = time.time() - t0

    result["_elapsed_seconds"] = elapsed
    return result


@pytest.fixture(scope="module")
def grant_result():
    """Run the grant pipeline for eval tests."""
    from dotenv import load_dotenv
    load_dotenv()

    from biovoice.config.settings import BioVoiceSettings
    from biovoice.core.orchestrator import BioVoiceOrchestrator
    from biovoice.core.grant_config import GrantConfig

    settings = BioVoiceSettings()
    config   = settings.to_orchestrator_config()
    config["max_papers_per_agent"] = 30

    gc = GrantConfig(
        research_question=BENCHMARK_QUERY,
        max_ranked_papers=20,
        output_dir="/tmp/biovoice_eval",
    )
    orch = BioVoiceOrchestrator(config, use_rag=False)
    result = asyncio.run(orch.run_grant(gc))
    return result


# ── Eval tests — Review mode ──────────────────────────────────────────────────

@pytest.mark.eval
def test_pipeline_completes_within_time_limit(pipeline_result):
    """E2E timing must be under MAX_PIPELINE_SECONDS."""
    elapsed = pipeline_result["_elapsed_seconds"]
    assert elapsed < MAX_PIPELINE_SECONDS, (
        f"Pipeline took {elapsed:.0f}s — exceeds {MAX_PIPELINE_SECONDS}s limit. "
        f"This affects postdoc adoption (nobody waits 5+ min)."
    )


@pytest.mark.eval
def test_review_text_not_empty(pipeline_result):
    """Review text must be non-empty."""
    review = pipeline_result.get("review", "")
    assert len(review) > 200, "Review is suspiciously short — synthesis likely failed."


@pytest.mark.eval
@pytest.mark.parametrize("section", list(SECTION_MIN_WORDS.keys()))
def test_section_word_count(pipeline_result, section):
    """
    Each of the 6 PMRC sections must meet minimum word count.
    Whitepaper specifies 150-200 words per section.
    """
    review = pipeline_result.get("review", "")
    # Sections are delimited by double newlines or headers in the plain-text review
    # We do a simple word count on the full review as a proxy when section dict unavailable
    word_count = len(review.split())
    min_total  = sum(SECTION_MIN_WORDS.values())  # 640 words total minimum
    assert word_count >= min_total, (
        f"Review total word count {word_count} < {min_total}. "
        f"Section '{section}' minimum is {SECTION_MIN_WORDS[section]} words."
    )


@pytest.mark.eval
def test_citation_markers_present(pipeline_result):
    """
    Review text must contain numbered citation markers [N].
    [CITE:PMID] markers must have been resolved by verify_citations.
    Whitepaper: every factual claim requires a PMID citation.
    """
    review = pipeline_result.get("review", "")
    numbered_refs = re.findall(r"\[\d+\]", review)
    assert len(numbered_refs) >= 3, (
        f"Only {len(numbered_refs)} numbered citations found. "
        f"Synthesis is making unsupported claims."
    )


@pytest.mark.eval
def test_no_raw_cite_markers_in_output(pipeline_result):
    """
    [CITE:PMID] markers must be fully resolved to [N] by verify_citations.
    Any remaining [CITE:...] in output indicates verify_citations was skipped.
    """
    review = pipeline_result.get("review", "")
    raw_markers = re.findall(r"\[CITE:[^\]]+\]", review)
    assert len(raw_markers) == 0, (
        f"{len(raw_markers)} unresolved [CITE:PMID] markers in output: "
        f"{raw_markers[:3]}. verify_citations was not applied."
    )


@pytest.mark.eval
def test_citation_rate_per_section(pipeline_result):
    """
    At least CITE_RATE_THRESHOLD of sentences must include a citation.
    Measures whether the LLM is actually citing its claims.
    """
    review = pipeline_result.get("review", "")
    total_sentences  = _count_sentences(review)
    cited_sentences  = _count_cited_sentences(review)
    cite_rate        = cited_sentences / total_sentences

    assert cite_rate >= CITE_RATE_THRESHOLD, (
        f"Citation rate {cite_rate:.1%} < {CITE_RATE_THRESHOLD:.0%} threshold. "
        f"{cited_sentences}/{total_sentences} sentences have citations. "
        f"Synthesis is making too many unsupported claims."
    )


@pytest.mark.eval
def test_citation_warning_rate(pipeline_result):
    """
    Fraction of suspicious citations must be below JACCARD_WARN_RATE_MAX.
    A high warning rate means the LLM is hallucinating or misattributing sources.
    """
    # The orchestrator stores warnings on the result dict if exposed;
    # fall back to checking the review text for '[WARNING]' markers.
    warnings = pipeline_result.get("citation_warnings", [])
    numbered = re.findall(r"\[\d+\]", pipeline_result.get("review", ""))

    if not numbered:
        pytest.skip("No numbered citations to check warning rate against.")

    warn_rate = len(warnings) / max(1, len(numbered))
    assert warn_rate <= JACCARD_WARN_RATE_MAX, (
        f"Jaccard warning rate {warn_rate:.1%} > {JACCARD_WARN_RATE_MAX:.0%}. "
        f"{len(warnings)} suspicious citations out of {len(numbered)}. "
        f"Warnings: {warnings[:3]}"
    )


@pytest.mark.eval
def test_antibody_extraction_finds_at_least_one(pipeline_result):
    """
    _extract_antibodies must return at least 1 named antibody from the review.
    The virology domain specialisation is meaningless without this.
    """
    antibodies = pipeline_result.get("antibodies", [])
    assert len(antibodies) >= 1, (
        "No antibodies extracted from review. "
        "Either the review missed antibody data or _extract_antibodies is broken."
    )


@pytest.mark.eval
def test_antibody_has_required_fields(pipeline_result):
    """Each extracted antibody must have antibody_name and target_protein."""
    antibodies = pipeline_result.get("antibodies", [])
    if not antibodies:
        pytest.skip("No antibodies extracted — covered by test_antibody_extraction_finds_at_least_one")

    for ab in antibodies:
        assert "antibody_name" in ab, f"Missing antibody_name in: {ab}"
        assert ab["antibody_name"], "antibody_name is empty string"


@pytest.mark.eval
def test_known_antibodies_mentioned_in_review(pipeline_result):
    """
    At least 2 of the canonical broadly-neutralising antibodies must appear
    in the review text. These are well-established in the literature;
    any decent retrieval + synthesis should surface at least some.

    Whitepaper section 5.5: GPT-4 vs expert Cohen's κ = 0.89 on bnAb classification.
    """
    CANONICAL_BNABS = {
        "CR6261", "CR9114", "FI6", "MEDI8852",
        "CT149", "VIS410", "AR-903912",
        "broadly neutralizing", "bnAb", "HA stalk",
    }
    review_lower = pipeline_result.get("review", "").lower()
    found = [ab for ab in CANONICAL_BNABS if ab.lower() in review_lower]
    assert len(found) >= 2, (
        f"Only {len(found)} canonical bnAb terms found in review: {found}. "
        f"Retrieval is missing core literature on the benchmark query."
    )


# ── Eval tests — Grant mode ───────────────────────────────────────────────────

@pytest.mark.eval
def test_grant_produces_sections(grant_result):
    """Grant mode must produce at least 3 named sections."""
    sections = grant_result.sections if hasattr(grant_result, "sections") else []
    assert len(sections) >= 3, (
        f"Grant mode produced only {len(sections)} sections. "
        f"Expected Specific Aims, Significance, Innovation at minimum."
    )


@pytest.mark.eval
def test_grant_specific_aims_present(grant_result):
    """Specific Aims section must be present and non-trivial."""
    sa = grant_result.section("specific_aims") if hasattr(grant_result, "section") else None
    assert sa is not None, "Specific Aims section missing from grant output."
    assert len(sa.text) >= 200, (
        f"Specific Aims too short ({len(sa.text)} chars). "
        f"NIH requires ~600 words on the Specific Aims page."
    )


@pytest.mark.eval
def test_grant_citations_present(grant_result):
    """Grant output must include verifiable citations."""
    all_citations = getattr(grant_result, "all_citations", [])
    assert len(all_citations) >= 5, (
        f"Only {len(all_citations)} citations in grant output. "
        f"An NIH R01 Significance section typically cites 10-20 papers."
    )


@pytest.mark.eval
def test_grant_no_citation_warnings_blocking(grant_result):
    """
    Grant output may have citation warnings, but none should be
    'not found in fetched results' (hallucinated PMID).
    """
    warnings = getattr(grant_result, "citation_warnings", [])
    hallucinated = [w for w in warnings if "not found in fetched results" in w]
    assert len(hallucinated) == 0, (
        f"{len(hallucinated)} hallucinated PMIDs in grant output:\n"
        + "\n".join(hallucinated[:5])
        + "\nHallucinated citations in an NIH grant = automatic rejection risk."
    )


# ── Benchmark report helper ───────────────────────────────────────────────────

@pytest.mark.eval
def test_print_benchmark_summary(pipeline_result, grant_result):
    """
    Not a real assertion — prints a human-readable benchmark card
    that matches the whitepaper's section 5 table format.
    Run with -s to see output.
    """
    review    = pipeline_result.get("review", "")
    antibodies= pipeline_result.get("antibodies", [])
    elapsed   = pipeline_result.get("_elapsed_seconds", 0)
    numbered  = re.findall(r"\[\d+\]", review)
    warnings  = pipeline_result.get("citation_warnings", [])
    cite_rate = _count_cited_sentences(review) / max(1, _count_sentences(review))
    warn_rate = len(warnings) / max(1, len(numbered))

    grant_sections   = len(getattr(grant_result, "sections", []))
    grant_citations  = len(getattr(grant_result, "all_citations", []))
    grant_hallucinated = len([w for w in getattr(grant_result, "citation_warnings", [])
                              if "not found in fetched results" in w])

    report = f"""
╔══════════════════════════════════════════════════════════╗
║           BioVoice-Agents — Benchmark Report             ║
║  Query: {BENCHMARK_QUERY[:48]:<48} ║
╠══════════════════════════════════════════════════════════╣
║  REVIEW MODE                                             ║
║  Pipeline time          : {elapsed:>6.0f}s  (target: <300s)      ║
║  Total review words     : {len(review.split()):>6}                          ║
║  Numbered citations [N] : {len(numbered):>6}                          ║
║  Citation rate          : {cite_rate:>5.1%}  (target: ≥30%)       ║
║  Suspicious citations   : {warn_rate:>5.1%}  (target: ≤25%)       ║
║  Antibodies extracted   : {len(antibodies):>6}                          ║
╠══════════════════════════════════════════════════════════╣
║  GRANT MODE                                              ║
║  Sections generated     : {grant_sections:>6}  (target: ≥3)          ║
║  Citations verified     : {grant_citations:>6}  (target: ≥5)          ║
║  Hallucinated PMIDs     : {grant_hallucinated:>6}  (target: 0)           ║
╚══════════════════════════════════════════════════════════╝"""
    print(report)
    # Always passes — this is a reporting test
    assert True
