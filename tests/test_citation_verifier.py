"""
tests/test_citation_verifier.py
Unit tests for the citation verification logic in biovoice/core/orchestrator.py.

These are pure logic tests — no LLM calls, no network.
"""

import pytest
from biovoice.core.orchestrator import (
    _extract_cite_pmids,
    _jaccard,
    _token_set,
    verify_citations,
)
from domain.virology.prompts.pmrc_templates import BIOMEDICAL_STOPLIST


# ── _extract_cite_pmids ───────────────────────────────────────────────────────

def test_extract_cite_pmids_single():
    text = "Antibody CR6261 was shown to neutralize group 1 strains [CITE:12345678]."
    assert _extract_cite_pmids(text) == ["12345678"]


def test_extract_cite_pmids_multiple():
    text = (
        "HA stalk antibodies [CITE:11111111] show broad neutralization "
        "and Fc-mediated clearance [CITE:22222222]."
    )
    assert _extract_cite_pmids(text) == ["11111111", "22222222"]


def test_extract_cite_pmids_none():
    text = "This sentence has no citation markers at all."
    assert _extract_cite_pmids(text) == []


# ── _jaccard ──────────────────────────────────────────────────────────────────

def test_jaccard_identical():
    s = {"influenza", "antibody", "stem"}
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"cat", "dog"}, {"bird", "fish"}) == 0.0


def test_jaccard_partial():
    a = {"influenza", "antibody", "hemagglutinin"}
    b = {"influenza", "vaccine", "hemagglutinin"}
    # intersection = 2, union = 4 → 0.5
    assert _jaccard(a, b) == pytest.approx(0.5)


def test_jaccard_empty_sets():
    assert _jaccard(set(), {"x"}) == 0.0
    assert _jaccard({"x"}, set()) == 0.0


# ── _token_set ────────────────────────────────────────────────────────────────

def test_token_set_lowercases():
    tokens = _token_set("Influenza VIRUS Antibody", set())
    assert "influenza" in tokens
    assert "virus" in tokens
    assert "antibody" in tokens


def test_token_set_removes_stoplist():
    tokens = _token_set("The virus cells protein response", BIOMEDICAL_STOPLIST)
    # "the", "virus", "cells", "protein", "response" are all in the stoplist
    assert "the" not in tokens
    assert "virus" not in tokens
    assert "cells" not in tokens


def test_token_set_keeps_domain_terms():
    tokens = _token_set("hemagglutinin stem IGHV1-69 CR6261 neutralization", BIOMEDICAL_STOPLIST)
    assert "hemagglutinin" in tokens
    assert "stem" in tokens
    assert "cr6261" in tokens


def test_token_set_filters_single_chars():
    tokens = _token_set("a b c hemagglutinin", set())
    assert "a" not in tokens
    assert "b" not in tokens
    assert "hemagglutinin" in tokens


# ── verify_citations ──────────────────────────────────────────────────────────

PAPER_A = {
    "pmid": "12345678",
    "title": "CR6261 neutralizes group 1 influenza via HA stalk binding",
    "abstract": (
        "CR6261 is a broadly neutralizing antibody targeting the hemagglutinin "
        "stalk domain. It neutralizes all group 1 influenza A strains with IC50 "
        "values of 0.01-0.1 ug/mL. IGHV1-69 germline usage was confirmed."
    ),
}

PAPER_B = {
    "pmid": "87654321",
    "title": "Fc effector functions enhance in vivo protection",
    "abstract": (
        "Fc-mediated ADCC and ADCP are required for full protection in mouse models. "
        "Fc-knockout variants showed reduced efficacy despite equivalent neutralization."
    ),
}


def test_verify_citations_clean_pass():
    text = (
        "CR6261 targets the hemagglutinin stalk domain with IGHV1-69 germline "
        "usage [CITE:12345678]."
    )
    verified, warnings = verify_citations(
        text,
        [PAPER_A, PAPER_B],
        jaccard_threshold=0.08,
        stoplist=BIOMEDICAL_STOPLIST,
    )
    # No warnings expected — claim closely matches title + abstract
    assert "[1]" in verified
    assert "[CITE:12345678]" not in verified
    assert not warnings


def test_verify_citations_pmid_not_in_results():
    text = "Some claim about an antibody [CITE:99999999]."
    verified, warnings = verify_citations(
        text,
        [PAPER_A],
        jaccard_threshold=0.08,
        stoplist=BIOMEDICAL_STOPLIST,
    )
    assert any("not found in fetched results" in w for w in warnings)
    assert any("99999999" in w for w in warnings)


def test_verify_citations_low_jaccard_flagged():
    # Claim is completely unrelated to PAPER_A's title + abstract
    text = (
        "Climate change accelerates sea level rise [CITE:12345678]."
    )
    _, warnings = verify_citations(
        text,
        [PAPER_A],
        jaccard_threshold=0.08,
        stoplist=BIOMEDICAL_STOPLIST,
    )
    assert any("Jaccard" in w for w in warnings)


def test_verify_citations_no_markers():
    text = "This paragraph has no citation markers."
    _, warnings = verify_citations(
        text,
        [PAPER_A],
        jaccard_threshold=0.08,
        stoplist=BIOMEDICAL_STOPLIST,
    )
    assert any("no [CITE:PMID] markers" in w for w in warnings)


def test_verify_citations_multiple_papers():
    text = (
        "ADCC ADCP Fc effector functions required in vivo protection "
        "[CITE:87654321]. CR6261 IGHV1-69 germline hemagglutinin stalk "
        "neutralization IC50 [CITE:12345678]."
    )
    verified, warnings = verify_citations(
        text,
        [PAPER_A, PAPER_B],
        jaccard_threshold=0.08,
        stoplist=BIOMEDICAL_STOPLIST,
    )
    assert "[1]" in verified
    assert "[2]" in verified
    assert not warnings


def test_verify_citations_numbered_in_order():
    text = (
        "First claim [CITE:12345678]. Second claim [CITE:87654321]. "
        "Third repeat of first [CITE:12345678]."
    )
    verified, _ = verify_citations(
        text,
        [PAPER_A, PAPER_B],
        jaccard_threshold=0.0,  # no Jaccard threshold for this test
        stoplist=set(),
    )
    # First PMID encountered gets [1], second gets [2], repeated ref stays [1]
    assert verified.count("[1]") == 2
    assert verified.count("[2]") == 1


# ── antibody_schema import guard ──────────────────────────────────────────────

def test_antibody_schema_import():
    """Regression guard: antibody_schema.py must not be empty."""
    from domain.virology.schemas.antibody_schema import antibody_schema
    assert isinstance(antibody_schema, dict)
    assert len(antibody_schema) >= 5
    assert "antibody_name" in antibody_schema
    assert "key_pmids" in antibody_schema
