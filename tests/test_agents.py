"""
tests/test_agents.py
Unit tests for agent contracts and FetchResult model.

These tests mock heavy dependencies (biopython, pdfplumber) so the suite
runs without the full scientific stack installed. Error propagation is
tested at the orchestrator level in test_orchestrator.py.

Integration test requires real NCBI API access; marked @pytest.mark.integration,
skipped by default.

Run integration test:
    NCBI_EMAIL=you@example.com pytest -m integration tests/test_agents.py
"""

import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from biovoice.agents.base import AgentConfig, FetchResult


# ── FetchResult model ─────────────────────────────────────────────────────────

def test_fetch_result_empty_items():
    fr = FetchResult(source="uniprot", items=[], metadata={}, prompt_context="")
    assert fr.count == 0
    assert fr.items == []


def test_fetch_result_count():
    items = [{"pmid": str(i)} for i in range(5)]
    fr = FetchResult(source="pubmed", items=items, metadata={}, prompt_context="")
    assert fr.count == 5


# ── AgentConfig defaults ─────────────────────────────────��────────────────────

def test_agent_config_defaults():
    cfg = AgentConfig(name="pubmed")
    assert cfg.enabled is True
    assert cfg.api_key is None
    assert cfg.extra_params == {}


# ── PubMedAgent ───────────────────────────────────────────────────────────────

def _mock_pubmed_modules():
    """Return a patch.dict that stubs out Bio and pdfplumber for import."""
    return patch.dict(sys.modules, {
        "Bio":                  MagicMock(),
        "Bio.Entrez":           MagicMock(),
        "pdfplumber":           MagicMock(),
        "core.retrieval.pubmed":         MagicMock(PubMedFetcher=MagicMock),
        "core.retrieval.pmc_fulltext":   MagicMock(
            PMCFulltextFetcher=MagicMock,
            UnpaywallFetcher=MagicMock,
        ),
    })


@pytest.mark.asyncio
async def test_pubmed_agent_returns_fetch_result():
    """PubMedAgent wraps results in a FetchResult."""
    mock_items = [{"pmid": "12345", "title": "HA stalk antibody", "abstract": "Test."}]

    with _mock_pubmed_modules():
        from biovoice.agents.pubmed_agent import PubMedAgent
        cfg   = AgentConfig(name="pubmed", extra_params={"email": "test@test.com", "fetch_fulltext": False})
        agent = PubMedAgent(cfg)
        with patch.object(agent, "_fetch_articles", return_value=mock_items):
            result = await agent.fetch("influenza antibody", limit=5)

    assert isinstance(result, FetchResult)
    assert result.source == "pubmed"
    assert result.items[0]["pmid"] == "12345"


@pytest.mark.asyncio
async def test_pubmed_fetch_fulltext_false_skips_enrich():
    """When fetch_fulltext=False, _enrich is never called."""
    mock_items = [{"pmid": "99", "title": "Test", "abstract": "Abstract."}]

    with _mock_pubmed_modules():
        from biovoice.agents.pubmed_agent import PubMedAgent
        cfg   = AgentConfig(name="pubmed", extra_params={"email": "t@t.com", "fetch_fulltext": False})
        agent = PubMedAgent(cfg)
        enrich_mock = MagicMock(return_value=mock_items)
        with patch.object(agent, "_fetch_articles", return_value=mock_items), \
             patch.object(agent, "_enrich", enrich_mock):
            await agent.fetch("query", limit=5)

    enrich_mock.assert_not_called()


# ── EuropePMCAgent ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_europe_pmc_returns_fetch_result():
    """EuropePMCAgent wraps results in a FetchResult."""
    from biovoice.agents.europe_pmc_agent import EuropePMCAgent

    mock_items = [{"pmid": "55555", "title": "Europe PMC paper"}]
    cfg   = AgentConfig(name="europe_pmc")
    agent = EuropePMCAgent(cfg)

    with patch.object(agent, "_search", return_value=mock_items):
        result = await agent.fetch("influenza antibody", limit=5)

    assert isinstance(result, FetchResult)
    assert result.source == "europe_pmc"
    assert len(result.items) == 1


@pytest.mark.asyncio
async def test_europe_pmc_empty_results_no_crash():
    """EuropePMCAgent returns empty FetchResult when no results found."""
    from biovoice.agents.europe_pmc_agent import EuropePMCAgent

    cfg   = AgentConfig(name="europe_pmc")
    agent = EuropePMCAgent(cfg)

    with patch.object(agent, "_search", return_value=[]):
        result = await agent.fetch("influenza antibody", limit=5)

    assert isinstance(result, FetchResult)
    assert result.items == []


# ── UniProtAgent ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_uniprot_agent_returns_fetch_result():
    """UniProtAgent wraps results in a FetchResult."""
    from biovoice.agents.uniprot_agent import UniProtAgent

    mock_items = [{"accession": "P00533", "name": "Hemagglutinin", "gene": "HA", "organism": "Influenza A", "function": "Surface glycoprotein"}]
    cfg   = AgentConfig(name="uniprot")
    agent = UniProtAgent(cfg)

    with patch.object(agent, "_search", return_value=mock_items):
        result = await agent.fetch("influenza hemagglutinin", limit=5)

    assert isinstance(result, FetchResult)
    assert result.source == "uniprot"
    assert len(result.items) == 1


# ── Integration test ──────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pubmed_real_search():
    """
    Real PubMed search. Requires PUBMED_API_KEY or NCBI_EMAIL env var.
    Skip in CI: pytest -m 'not integration'
    """
    import os
    from biovoice.agents.pubmed_agent import PubMedAgent

    email   = os.environ.get("NCBI_EMAIL", os.environ.get("EMAIL", ""))
    api_key = os.environ.get("PUBMED_API_KEY", "")

    cfg    = AgentConfig(name="pubmed", api_key=api_key, extra_params={"email": email})
    agent  = PubMedAgent(cfg)
    result = await agent.fetch("influenza hemagglutinin antibody", limit=3)

    assert isinstance(result, FetchResult)
    assert len(result.items) >= 1
    first = result.items[0]
    assert "pmid" in first or "PMID" in first
