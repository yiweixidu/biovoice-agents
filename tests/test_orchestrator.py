"""
tests/test_orchestrator.py
Unit tests for BioVoiceOrchestrator — agent failure handling, _extract_antibodies,
grant mode specifics. No live LLM calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from biovoice.agents.base import FetchResult
from biovoice.core.orchestrator import BioVoiceOrchestrator, _is_relevant


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_config():
    return {
        "llm_type":    "openai",
        "llm_model":   "gpt-4o-mini",
        "openai_key":  "sk-test",
        "output_dir":  "/tmp/biovoice_test",
    }


@pytest.fixture()
def orch(minimal_config):
    """Orchestrator with RAG disabled and mocked model client."""
    with patch("biovoice.core.orchestrator.build_model_client") as mock_build, \
         patch("biovoice.core.orchestrator.PPTGenerator"), \
         patch("biovoice.core.orchestrator.AgentRegistry"):
        mock_build.return_value = MagicMock()
        return BioVoiceOrchestrator(minimal_config, use_rag=False)


# ── _is_relevant ──────────────────────────────────────────────────────────────

def test_is_relevant_matching_keyword():
    article = {"title": "Influenza broadly neutralizing antibody", "abstract": ""}
    assert _is_relevant(article) is True


def test_is_relevant_no_keywords():
    article = {"title": "Chocolate cake recipe", "abstract": "Mix flour and sugar."}
    assert _is_relevant(article) is False


def test_is_relevant_missing_abstract_key():
    """Must not raise KeyError when abstract is absent."""
    article = {"title": "Influenza hemagglutinin structure"}
    assert _is_relevant(article) is True


# ── _extract_antibodies ───────────────────────────────────────────────────────

def test_extract_antibodies_valid_json(orch):
    payload = json.dumps({
        "antibodies": [
            {"antibody_name": "CR6261", "target_protein": "HA stalk"},
            {"antibody_name": "MEDI8852", "target_protein": "HA stalk"},
        ]
    })
    orch.model.chat.return_value = payload
    result = orch._extract_antibodies("Some review text")
    assert len(result) == 2
    assert result[0]["antibody_name"] == "CR6261"


def test_extract_antibodies_malformed_json_returns_empty(orch, capsys):
    orch.model.chat.return_value = "this is not json at all {{"
    result = orch._extract_antibodies("Some text")
    assert result == []
    captured = capsys.readouterr()
    assert "malformed JSON" in captured.out


def test_extract_antibodies_empty_list_logs(orch, capsys):
    orch.model.chat.return_value = json.dumps({"antibodies": []})
    result = orch._extract_antibodies("Some text")
    assert result == []
    captured = capsys.readouterr()
    assert "empty" in captured.out


def test_extract_antibodies_strips_markdown_fences(orch):
    payload = "```json\n" + json.dumps({"antibodies": [{"antibody_name": "FI6v3"}]}) + "\n```"
    orch.model.chat.return_value = payload
    result = orch._extract_antibodies("text")
    assert result[0]["antibody_name"] == "FI6v3"


def test_extract_antibodies_filters_missing_name(orch):
    payload = json.dumps({
        "antibodies": [
            {"target_protein": "HA stalk"},       # no antibody_name — should be filtered
            {"antibody_name": "CR9114", "target_protein": "HA stem"},
        ]
    })
    orch.model.chat.return_value = payload
    result = orch._extract_antibodies("text")
    assert len(result) == 1
    assert result[0]["antibody_name"] == "CR9114"


# ── Grant mode specifics ──────────────────────────────────────────────────────

def test_rag_not_instantiated_when_use_rag_false(minimal_config):
    with patch("biovoice.core.orchestrator.build_model_client"), \
         patch("biovoice.core.orchestrator.PPTGenerator"), \
         patch("biovoice.core.orchestrator.AgentRegistry"):
        orch = BioVoiceOrchestrator(minimal_config, use_rag=False)
        assert orch.rag is None


def test_rag_instantiated_when_use_rag_true(minimal_config):
    mock_rag_cls = MagicMock()
    mock_rag_module = MagicMock()
    mock_rag_module.FluBroadRAG = mock_rag_cls

    import sys
    original = sys.modules.get("core.rag.vector_store")
    sys.modules["core.rag.vector_store"] = mock_rag_module

    try:
        with patch("biovoice.core.orchestrator.build_model_client"), \
             patch("biovoice.core.orchestrator.PPTGenerator"), \
             patch("biovoice.core.orchestrator.AgentRegistry"):
            orch = BioVoiceOrchestrator(minimal_config, use_rag=True)
            assert orch.rag is not None
    finally:
        if original is None:
            sys.modules.pop("core.rag.vector_store", None)
        else:
            sys.modules["core.rag.vector_store"] = original


# ── _fetch_all partial failure ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_all_partial_failure(minimal_config):
    """If one agent raises, _fetch_all returns results from the others."""
    good_result = FetchResult(
        source="pubmed",
        items=[{"pmid": "1234", "title": "Test"}],
        metadata={},
        prompt_context="",
    )

    bad_agent        = MagicMock()
    bad_agent.name   = "europe_pmc"
    bad_agent.fetch  = AsyncMock(side_effect=TimeoutError("network timeout"))

    good_agent       = MagicMock()
    good_agent.name  = "pubmed"
    good_agent.fetch = AsyncMock(return_value=good_result)

    with patch("biovoice.core.orchestrator.build_model_client"), \
         patch("biovoice.core.orchestrator.PPTGenerator"), \
         patch("biovoice.core.orchestrator.AgentRegistry") as mock_reg:

        mock_reg.build_from_config.return_value = [good_agent, bad_agent]
        orch = BioVoiceOrchestrator(minimal_config, use_rag=False)
        results = await orch._fetch_all("influenza antibody", ["pubmed", "europe_pmc"], None)

    # Only the successful FetchResult should come through
    assert len(results) == 1
    assert results[0].source == "pubmed"
