"""
tests/test_qa_engine.py
Unit tests for biovoice.qa.engine.QAEngine.

Uses a mock RAG object — no live API calls, no vector store required.
Run: pytest tests/test_qa_engine.py -v
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

from biovoice.qa.engine import QAEngine, QAResult


# ── Stubs ─────────────────────────────────────────────────────────────────────

@dataclass
class _FakeDoc:
    page_content: str
    metadata: dict = field(default_factory=dict)


class _FakeRAG:
    """Minimal RAG stub — returns canned docs for any query."""
    def similarity_search(self, query: str, k: int = 5) -> List[_FakeDoc]:
        return [
            _FakeDoc(
                page_content=(
                    "CR6261 targets the HA stalk epitope of influenza H1N1 "
                    "and H5N1. IGHV1-69 germline usage is common."
                ),
                metadata={"pmid": "12345678", "title": "CR6261 HA stalk paper"},
            ),
            _FakeDoc(
                page_content=(
                    "FI6 achieves broad neutralisation across both influenza A "
                    "group 1 and group 2 subtypes."
                ),
                metadata={"pmid": "87654321", "title": "FI6 pan-influenza paper"},
            ),
        ]


class _FakeModel:
    """Minimal model stub — echoes the question back with a citation."""
    def chat(self, system: str, human: str) -> str:
        # Extract question from human prompt
        if "CR6261" in human:
            return "CR6261 binds the HA stalk via IGHV1-69 [1]."
        if "FI6" in human:
            return "FI6 neutralises group 1 and group 2 influenza [2]."
        return "The literature describes broadly neutralising antibodies [1][2]."


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> QAEngine:
    return QAEngine(rag=_FakeRAG(), model=_FakeModel(), k=2)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ask_returns_qa_result(engine):
    result = engine.ask("What epitope does CR6261 target?")
    assert isinstance(result, QAResult)


def test_answer_text_non_empty(engine):
    result = engine.ask("What epitope does CR6261 target?")
    assert len(result.text) > 0


def test_references_populated(engine):
    result = engine.ask("What epitope does CR6261 target?")
    assert len(result.references) == 2  # _FakeRAG returns 2 docs
    assert result.references[0]["pmid"] == "12345678"
    assert result.references[1]["pmid"] == "87654321"


def test_history_grows_after_ask(engine):
    assert len(engine.history) == 0
    engine.ask("First question")
    assert len(engine.history) == 2   # user + assistant
    engine.ask("Second question")
    assert len(engine.history) == 4


def test_history_reset(engine):
    engine.ask("First question")
    engine.reset()
    assert len(engine.history) == 0


def test_format_response_contains_references(engine):
    result = engine.ask("What is CR6261?")
    formatted = engine.format_response(result)
    assert "References" in formatted
    assert "12345678" in formatted    # PMID from fake doc


def test_format_response_contains_answer_text(engine):
    result = engine.ask("What is CR6261?")
    formatted = engine.format_response(result)
    assert result.text in formatted


def test_history_as_gradio_chatbot(engine):
    engine.ask("Hello")
    engine.ask("Follow up")
    chatbot_hist = engine.history_as_gradio_chatbot()
    assert isinstance(chatbot_hist, list)
    assert len(chatbot_hist) == 2
    # Each entry is [user, assistant]
    for turn in chatbot_hist:
        assert isinstance(turn, list)
        assert len(turn) == 2


def test_multi_turn_history_included_in_prompt(engine):
    """The engine includes conversation history in follow-up prompts."""
    engine.ask("What is CR6261?")
    # After the first turn, history has 2 entries.
    # Second ask should receive history in the prompt.
    result2 = engine.ask("How does it compare to FI6?")
    # We can't inspect the prompt directly, but we can verify history_len
    assert result2.history_len >= 1


def test_empty_query_still_returns_result(engine):
    result = engine.ask("")
    assert isinstance(result, QAResult)


def test_history_window_capped():
    """Engine includes at most _HISTORY_WINDOW prior turns."""
    from biovoice.qa.engine import _HISTORY_WINDOW
    eng = QAEngine(rag=_FakeRAG(), model=_FakeModel(), k=2)
    # Ask many questions to overflow the window
    for i in range(_HISTORY_WINDOW + 3):
        eng.ask(f"Question {i}")
    assert len(eng.history) == (_HISTORY_WINDOW + 3) * 2   # all kept in list
    # But the last ask's history_len should be capped at _HISTORY_WINDOW // 2
    # (engine only passes _HISTORY_WINDOW entries to the model)
    last = eng.ask("Final question")
    # history_len reflects turns actually passed to model
    assert last.history_len <= _HISTORY_WINDOW // 2
