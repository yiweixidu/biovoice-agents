"""
tests/test_feedback_store.py
Unit tests for biovoice.finetuning.feedback_store.FeedbackStore.

No network, no LLM, no GPU — pure Python file I/O.
Run: pytest tests/test_feedback_store.py -v
"""

import json
import os
import pytest

from biovoice.finetuning.feedback_store import FeedbackStore, FeedbackRecord


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path) -> FeedbackStore:
    return FeedbackStore(feedback_dir=str(tmp_path / "feedback"))


# ── log_inference tests ───────────────────────────────────────────────────────

def test_log_inference_returns_id(store):
    rec_id = store.log_inference(
        query="broadly neutralizing antibodies",
        original_text="CR6261 targets the HA stalk [1].",
    )
    assert isinstance(rec_id, str) and len(rec_id) > 0


def test_log_inference_writes_to_disk(store, tmp_path):
    store.log_inference(
        query="influenza bnab",
        original_text="FI6 achieves pan-influenza breadth.",
        section="results",
    )
    inf_path = os.path.join(str(tmp_path / "feedback"), "inferences.jsonl")
    assert os.path.exists(inf_path)
    with open(inf_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    assert len(records) == 1
    assert records[0]["query"] == "influenza bnab"
    assert records[0]["section"] == "results"


# ── submit_correction tests ───────────────────────────────────────────────────

def test_submit_correction_stores_record(store):
    rec_id = store.log_inference(
        query="influenza H3N2 antibody",
        original_text="MEDI8852 is a broadly neutralising antibody.",
    )
    rec = store.submit_correction(
        record_id=rec_id,
        corrected_text="MEDI8852 targets the HA stalk of H3N2 (PMID:28481361).",
        rating=4,
        rating_reason="Added missing PMID.",
    )
    assert isinstance(rec, FeedbackRecord)
    assert rec.id == rec_id
    assert rec.rating == 4
    assert "PMID:28481361" in rec.corrected_text


def test_submit_correction_links_back_to_inference(store):
    rec_id = store.log_inference(
        query="vaccine design",
        original_text="Universal vaccines remain elusive.",
    )
    store.submit_correction(
        record_id=rec_id,
        corrected_text="Universal influenza vaccines remain elusive [PMID:12345678].",
        rating=3,
    )
    corrections = store.list_corrections(limit=10)
    assert any(c.id == rec_id for c in corrections)


def test_submit_correction_without_inference(store):
    """submit_correction with an unknown ID creates a standalone correction."""
    rec = store.submit_correction(
        record_id="unknown-id",
        corrected_text="Some expert-written synthesis.",
        rating=5,
    )
    assert rec.corrected_text == "Some expert-written synthesis."


# ── stats tests ───────────────────────────────────────────────────────────────

def test_stats_empty_store(store):
    s = store.stats()
    assert s["total_inferences"] == 0
    assert s["total_corrections"] == 0
    assert s["avg_rating"] == 0.0


def test_stats_after_activity(store):
    rid = store.log_inference(query="q1", original_text="original 1")
    store.log_inference(query="q2", original_text="original 2")
    store.submit_correction(rid, corrected_text="corrected 1", rating=4)
    s = store.stats()
    assert s["total_inferences"] == 2
    assert s["total_corrections"] == 1
    assert s["avg_rating"] == 4.0


# ── export_jsonl tests ────────────────────────────────────────────────────────

def test_export_sharegpt_format(store, tmp_path):
    rid = store.log_inference(
        query="broadly neutralizing antibodies influenza",
        original_text="CR6261 targets HA stalk.",
        section="results",
    )
    store.submit_correction(
        record_id=rid,
        corrected_text="CR6261 binds the HA stalk conserved across group 1 subtypes [PMID:12345678].",
        rating=5,
    )
    out = str(tmp_path / "train.jsonl")
    n = store.export_jsonl(output_path=out, format="sharegpt")
    assert n == 1
    assert os.path.exists(out)
    with open(out) as f:
        rec = json.loads(f.readline())
    assert "conversations" in rec
    convs = rec["conversations"]
    roles = [c["from"] for c in convs]
    assert "human" in roles
    assert "gpt" in roles
    # The corrected text should be the assistant turn
    gpt_turn = next(c for c in convs if c["from"] == "gpt")
    assert "PMID:12345678" in gpt_turn["value"]


def test_export_alpaca_format(store, tmp_path):
    rid = store.log_inference(query="q", original_text="orig")
    store.submit_correction(record_id=rid, corrected_text="corrected", rating=4)
    out = str(tmp_path / "train_alpaca.jsonl")
    n = store.export_jsonl(output_path=out, format="alpaca")
    assert n == 1
    with open(out) as f:
        rec = json.loads(f.readline())
    assert "instruction" in rec
    assert "input" in rec
    assert "output" in rec
    assert rec["output"] == "corrected"


def test_export_min_rating_filter(store, tmp_path):
    rid1 = store.log_inference(query="q1", original_text="o1")
    rid2 = store.log_inference(query="q2", original_text="o2")
    store.submit_correction(rid1, corrected_text="c1", rating=2)   # below threshold
    store.submit_correction(rid2, corrected_text="c2", rating=5)   # above threshold
    out = str(tmp_path / "train_filtered.jsonl")
    n = store.export_jsonl(output_path=out, min_rating=3)
    assert n == 1   # only the rating-5 correction


def test_export_empty_store_returns_zero(store, tmp_path):
    out = str(tmp_path / "empty.jsonl")
    n = store.export_jsonl(output_path=out)
    assert n == 0


def test_export_include_uncorrected(store, tmp_path):
    # Two inferences, one corrected, one not
    rid1 = store.log_inference(query="q1", original_text="original 1")
    store.log_inference(query="q2", original_text="original 2")
    store.submit_correction(rid1, corrected_text="corrected 1", rating=5)
    out = str(tmp_path / "train_all.jsonl")
    n = store.export_jsonl(
        output_path=out,
        include_uncorrected=True,
    )
    assert n == 2   # 1 corrected + 1 uncorrected


# ── list_corrections tests ────────────────────────────────────────────────────

def test_list_corrections_returns_records(store):
    for i in range(5):
        rid = store.log_inference(query=f"q{i}", original_text=f"orig {i}")
        store.submit_correction(rid, corrected_text=f"corr {i}", rating=4)
    corrections = store.list_corrections(limit=3)
    assert len(corrections) == 3
    assert all(isinstance(c, FeedbackRecord) for c in corrections)
