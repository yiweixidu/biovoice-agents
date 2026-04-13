"""
tests/test_airr_agent.py
Unit tests for biovoice.agents.airr_agent.

Uses a temporary in-memory AIRR TSV file — no live API calls, no network.
Run: pytest tests/test_airr_agent.py -v
"""

import os
import pytest

from biovoice.agents.airr_agent import parse_airr_tsv, AIRRAgent
from biovoice.agents.base import AgentConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────

AIRR_TSV_CONTENT = """\
sequence_id\tv_call\tj_call\tc_call\tjunction_aa\tjunction_length\tproductive\tv_identity\tduplicate_count\tsubject_id\tsample_id\tantigen
seq001\tIGHV1-69*01\tIGHJ4*02\tIGHA1\tCARDRYSGGAFDYW\t39\tT\t0.97\t42\tdonor01\tsample_A\tinfluenza H3N2
seq002\tIGHV3-30*01\tIGHJ6*02\tIGHG1\tCARSGYSSAFDIW\t36\tT\t0.93\t15\tdonor01\tsample_A\t
seq003\tIGHV1-69*02\tIGHJ4*01\tIGHA2\tCARDRYSGSSAFDYW\t42\tF\t0.85\t8\tdonor02\tsample_B\tinfluenza H1N1
seq004\tIGHV3-23*01\tIGHJ5*01\tIGHM\tCARVGSSGFAFDYW\t36\tT\t0.75\t3\tdonor02\tsample_B\t
seq005\tIGHV1-18*01\tIGHJ4*02\tIGHG2\tCARDRYSAFDIW\t33\tT\t0.91\t21\tdonor01\tsample_A\tinfluenza HA stalk
"""


@pytest.fixture
def airr_file(tmp_path):
    tsv_path = tmp_path / "test.airr.tsv"
    tsv_path.write_text(AIRR_TSV_CONTENT, encoding="utf-8")
    return str(tsv_path)


# ── Parser tests ──────────────────────────────────────────────────────────────

def test_parse_returns_records(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=False)
    assert len(records) == 5


def test_productive_filter(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=True)
    # seq003 has productive=F → excluded
    assert len(records) == 4


def test_v_identity_filter(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=False, min_v_identity=0.90)
    # seq001 (0.97), seq002 (0.93), seq005 (0.91) pass; seq003 (0.85), seq004 (0.75) fail
    assert len(records) == 3


def test_field_extraction(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=False)
    first = records[0]
    assert first["sequence_id"] == "seq001"
    assert first["v_call"] == "IGHV1-69*01"
    assert first["v_gene"] == "IGHV1-69"        # allele stripped
    assert first["j_call"] == "IGHJ4*02"
    assert first["cdr3_aa"] == "CARDRYSGGA FDY W".replace(" ", "")
    assert first["subject_id"] == "donor01"
    assert first["sample_id"] == "sample_A"
    assert first["antigen"] == "influenza H3N2"
    assert first["duplicate_count"] == 42
    assert abs(first["v_identity"] - 0.97) < 1e-3


def test_source_field(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=False)
    assert all(r["source"] == "airr" for r in records)
    assert all(r["_agent_source"] == "airr" for r in records)


def test_abstract_contains_key_info(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=False)
    first = records[0]
    assert "IGHV1-69" in first["abstract"]
    assert "donor01" in first["abstract"]


def test_max_records_cap(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=False, max_records=2)
    assert len(records) == 2


def test_title_contains_sequence_id(airr_file):
    records = parse_airr_tsv(airr_file, productive_only=False)
    assert "seq001" in records[0]["title"]
    assert "IGHV1-69" in records[0]["title"]


# ── Agent tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_fetch_returns_result(airr_file):
    config = AgentConfig(
        name="airr",
        enabled=True,
        extra_params={
            "data_path":       airr_file,
            "productive_only": True,
            "min_v_identity":  0.0,
        },
    )
    agent = AIRRAgent(config)
    result = await agent.fetch("influenza broadly neutralizing", limit=100)
    assert result.source == "airr"
    assert len(result.items) == 4          # 4 productive records
    assert result.metadata["total"] == 4


@pytest.mark.asyncio
async def test_agent_no_data_path_returns_empty():
    config = AgentConfig(
        name="airr",
        enabled=True,
        extra_params={"data_path": ""},
    )
    agent = AIRRAgent(config)
    result = await agent.fetch("influenza")
    assert len(result.items) == 0


@pytest.mark.asyncio
async def test_agent_nonexistent_path_returns_empty():
    config = AgentConfig(
        name="airr",
        enabled=True,
        extra_params={"data_path": "/nonexistent/path/to/data.tsv"},
    )
    agent = AIRRAgent(config)
    result = await agent.fetch("influenza")
    assert len(result.items) == 0


@pytest.mark.asyncio
async def test_agent_directory_ingestion(tmp_path):
    """Agent reads all .tsv files from a directory."""
    for i in range(3):
        p = tmp_path / f"donor{i:02d}.tsv"
        p.write_text(AIRR_TSV_CONTENT, encoding="utf-8")
    config = AgentConfig(
        name="airr",
        enabled=True,
        extra_params={
            "data_path":       str(tmp_path),
            "productive_only": True,
        },
    )
    agent = AIRRAgent(config)
    result = await agent.fetch("influenza", limit=100)
    # 4 productive per file × 3 files = 12
    assert len(result.items) == 12


def test_context_contains_v_gene_summary(airr_file):
    config = AgentConfig(
        name="airr",
        enabled=True,
        extra_params={"data_path": airr_file, "productive_only": False},
    )
    import asyncio
    agent  = AIRRAgent(config)
    result = asyncio.run(agent.fetch("influenza"))
    assert "IGHV" in result.prompt_context
