"""
biovoice/agents/airr_agent.py
AIRR (Adaptive Immune Receptor Repertoire) private data agent.

Ingests BCR/TCR sequencing data in AIRR-schema TSV format (v1.4+) and
converts it into BioVoice corpus items so the orchestrator can build a
RAG index and knowledge graph over the lab's own sequencing data.

AIRR Community standard: https://docs.airr-community.org/en/stable/datarep/rearrangements.html

Supported input formats
-----------------------
- AIRR TSV  (.tsv / .airr) — the primary community standard
- Change-O TSV (legacy)    — MakeDb.py output from IgBLAST / IMGT

Key fields extracted (AIRR column names)
-----------------------------------------
  sequence_id       — unique read identifier
  sequence          — full nucleotide sequence
  v_call            — IGHV / IGLV / TRBV gene call (may have allele, e.g. IGHV1-69*01)
  d_call            — IGHD gene call
  j_call            — IGHJ / IGLJ / TRBJ gene call
  c_call            — isotype / constant region
  junction_aa       — CDR3 amino acid sequence
  junction_length   — CDR3 nucleotide length
  productive        — True/False
  v_identity        — % identity to germline V
  duplicate_count   — copy number / count in repertoire
  clone_id          — clonal family assignment (optional)
  subject_id        — donor ID (custom field, often added by labs)
  sample_id         — sample label (custom field)
  antigen           — binding target if known (custom field)

Usage
-----
    # Via agent registry (config-driven):
    config = AgentConfig(name="airr", enabled=True, extra_params={
        "data_path": "/lab/bcr_data/subject01_IgG.airr.tsv",
        "productive_only": True,
        "min_v_identity": 0.80,
    })
    agent = AIRRAgent(config)
    result = await agent.fetch("broadly neutralizing antibody influenza H3N2")

    # Direct:
    from biovoice.agents.airr_agent import AIRRAgent, parse_airr_tsv
    records = parse_airr_tsv("/lab/donor.tsv", productive_only=True)
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
from typing import Dict, List, Optional

from .base import AgentConfig, BaseAgent, FetchResult


# ── AIRR column name maps ─────────────────────────────────────────────────────
# Maps canonical BioVoice field names → AIRR column names (with Change-O aliases)
_FIELD_MAP = {
    "sequence_id":      ["sequence_id", "SEQUENCE_ID"],
    "v_call":           ["v_call",      "V_CALL"],
    "d_call":           ["d_call",      "D_CALL"],
    "j_call":           ["j_call",      "J_CALL"],
    "c_call":           ["c_call",      "C_CALL"],
    "junction_aa":      ["junction_aa", "JUNCTION_AA", "CDR3_IMGT"],
    "junction_length":  ["junction_length", "JUNCTION_LENGTH"],
    "productive":       ["productive",  "PRODUCTIVE", "FUNCTIONAL"],
    "v_identity":       ["v_identity",  "V_IDENTITY"],
    "duplicate_count":  ["duplicate_count", "DUPCOUNT", "CLONE_COUNT"],
    "clone_id":         ["clone_id",    "CLONE"],
    "subject_id":       ["subject_id",  "SUBJECT", "DONOR"],
    "sample_id":        ["sample_id",   "SAMPLE"],
    "antigen":          ["antigen",     "ANTIGEN", "TARGET_ANTIGEN"],
    "sequence":         ["sequence",    "SEQUENCE_VDJ"],
}


def _resolve(row: Dict, canonical: str, default: str = "") -> str:
    """Return the first non-empty value found for a canonical field."""
    for col in _FIELD_MAP.get(canonical, [canonical]):
        val = row.get(col, "")
        stripped = str(val).strip() if val is not None else ""
        if stripped and stripped.lower() not in ("", "nan", "none"):
            return stripped
    return default


def _is_productive(row: Dict) -> bool:
    """
    AIRR standard: productive = T/TRUE/1 means productive.
    Change-O: FUNCTIONAL column uses T/F, where T = functional (productive).
    Both "F" and "FALSE" mean non-productive.
    """
    raw = _resolve(row, "productive", "T").upper()
    return raw in ("T", "TRUE", "YES", "1")


def _v_identity(row: Dict) -> float:
    raw = _resolve(row, "v_identity", "0")
    try:
        val = float(raw)
        # AIRR reports as 0–1; Change-O sometimes as 0–100
        return val / 100 if val > 1.0 else val
    except (ValueError, TypeError):
        return 0.0


# ── Public parser ─────────────────────────────────────────────────────────────

def parse_airr_tsv(
    path: str,
    productive_only: bool = True,
    min_v_identity:  float = 0.0,
    max_records:     int   = 0,     # 0 = no cap
) -> List[Dict]:
    """
    Parse an AIRR TSV file and return normalised record dicts.

    Parameters
    ----------
    path             : path to .tsv / .airr file
    productive_only  : skip non-productive rearrangements
    min_v_identity   : minimum V-gene identity (0–1). 0 = no filter.
    max_records      : maximum records to return (0 = unlimited)

    Returns
    -------
    List of normalised dicts with BioVoice-standard field names.
    """
    records: List[Dict] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if productive_only and not _is_productive(row):
                continue
            v_id = _v_identity(row)
            if min_v_identity > 0 and v_id < min_v_identity:
                continue

            seq_id    = _resolve(row, "sequence_id")
            v_call    = _resolve(row, "v_call")
            j_call    = _resolve(row, "j_call")
            c_call    = _resolve(row, "c_call")
            cdr3_aa   = _resolve(row, "junction_aa")
            cdr3_len  = _resolve(row, "junction_length")
            clone_id  = _resolve(row, "clone_id")
            subject   = _resolve(row, "subject_id")
            sample    = _resolve(row, "sample_id")
            antigen   = _resolve(row, "antigen")
            dup_count = _resolve(row, "duplicate_count", "1")

            # Extract gene family (IGHV1-69*01 → IGHV1-69)
            v_gene = v_call.split("*")[0] if "*" in v_call else v_call

            abstract = (
                f"BCR/TCR rearrangement from {subject or 'unknown donor'}, "
                f"sample {sample or 'unknown'}. "
                f"V-gene: {v_call}. J-gene: {j_call}. "
                + (f"Isotype: {c_call}. " if c_call else "")
                + (f"CDR3(aa): {cdr3_aa} (length {cdr3_len}). " if cdr3_aa else "")
                + (f"Clone: {clone_id}. " if clone_id else "")
                + (f"V-identity: {v_id:.1%}. " if v_id else "")
                + (f"Target antigen: {antigen}." if antigen else "")
            )

            records.append({
                # BioVoice corpus standard fields
                "source":              "airr",
                "pmid":                "",
                "doi":                 "",
                "title":               (
                    f"AIRR:{seq_id} | V:{v_gene} | CDR3:{cdr3_aa or '?'}"
                    + (f" | {antigen}" if antigen else "")
                ),
                "abstract":            abstract,
                "year":                "",
                "citation_count":      0,
                "fulltext_available":  False,

                # AIRR-specific fields
                "sequence_id":         seq_id,
                "v_call":              v_call,
                "v_gene":              v_gene,
                "d_call":              _resolve(row, "d_call"),
                "j_call":              j_call,
                "c_call":              c_call,
                "cdr3_aa":             cdr3_aa,
                "cdr3_length":         cdr3_len,
                "v_identity":          v_id,
                "duplicate_count":     int(float(dup_count)) if dup_count else 1,
                "clone_id":            clone_id,
                "subject_id":          subject,
                "sample_id":           sample,
                "antigen":             antigen,
                "_agent_source":       "airr",
            })

            if max_records and len(records) >= max_records:
                break

    return records


# ── AIRRAgent ─────────────────────────────────────────────────────────────────

class AIRRAgent(BaseAgent):
    """
    Agent for ingesting private BCR/TCR-seq data in AIRR TSV format.

    Config extra_params
    -------------------
    data_path        : path to an AIRR TSV file, or a directory containing
                       multiple .tsv / .airr files. Required.
    productive_only  : skip non-productive rearrangements (default: True)
    min_v_identity   : minimum V-gene identity filter (default: 0.0 = no filter)

    The agent ignores the `query` argument for file-based data (all records are
    loaded). Set limit to control the maximum number of records returned.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._data_path      = config.extra_params.get("data_path", "")
        self._productive     = config.extra_params.get("productive_only", True)
        self._min_v_identity = float(config.extra_params.get("min_v_identity", 0.0))

    def get_capabilities(self) -> List[str]:
        return ["airr", "bcr", "tcr", "repertoire", "cdr3", "v_gene", "private_data"]

    def get_default_prompt(self) -> str:
        return (
            "You are an immunologist analysing BCR repertoire data. "
            "Based on these AIRR rearrangement records about {topic}, "
            "describe the V-gene usage patterns, CDR3 length distributions, "
            "clonal expansion signatures, and any relationship to known "
            "broadly neutralising antibody gene families."
        )

    async def fetch(self, query: str, limit: int = 100, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(
            self._load_airr, query, limit
        )
        context = self._build_context(items)
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query,
                      "data_path": self._data_path},
            prompt_context=context,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_airr(self, query: str, limit: int) -> List[Dict]:
        if not self._data_path:
            print("[AIRR] No data_path configured — skipping")
            return []

        paths = self._resolve_paths()
        if not paths:
            print(f"[AIRR] No AIRR files found at: {self._data_path}")
            return []

        all_records: List[Dict] = []
        for path in paths:
            try:
                records = parse_airr_tsv(
                    path,
                    productive_only=self._productive,
                    min_v_identity=self._min_v_identity,
                    max_records=limit - len(all_records) if limit else 0,
                )
                all_records.extend(records)
                print(f"[AIRR] {len(records)} records from {os.path.basename(path)}")
                if limit and len(all_records) >= limit:
                    break
            except Exception as e:
                print(f"[AIRR] Failed to parse {path}: {e}")

        print(f"[AIRR] Total: {len(all_records)} records")
        return all_records[:limit] if limit else all_records

    def _resolve_paths(self) -> List[str]:
        path = self._data_path
        if os.path.isfile(path):
            return [path]
        if os.path.isdir(path):
            found = []
            for fname in sorted(os.listdir(path)):
                if fname.endswith((".tsv", ".airr", ".txt")):
                    found.append(os.path.join(path, fname))
            return found
        return []

    def _build_context(self, items: List[Dict]) -> str:
        if not items:
            return ""
        # Summarise V-gene usage distribution
        v_counts: Dict[str, int] = {}
        for it in items:
            gene = it.get("v_gene", "")
            if gene:
                v_counts[gene] = v_counts.get(gene, 0) + it.get("duplicate_count", 1)

        top_v = sorted(v_counts.items(), key=lambda x: -x[1])[:8]
        v_summary = ", ".join(f"{g}:{n}" for g, n in top_v)

        # Unique CDR3 examples
        cdrs = list({it["cdr3_aa"] for it in items[:100] if it.get("cdr3_aa")})[:5]
        cdr_examples = " | ".join(cdrs) if cdrs else "N/A"

        lines = [
            f"AIRR BCR/TCR repertoire data: {len(items)} sequences",
            f"Top V-genes (by copy count): {v_summary or 'N/A'}",
            f"CDR3 examples: {cdr_examples}",
        ]
        # Append sample-level summaries
        for it in items[:20]:
            lines.append(
                f"[AIRR] {it.get('sequence_id','?')} | V:{it.get('v_call','?')} "
                f"CDR3:{it.get('cdr3_aa','?')} "
                + (f"Ag:{it['antigen']}" if it.get("antigen") else "")
            )
        return "\n".join(lines)
