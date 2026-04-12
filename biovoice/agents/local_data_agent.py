"""
biovoice/agents/local_data_agent.py  — user-uploaded files (PDF, FASTA, CSV)
biovoice/agents/europe_pmc_agent.py  — Europe PMC preprints + PubMed mirror
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
from pathlib import Path
from typing import Dict, List

import requests

from .base import AgentConfig, BaseAgent, FetchResult


# ── Local Data Agent ──────────────────────────────────────────────────────────

class LocalDataAgent(BaseAgent):
    """
    Ingests user-uploaded files from a local directory.
    Supported formats: .pdf, .fasta, .fa, .csv, .tsv, .txt
    The agent scans config.extra_params['data_dir'] and returns all
    parseable files as structured items.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._data_dir = Path(
            config.extra_params.get("data_dir", "./data/uploads")
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def get_capabilities(self) -> List[str]:
        return ["local", "pdf", "sequence", "tabular"]

    def get_default_prompt(self) -> str:
        return (
            "You are a research assistant. "
            "Based on the user's uploaded data files about {topic}, "
            "extract key findings and integrate them with the literature."
        )

    async def fetch(self, query: str, limit: int = 50, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._load_files)
        context_parts = [
            f"[Local file: {i['filename']}]\n{i['content'][:2000]}"
            for i in items[:limit]
        ]
        return FetchResult(
            source=self.name,
            items=items[:limit],
            metadata={"total": len(items), "data_dir": str(self._data_dir)},
            prompt_context="\n\n---\n\n".join(context_parts),
        )

    def _load_files(self) -> List[Dict]:
        results = []
        for path in sorted(self._data_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".pdf":
                    content = self._read_pdf(path)
                    ftype   = "pdf"
                elif suffix in (".fasta", ".fa", ".fna", ".faa"):
                    content = self._read_fasta(path)
                    ftype   = "fasta"
                elif suffix in (".csv", ".tsv"):
                    content = self._read_tabular(path)
                    ftype   = "tabular"
                elif suffix == ".txt":
                    content = path.read_text(encoding="utf-8", errors="replace")
                    ftype   = "text"
                else:
                    continue
                results.append({
                    "filename": path.name,
                    "type":     ftype,
                    "size_kb":  round(path.stat().st_size / 1024, 1),
                    "content":  content,
                })
            except Exception as e:
                print(f"[LocalDataAgent] Failed to read {path.name}: {e}")
        return results

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
            return "\n\n".join(parts)
        except ImportError:
            return f"[pdfplumber not installed — cannot read {path.name}]"

    @staticmethod
    def _read_fasta(path: Path) -> str:
        text  = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        seqs  = []
        current_header = ""
        current_seq    = []
        for line in lines:
            if line.startswith(">"):
                if current_header:
                    seqs.append(f"{current_header}\nLength: {len(''.join(current_seq))} aa/nt")
                current_header = line
                current_seq    = []
            else:
                current_seq.append(line.strip())
        if current_header:
            seqs.append(f"{current_header}\nLength: {len(''.join(current_seq))} aa/nt")
        return "\n".join(seqs[:50])  # cap at 50 sequences

    @staticmethod
    def _read_tabular(path: Path) -> str:
        dialect = "excel-tab" if path.suffix == ".tsv" else "excel"
        text    = path.read_text(encoding="utf-8", errors="replace")
        reader  = csv.DictReader(io.StringIO(text), dialect=dialect)
        rows    = list(reader)[:100]
        if not rows:
            return ""
        headers = list(rows[0].keys())
        lines   = ["\t".join(headers)]
        for row in rows:
            lines.append("\t".join(str(row.get(h, "")) for h in headers))
        return "\n".join(lines)

