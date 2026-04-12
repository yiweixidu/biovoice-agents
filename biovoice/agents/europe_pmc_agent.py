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

# ── Europe PMC Agent ──────────────────────────────────────────────────────────

class EuropePMCAgent(BaseAgent):
    """
    Query Europe PMC for literature including preprints (bioRxiv, medRxiv)
    and open-access full text.
    """

    BASE_URL   = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    SEARCH_URL = f"{BASE_URL}/search"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        email = config.extra_params.get("email", "biovoice@example.com")
        self._session.headers.update({"User-Agent": f"BioVoice/1.0 ({email})"})
        self._delay = float(config.extra_params.get("delay", 0.3))

    def get_capabilities(self) -> List[str]:
        return ["literature", "preprint", "abstract"]

    def get_default_prompt(self) -> str:
        return (
            "You are a biomedical researcher. "
            "Based on these Europe PMC results about {topic}, "
            "summarise key findings and cite PMIDs or DOIs."
        )

    async def fetch(self, query: str, limit: int = 50, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = "\n\n".join(
            f"[PMID: {i.get('pmid','N/A')} | {i.get('source','europepmc')}] "
            f"{i.get('title','')}\n{i.get('abstract','')[:500]}"
            for i in items
        )
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    def _search(self, query: str, limit: int) -> List[Dict]:
        params = {
            "query":      query,
            "format":     "json",
            "pageSize":   min(limit, 1000),
            "resultType": "core",
            "sort":       "CITED desc",
        }
        import time
        try:
            time.sleep(self._delay)
            resp = self._session.get(self.SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            results = []
            for item in resp.json().get("resultList", {}).get("result", []):
                results.append({
                    "pmid":     item.get("pmid", ""),
                    "pmcid":    item.get("pmcid", ""),
                    "doi":      item.get("doi", ""),
                    "title":    item.get("title", ""),
                    "abstract": item.get("abstractText", ""),
                    "journal":  item.get("journalTitle", ""),
                    "year":     str(item.get("pubYear", "")),
                    "source":   item.get("source", ""),
                    "cited_by": item.get("citedByCount", 0),
                    "fulltext_available": False,
                    "fulltext_content":   None,
                    "fulltext_source":    None,
                })
            return results
        except Exception as e:
            print(f"[EuropePMCAgent] search failed: {e}")
            return []