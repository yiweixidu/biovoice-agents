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
        """
        Cursor-paginated EuropePMC search.
        Each page is 1000 items (API max). Continues until `limit` is
        reached or no more results are available.
        No artificial cap when limit >= 9999 (i.e. "unlimited" mode).
        """
        import time

        PAGE = 1000
        results: List[Dict] = []
        cursor = "*"

        while True:
            params = {
                "query":      query,
                "format":     "json",
                "pageSize":   PAGE,
                "resultType": "core",
                "sort":       "CITED desc",
                "cursorMark": cursor,
            }
            try:
                time.sleep(self._delay)
                resp = self._session.get(self.SEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"[EuropePMCAgent] search failed: {e}")
                break

            page_items = data.get("resultList", {}).get("result", [])
            if not page_items:
                break

            for item in page_items:
                results.append({
                    "pmid":              item.get("pmid", ""),
                    "pmcid":             item.get("pmcid", ""),
                    "doi":               item.get("doi", ""),
                    "title":             item.get("title", ""),
                    "abstract":          item.get("abstractText", ""),
                    "journal":           item.get("journalTitle", ""),
                    "year":              str(item.get("pubYear", "")),
                    "citation_count":    item.get("citedByCount", 0),
                    "source":            item.get("source", ""),
                    "cited_by":          item.get("citedByCount", 0),
                    "fulltext_available": False,
                    "fulltext_content":   None,
                    "fulltext_source":    None,
                })

            next_cursor = data.get("nextCursorMark")
            total_found = data.get("hitCount", 0)
            fetched     = len(results)

            print(f"[EuropePMC] fetched {fetched}/{total_found} "
                  f"(limit={limit if limit < 9999 else 'unlimited'})")

            # Stop if: hard limit reached, no next page, or got everything
            if limit < 9999 and fetched >= limit:
                break
            if not next_cursor or next_cursor == cursor:
                break
            if fetched >= total_found:
                break

            cursor = next_cursor

        return results[:limit] if limit < 9999 else results