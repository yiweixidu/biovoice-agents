"""
biovoice/agents/biorxiv_agent.py
bioRxiv + medRxiv preprint server agent.

Fetches the latest preprints via the bioRxiv/medRxiv REST API.
No auth required.  Rate limit: ~1 req/s polite.

API: https://api.biorxiv.org/details/{server}/{interval}/{cursor}/{format}
  server   : biorxiv | medrxiv
  interval : date range "YYYY-MM-DD/YYYY-MM-DD" OR a single doi
  cursor   : pagination offset (integer)
  format   : json

For keyword search we use the JATS full-text search endpoint:
  GET https://api.biorxiv.org/details/biorxiv/{interval}/0/json
  (interval = "2000-01-01/2099-01-01" for all time)

Note: biorxiv API does NOT support keyword search natively.
We use the content search via:
  GET https://www.biorxiv.org/search/{query}?limit_from=0&limit=75&format=json
which returns preprints matching a query.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, timedelta
from typing import Dict, List

import requests

from .base import AgentConfig, BaseAgent, FetchResult


class BioRxivAgent(BaseAgent):
    """Fetch recent preprints from bioRxiv and medRxiv."""

    # Content search (keyword)
    CONTENT_URL  = "https://api.biorxiv.org/details/{server}/{start}/{cursor}/json"
    # Published papers with DOI mapping
    PUBLISHED_URL = "https://api.biorxiv.org/publisher/{doi_prefix}/0/json"

    SEARCH_URL   = "https://www.biorxiv.org/search/{query}%20numresults%3A{n}%20sort%3Arelevance-rank%20format_result%3Astandard?format=json"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "BioVoice/1.0"})
        self._delay     = float(config.extra_params.get("delay", 0.5))
        self._servers   = config.extra_params.get("servers", "biorxiv,medrxiv").split(",")
        self._days_back = int(config.extra_params.get("days_back", 730))   # 2 years

    def get_capabilities(self) -> List[str]:
        return ["preprint", "literature", "open_access"]

    def get_default_prompt(self) -> str:
        return (
            "You are a biomedical researcher scanning preprints. "
            "Based on these bioRxiv/medRxiv preprints about {topic}, "
            "identify emerging findings not yet in peer-reviewed literature."
        )

    async def fetch(self, query: str, limit: int = 100, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._fetch_all_servers, query, limit)
        context = "\n\n".join(
            f"[{i.get('source','biorxiv').upper()}:{i.get('doi','')}] {i['title']}\n"
            f"  Posted: {i.get('year','')}  Category: {i.get('category','')}\n"
            f"  {(i.get('abstract') or '')[:300]}"
            for i in items
        )
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    def _fetch_all_servers(self, query: str, limit: int) -> List[Dict]:
        results: List[Dict] = []
        per_server = max(1, limit // len(self._servers))
        for server in self._servers:
            batch = self._fetch_server(server.strip(), query, per_server)
            results.extend(batch)
        # Deduplicate by DOI
        seen, deduped = set(), []
        for r in results:
            key = r.get("doi") or r.get("title", "")
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:limit]

    def _fetch_server(self, server: str, query: str, limit: int) -> List[Dict]:
        """
        The bioRxiv API doesn't have keyword search, so we fetch recent
        papers in a date window and filter by query terms client-side.
        For broader coverage we fetch 90-day windows.
        """
        end   = date.today().isoformat()
        start = (date.today() - timedelta(days=self._days_back)).isoformat()
        interval = f"{start}/{end}"

        results: List[Dict] = []
        cursor = 0
        query_terms = set(query.lower().split())

        while len(results) < limit:
            url = self.CONTENT_URL.format(
                server=server, start=interval, cursor=cursor
            )
            try:
                time.sleep(self._delay)
                resp = self._session.get(url, timeout=20)
                resp.raise_for_status()
                data       = resp.json()
                collection = data.get("collection", [])
                if not collection:
                    break

                for p in collection:
                    title_abs = (
                        (p.get("title") or "") + " " +
                        (p.get("abstract") or "")
                    ).lower()
                    # Relevance filter: at least 2 query terms must appear
                    hits = sum(1 for t in query_terms if t in title_abs)
                    if hits < 2:
                        continue
                    results.append(self._normalise(p, server))

                cursor += len(collection)
                messages = data.get("messages", [{}])
                total    = int((messages[0] if messages else {}).get("total", 0))
                if cursor >= total:
                    break

            except Exception as e:
                print(f"[BioRxiv:{server}] fetch failed: {e}")
                break

        print(f"[BioRxiv:{server}] {len(results)} relevant preprints")
        return results[:limit]

    def _normalise(self, p: Dict, server: str) -> Dict:
        doi  = p.get("doi", "")
        date_str = p.get("date", "")
        return {
            "doi":               doi,
            "pmid":              "",
            "title":             p.get("title", ""),
            "abstract":          p.get("abstract", ""),
            "authors":           p.get("authors", ""),
            "year":              date_str[:4],
            "citation_count":    0,
            "journal":           f"{server.title()} preprint",
            "category":          p.get("category", ""),
            "server":            server,
            "source":            server,
            "fulltext_available": True,   # all preprints are open access
            "pdf_url":           f"https://www.biorxiv.org/content/{doi}.full.pdf" if doi else "",
        }
