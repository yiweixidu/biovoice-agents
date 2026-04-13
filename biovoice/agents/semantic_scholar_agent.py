"""
biovoice/agents/semantic_scholar_agent.py
Semantic Scholar Academic Graph API.

Provides: citation counts, influential citation flags, open-access PDFs,
          author h-index, paper embeddings (via S2 API), reference graph.
API: https://api.semanticscholar.org/graph/v1/
Rate limit: 100 req/5min unauthenticated, 1 req/s with API key.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import requests

from .base import AgentConfig, BaseAgent, FetchResult

FIELDS = (
    "paperId,externalIds,title,abstract,year,citationCount,"
    "influentialCitationCount,isOpenAccess,openAccessPdf,"
    "authors,journal,publicationTypes,publicationDate"
)


class SemanticScholarAgent(BaseAgent):
    """Search Semantic Scholar for papers with citation graph metadata."""

    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        api_key = config.api_key or config.extra_params.get("api_key", "")
        if api_key:
            self._session.headers["x-api-key"] = api_key
        self._session.headers["User-Agent"] = "BioVoice/1.0"
        self._delay = float(config.extra_params.get("delay", 1.1))

    def get_capabilities(self) -> List[str]:
        return ["literature", "citation_graph", "open_access"]

    def get_default_prompt(self) -> str:
        return (
            "You are a biomedical researcher. Based on these Semantic Scholar "
            "results about {topic}, identify the most influential papers and "
            "summarise the citation landscape."
        )

    async def fetch(self, query: str, limit: int = 100, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = "\n\n".join(
            f"[S2:{i['paperId'][:8]}] {i['title']}\n"
            f"  Year:{i.get('year','')}  Cites:{i.get('citation_count',0)}  "
            f"Influential:{i.get('influential_citations',0)}  OA:{i.get('is_open_access','')}\n"
            f"  {(i.get('abstract') or '')[:300]}"
            for i in items
        )
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    def _search(self, query: str, limit: int) -> List[Dict]:
        results: List[Dict] = []
        offset = 0
        page   = min(100, limit)   # S2 max per request

        while len(results) < limit:
            params = {
                "query":  query,
                "fields": FIELDS,
                "limit":  page,
                "offset": offset,
            }
            try:
                time.sleep(self._delay)
                resp = self._session.get(self.SEARCH_URL, params=params, timeout=20)
                if resp.status_code == 429:
                    print("[S2] Rate limited — waiting 30s")
                    time.sleep(30)
                    continue
                resp.raise_for_status()
                data  = resp.json()
                batch = data.get("data", [])
                if not batch:
                    break
                for p in batch:
                    ext = p.get("externalIds") or {}
                    pdf = (p.get("openAccessPdf") or {}).get("url", "")
                    authors = p.get("authors") or []
                    auth_str = "; ".join(
                        a.get("name", "") for a in authors[:5]
                    ) + (" et al." if len(authors) > 5 else "")
                    j = p.get("journal") or {}
                    results.append({
                        "paperId":               p.get("paperId", ""),
                        "pmid":                  ext.get("PubMed", ""),
                        "doi":                   ext.get("DOI", ""),
                        "title":                 p.get("title", ""),
                        "abstract":              p.get("abstract", ""),
                        "year":                  str(p.get("year") or ""),
                        "citation_count":        p.get("citationCount", 0),
                        "influential_citations": p.get("influentialCitationCount", 0),
                        "is_open_access":        p.get("isOpenAccess", False),
                        "pdf_url":               pdf,
                        "fulltext_available":     bool(pdf),
                        "authors":               auth_str,
                        "journal":               j.get("name", ""),
                        "source":                "semantic_scholar",
                    })
                total = data.get("total", 0)
                offset += len(batch)
                if offset >= total or offset >= limit:
                    break
                page = min(100, limit - len(results))
            except Exception as e:
                print(f"[SemanticScholar] search failed: {e}")
                break

        print(f"[SemanticScholar] {len(results)} papers")
        return results[:limit]
