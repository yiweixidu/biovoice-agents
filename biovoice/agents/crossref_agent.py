"""
biovoice/agents/crossref_agent.py
Crossref REST API agent.

Crossref indexes 150M+ DOI-registered works across all disciplines.
Complements PubMed for: conference papers, book chapters, non-MEDLINE
journals, and provides real citation counts via `is-referenced-by-count`.

API: https://api.crossref.org/works?query=...
Polite pool: add ?mailto=email for higher rate limits.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import requests

from .base import AgentConfig, BaseAgent, FetchResult


class CrossrefAgent(BaseAgent):
    """Query Crossref for DOI-registered publications."""

    SEARCH_URL = "https://api.crossref.org/works"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        email = config.extra_params.get("email", "biovoice@example.com")
        self._session.headers.update({
            "User-Agent": f"BioVoice/1.0 (mailto:{email})",
        })
        self._delay = float(config.extra_params.get("delay", 0.5))
        self._email = email

    def get_capabilities(self) -> List[str]:
        return ["literature", "citation_count", "doi"]

    def get_default_prompt(self) -> str:
        return (
            "You are a biomedical researcher. Based on these Crossref results "
            "about {topic}, identify key publications and citation patterns."
        )

    async def fetch(self, query: str, limit: int = 200, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = "\n\n".join(
            f"[DOI:{i.get('doi','')}] {i['title']}\n"
            f"  Year:{i.get('year','')}  Cites:{i.get('citation_count',0)}  "
            f"Type:{i.get('type','')}\n"
            f"  {(i.get('abstract') or '')[:200]}"
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
        rows   = min(200, limit)   # Crossref max per request

        while len(results) < limit:
            params = {
                "query":            query,
                "rows":             rows,
                "offset":           offset,
                "mailto":           self._email,
                "select": (
                    "DOI,title,abstract,author,published,type,"
                    "is-referenced-by-count,container-title,ISSN"
                ),
                "sort":  "relevance",
                "order": "desc",
            }
            try:
                time.sleep(self._delay)
                resp = self._session.get(self.SEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                data  = resp.json().get("message", {})
                items = data.get("items", [])
                if not items:
                    break

                for p in items:
                    results.append(self._normalise(p))

                total  = data.get("total-results", 0)
                offset += len(items)
                if offset >= total or offset >= limit:
                    break
                rows = min(200, limit - len(results))

            except Exception as e:
                print(f"[Crossref] search failed: {e}")
                break

        print(f"[Crossref] {len(results)} works")
        return results[:limit]

    def _normalise(self, p: Dict) -> Dict:
        # Title can be a list
        titles = p.get("title") or []
        title  = titles[0] if titles else ""

        # Author list
        authors_raw = p.get("author") or []
        auth_parts  = []
        for a in authors_raw[:5]:
            name = a.get("family", "") + (f" {a.get('given','')[0]}." if a.get("given") else "")
            auth_parts.append(name.strip())
        authors = "; ".join(auth_parts) + (" et al." if len(authors_raw) > 5 else "")

        # Publication year
        pub = p.get("published") or p.get("published-print") or {}
        date_parts = pub.get("date-parts", [[None]])[0]
        year = str(date_parts[0]) if date_parts and date_parts[0] else ""

        # Journal
        containers = p.get("container-title") or []
        journal    = containers[0] if containers else ""

        return {
            "doi":             p.get("DOI", ""),
            "pmid":            "",
            "title":           title,
            "abstract":        p.get("abstract", ""),
            "authors":         authors,
            "year":            year,
            "citation_count":  p.get("is-referenced-by-count", 0),
            "journal":         journal,
            "type":            p.get("type", ""),
            "source":          "crossref",
            "fulltext_available": False,
        }
