"""
biovoice/agents/pubmed_agent.py
PubMed data source agent.
Wraps the existing PubMedFetcher + PMCFulltextFetcher from FluBroad-Voice
and exposes them through the BaseAgent interface.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

from .base import AgentConfig, BaseAgent, FetchResult

# Re-use the production-grade fetchers from FluBroad-Voice
from core.retrieval.pubmed import PubMedFetcher
from core.retrieval.pmc_fulltext import PMCFulltextFetcher, UnpaywallFetcher


class PubMedAgent(BaseAgent):
    """
    Fetches PubMed abstracts + optionally enriches with PMC full text
    and Unpaywall OA PDFs.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        email      = config.extra_params.get("email", "biovoice@example.com")
        api_key    = config.api_key
        pmc_delay  = float(config.extra_params.get("pmc_delay", 0.5))
        oa_delay   = float(config.extra_params.get("unpaywall_delay", 0.5))
        fetch_full = config.extra_params.get("fetch_fulltext", True)

        self._pubmed     = PubMedFetcher(email=email, api_key=api_key)
        self._pmc        = PMCFulltextFetcher(email=email, delay=pmc_delay)
        self._unpaywall  = UnpaywallFetcher(email=email, delay=oa_delay)
        self._fetch_full = fetch_full

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def get_capabilities(self) -> List[str]:
        return ["literature", "abstract", "fulltext"]

    def get_default_prompt(self) -> str:
        return (
            "You are a biomedical researcher. "
            "Based on the following PubMed literature, answer the question "
            "about {topic}. Cite PMIDs for every factual claim."
        )

    async def fetch(self, query: str, limit: int = 100, **kwargs) -> FetchResult:
        """
        Search PubMed, fetch details, and optionally enrich with full text.
        Runs synchronous network I/O in a thread pool to stay async-safe.
        """
        days_back = kwargs.get("days_back", 3650)

        # 1. Search + fetch abstracts (blocking → thread)
        articles = await asyncio.to_thread(
            self._fetch_articles, query, limit, days_back
        )

        # 2. Optional full-text enrichment
        if self._fetch_full:
            articles = await asyncio.to_thread(self._enrich, articles)

        # 3. Build LLM context string
        context_parts = []
        for art in articles:
            if art.get("fulltext_available") and art.get("fulltext_content"):
                body = art["fulltext_content"][:3000]  # cap per article
                src  = art.get("fulltext_source", "pmc")
            else:
                body = art.get("abstract", "")
                src  = "abstract"
            context_parts.append(
                f"[PMID: {art.get('pmid','')} | {src}] "
                f"{art.get('title','')}\n{body}"
            )

        ft_count = sum(1 for a in articles if a.get("fulltext_available"))

        return FetchResult(
            source=self.name,
            items=articles,
            metadata={
                "total":          len(articles),
                "fulltext_count": ft_count,
                "abstract_only":  len(articles) - ft_count,
                "query":          query,
            },
            prompt_context="\n\n---\n\n".join(context_parts),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_articles(
        self, query: str, limit: int, days_back: int
    ) -> List[Dict]:
        pmids = self._pubmed.search(query, max_results=limit, days_back=days_back)
        return self._pubmed.fetch_details(pmids)

    def _enrich(self, articles: List[Dict]) -> List[Dict]:
        for art in articles:
            pmcid = art.get("pmcid", "")
            doi   = art.get("doi", "")

            # Lookup PMCID via Europe PMC if PubMed didn't supply it
            if not pmcid and art.get("pmid"):
                found = self._pmc.lookup_pmcid(art["pmid"])
                if found:
                    art["pmcid"] = found
                    pmcid = found

            if pmcid and not art.get("fulltext_available"):
                self._pmc.enrich_article(art)

            if doi and not art.get("fulltext_available"):
                self._unpaywall.enrich_article(art)

        return articles