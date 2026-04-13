"""
biovoice/agents/iedb_agent.py
Immune Epitope Database (IEDB) agent.

IEDB holds experimentally validated T-cell and B-cell epitopes, MHC binding
data, and antibody-antigen interaction records — the primary database for
bnAb epitope characterisation.

API: https://query.iedb.org/
  POST /epitope_search   (JSON body)
  POST /assay_search
  GET  /tcell_search / /bcell_search
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import requests

from .base import AgentConfig, BaseAgent, FetchResult


class IEDBAgent(BaseAgent):
    """Query IEDB for experimentally validated epitopes."""

    # IEDB free-text search (returns epitope records)
    SEARCH_URL  = "https://query.iedb.org/epitope_search"
    BCELL_URL   = "https://query.iedb.org/bcell_search"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent":   "BioVoice/1.0",
            "Content-Type": "application/json",
            "Accept":       "application/json",
        })
        self._delay = float(config.extra_params.get("delay", 0.5))

    def get_capabilities(self) -> List[str]:
        return ["epitope", "immunology", "antibody", "bcell", "tcell"]

    def get_default_prompt(self) -> str:
        return (
            "You are an immunologist specialising in epitope mapping. "
            "Based on these IEDB epitope records about {topic}, describe "
            "the key experimentally validated epitopes, their MHC restriction, "
            "and which antibodies recognise them."
        )

    async def fetch(self, query: str, limit: int = 200, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = self._build_context(items)
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    def _search(self, query: str, limit: int) -> List[Dict]:
        results: List[Dict] = []

        # B-cell / antibody epitopes — most relevant for bnAb research
        results.extend(self._bcell_search(query, limit))

        # Deduplicate by epitope_id
        seen, deduped = set(), []
        for r in results:
            key = r.get("epitope_id") or r.get("linear_sequence") or str(id(r))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        print(f"[IEDB] {len(deduped)} epitope records")
        return deduped[:limit]

    def _bcell_search(self, query: str, limit: int) -> List[Dict]:
        """
        Search B-cell assays. IEDB exposes a URL-encoded query interface.
        We query by antigen name using their REST endpoint.
        """
        # IEDB uses a specific URL-query format
        params = {
            "format":           "json",
            "antigen_name":     query,
            "organism_id":      11520,   # Influenza A
            "rows":             min(limit, 500),
            "start":            0,
        }
        items = []
        try:
            time.sleep(self._delay)
            # IEDB REST endpoint for B-cell assays
            url = "https://query.iedb.org/bcell"
            resp = self._session.get(url, params=params, timeout=30)

            if resp.status_code in (404, 405):
                # Fall back to epitope search endpoint
                return self._epitope_search(query, limit)

            resp.raise_for_status()
            data = resp.json()

            rows = data if isinstance(data, list) else data.get("data", [])
            for row in rows:
                items.append(self._normalise_bcell(row))

        except Exception as e:
            print(f"[IEDB] B-cell search failed: {e}")
            return self._epitope_search(query, limit)

        return items

    def _epitope_search(self, query: str, limit: int) -> List[Dict]:
        """
        Fallback: use the IEDB epitope full-text search.
        Returns epitope records (not assay-level).
        """
        params = {
            "format":        "json",
            "epitope_seq":   "",
            "antigen_name":  query,
            "rows":          min(limit, 500),
        }
        items = []
        try:
            time.sleep(self._delay)
            resp = self._session.get(
                "https://query.iedb.org/epitope", params=params, timeout=30
            )
            if not resp.ok:
                # Last resort: use the public IEDB table download
                return self._public_search(query, limit)
            resp.raise_for_status()
            data = resp.json()
            rows = data if isinstance(data, list) else data.get("data", [])
            for row in rows:
                items.append(self._normalise_epitope(row))
        except Exception as e:
            print(f"[IEDB] epitope search failed: {e}")
        return items

    def _public_search(self, query: str, limit: int) -> List[Dict]:
        """
        Use IEDB's public query builder endpoint.
        https://www.iedb.org/result_v3.php?query=...
        """
        items = []
        try:
            time.sleep(self._delay)
            url = "https://www.iedb.org/api/v1/epitope/"
            params = {
                "format":       "json",
                "search_query": query,
                "limit":        min(limit, 100),
                "offset":       0,
            }
            resp = self._session.get(url, params=params, timeout=20)
            if not resp.ok:
                print(f"[IEDB] public search returned {resp.status_code}")
                return []
            data = resp.json()
            for row in (data.get("objects") or data.get("results") or []):
                items.append(self._normalise_epitope(row))
        except Exception as e:
            print(f"[IEDB] public search failed: {e}")
        return items

    def _normalise_bcell(self, row: Dict) -> Dict:
        return {
            "epitope_id":        row.get("Epitope ID", row.get("epitope_id", "")),
            "linear_sequence":   row.get("Description", row.get("linear_sequence", "")),
            "antigen_name":      row.get("Antigen Name", row.get("antigen_name", "")),
            "organism":          row.get("Organism", ""),
            "mhc_allele":        "",
            "assay_type":        row.get("Assay Type", "B-cell"),
            "response":          row.get("Qualitative Measure", ""),
            "pmid":              str(row.get("Reference ID", row.get("pmid", ""))),
            "source":            "iedb",
            "title":             row.get("Antigen Name", ""),
            "abstract":          self._make_abstract(row),
            "year":              "",
            "citation_count":    0,
        }

    def _normalise_epitope(self, row: Dict) -> Dict:
        return {
            "epitope_id":      row.get("epitope_id", row.get("id", "")),
            "linear_sequence": row.get("linear_sequence", row.get("description", "")),
            "antigen_name":    row.get("antigen_name", row.get("object", {}).get("antigen_name", "")),
            "organism":        row.get("organism", ""),
            "mhc_allele":      row.get("mhc_allele", ""),
            "assay_type":      row.get("assay_type", ""),
            "response":        row.get("response", ""),
            "pmid":            str(row.get("pmid", row.get("reference_id", ""))),
            "source":          "iedb",
            "title":           row.get("antigen_name", "IEDB epitope"),
            "abstract":        self._make_abstract(row),
            "year":            "",
            "citation_count":  0,
        }

    def _make_abstract(self, row: Dict) -> str:
        seq  = row.get("Description") or row.get("linear_sequence") or row.get("description", "")
        ag   = row.get("Antigen Name") or row.get("antigen_name", "")
        org  = row.get("Organism") or row.get("organism", "")
        mhc  = row.get("MHC Allele") or row.get("mhc_allele", "")
        resp = row.get("Qualitative Measure") or row.get("response", "")
        return (
            f"IEDB epitope record. Antigen: {ag} ({org}). "
            f"Sequence: {seq}. "
            + (f"MHC restriction: {mhc}. " if mhc else "")
            + (f"Assay response: {resp}." if resp else "")
        )

    def _build_context(self, items: List[Dict]) -> str:
        lines = []
        for it in items[:50]:
            lines.append(
                f"[IEDB:{it.get('epitope_id','')}] {it.get('antigen_name','')} "
                f"| seq: {it.get('linear_sequence','')[:30]} "
                f"| assay: {it.get('assay_type','')} "
                f"| response: {it.get('response','')}"
                + (f" | PMID:{it['pmid']}" if it.get('pmid') else "")
            )
        return "\n".join(lines)
