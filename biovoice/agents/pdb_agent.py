"""
biovoice/agents/pdb_agent.py       — RCSB PDB
biovoice/agents/uniprot_agent.py   — UniProt
biovoice/agents/clinicaltrials_agent.py
biovoice/agents/chembl_agent.py
 
All four agents follow the same pattern:
  - REST API call in _fetch_sync()
  - Wrapped in asyncio.to_thread() inside fetch()
  - Returns FetchResult with structured items + LLM context string
"""
 
from __future__ import annotations
 
import asyncio
import time
from typing import Dict, List
 
import requests
 
from .base import AgentConfig, BaseAgent, FetchResult


# ── RCSB PDB ──────────────────────────────────────────────────────────────────
 
class PDBAgent(BaseAgent):
    """Search RCSB PDB for protein structures matching a query."""
 
    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    ENTRY_URL  = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
 
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
 
    def get_capabilities(self) -> List[str]:
        return ["structure", "protein"]
 
    def get_default_prompt(self) -> str:
        return (
            "You are a structural biologist. "
            "Based on these PDB structures related to {topic}, "
            "describe the key binding interfaces and conserved features."
        )
 
    async def fetch(self, query: str, limit: int = 20, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = "\n\n".join(
            f"[PDB: {i['pdb_id']}] {i['title']}\n"
            f"Organism: {i.get('organism','N/A')} | "
            f"Method: {i.get('method','N/A')} | "
            f"Resolution: {i.get('resolution','N/A')} Å"
            for i in items
        )
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )
 
    def _search(self, query: str, limit: int) -> List[Dict]:
        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query},
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": limit}},
        }
        try:
            resp = self._session.post(
                self.SEARCH_URL, json=payload, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for hit in data.get("result_set", []):
                pdb_id = hit.get("identifier", "")
                detail = self._fetch_entry(pdb_id)
                results.append(detail)
            return results
        except Exception as e:
            print(f"[PDBAgent] search failed: {e}")
            return []
 
    def _fetch_entry(self, pdb_id: str) -> Dict:
        try:
            resp = self._session.get(
                self.ENTRY_URL.format(pdb_id=pdb_id), timeout=10
            )
            resp.raise_for_status()
            d = resp.json()
            struct = d.get("struct", {})
            expt   = d.get("exptl", [{}])[0]
            src    = d.get("rcsb_entry_info", {})
            return {
                "pdb_id":     pdb_id,
                "title":      struct.get("title", ""),
                "method":     expt.get("method", ""),
                "resolution": src.get("resolution_combined", [None])[0],
                "organism":   d.get("rcsb_entry_container_identifiers", {})
                               .get("assembly_ids", [""])[0],
            }
        except Exception:
            return {"pdb_id": pdb_id, "title": "", "method": "", "resolution": None}
 
 