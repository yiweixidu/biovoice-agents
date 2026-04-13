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

# ── UniProt ───────────────────────────────────────────────────────────────────
 
class UniProtAgent(BaseAgent):
    """Query UniProt for protein function, pathways, and variants."""
 
    BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
 
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
 
    def get_capabilities(self) -> List[str]:
        return ["protein", "function", "pathway"]
 
    def get_default_prompt(self) -> str:
        return (
            "You are a protein biochemist. "
            "Summarise the function, disease associations, and key variants "
            "of the following proteins related to {topic}."
        )
 
    async def fetch(self, query: str, limit: int = 20, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = "\n\n".join(
            f"[UniProt: {i['accession']}] {i['name']}\n"
            f"Gene: {i.get('gene','N/A')} | Organism: {i.get('organism','N/A')}\n"
            f"Function: {i.get('function','N/A')}"
            for i in items
        )
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items)},
            prompt_context=context,
        )
 
    def _search(self, query: str, limit: int) -> List[Dict]:
        params = {
            "query": query,
            "format": "json",
            "size": min(limit, 500),   # UniProt API hard cap is 500
            "fields": "accession,gene_names,organism_name,protein_name,cc_function",
        }
        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            results = []
            for entry in resp.json().get("results", []):
                pname  = entry.get("proteinDescription", {})
                rec    = pname.get("recommendedName", {})
                fname  = rec.get("fullName", {}).get("value", "")
                genes  = entry.get("genes", [])
                gene   = genes[0].get("geneName", {}).get("value", "") if genes else ""
                org    = entry.get("organism", {}).get("scientificName", "")
                ccfn   = entry.get("comments", [])
                fn_txt = next(
                    (c.get("texts", [{}])[0].get("value", "")
                     for c in ccfn if c.get("commentType") == "FUNCTION"),
                    ""
                )
                results.append({
                    "accession": entry.get("primaryAccession", ""),
                    "name":      fname,
                    "gene":      gene,
                    "organism":  org,
                    "function":  fn_txt[:400],
                })
            return results
        except Exception as e:
            print(f"[UniProtAgent] search failed: {e}")
            return []
 
 