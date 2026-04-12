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

# ── ChEMBL ────────────────────────────────────────────────────────────────────
 
class ChEMBLAgent(BaseAgent):
    """Query ChEMBL for small molecule compounds and bioactivity data."""
 
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
 
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
 
    def get_capabilities(self) -> List[str]:
        return ["chemical", "drug", "bioactivity"]
 
    def get_default_prompt(self) -> str:
        return (
            "You are a medicinal chemist. "
            "Describe the key small molecule inhibitors related to {topic}, "
            "including their potency, selectivity, and development stage."
        )
 
    async def fetch(self, query: str, limit: int = 20, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = "\n\n".join(
            f"[ChEMBL: {i['chembl_id']}] {i['name']}\n"
            f"Type: {i.get('type','N/A')} | Max phase: {i.get('max_phase','N/A')}\n"
            f"MW: {i.get('mw','N/A')} | AlogP: {i.get('alogp','N/A')}"
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
            "q":      query,
            "limit":  limit,
            "format": "json",
        }
        try:
            resp = self._session.get(
                f"{self.BASE_URL}/molecule/search",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            results = []
            for mol in resp.json().get("molecules", []):
                props = mol.get("molecule_properties") or {}
                results.append({
                    "chembl_id": mol.get("molecule_chembl_id", ""),
                    "name":      mol.get("pref_name") or mol.get("molecule_chembl_id", ""),
                    "type":      mol.get("molecule_type", ""),
                    "max_phase": mol.get("max_phase"),
                    "mw":        props.get("full_mwt"),
                    "alogp":     props.get("alogp"),
                })
            return results
        except Exception as e:
            print(f"[ChEMBLAgent] search failed: {e}")
            return []
 