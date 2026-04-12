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

# ── ClinicalTrials.gov ────────────────────────────────────────────────────────
 
class ClinicalTrialsAgent(BaseAgent):
    """Query ClinicalTrials.gov for trial status and results."""
 
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
 
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
 
    def get_capabilities(self) -> List[str]:
        return ["clinical", "trial"]
 
    def get_default_prompt(self) -> str:
        return (
            "You are a clinical research analyst. "
            "Summarise the clinical trial landscape for {topic}, "
            "noting phase distribution, primary endpoints, and recent results."
        )
 
    async def fetch(self, query: str, limit: int = 20, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search, query, limit)
        context = "\n\n".join(
            f"[NCT: {i['nct_id']}] {i['title']}\n"
            f"Phase: {i.get('phase','N/A')} | Status: {i.get('status','N/A')}\n"
            f"Condition: {i.get('condition','N/A')} | "
            f"Intervention: {i.get('intervention','N/A')}"
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
            "query.term": query,
            "pageSize":   limit,
            "format":     "json",
            "fields":     "NCTId,BriefTitle,Phase,OverallStatus,Condition,InterventionName",
        }
        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            studies = resp.json().get("studies", [])
            results = []
            for s in studies:
                pmod = s.get("protocolSection", {})
                ident = pmod.get("identificationModule", {})
                stat  = pmod.get("statusModule", {})
                desc  = pmod.get("descriptionModule", {})
                cond  = pmod.get("conditionsModule", {})
                arms  = pmod.get("armsInterventionsModule", {})
                iname = arms.get("interventions", [{}])[0].get("name", "") if arms.get("interventions") else ""
                results.append({
                    "nct_id":       ident.get("nctId", ""),
                    "title":        ident.get("briefTitle", ""),
                    "phase":        stat.get("phase", ""),
                    "status":       stat.get("overallStatus", ""),
                    "condition":    ", ".join(cond.get("conditions", [])),
                    "intervention": iname,
                })
            return results
        except Exception as e:
            print(f"[ClinicalTrialsAgent] search failed: {e}")
            return []
 
 