"""
biovoice/agents/opentargets_agent.py
Open Targets Platform agent (GraphQL API).

Provides target-disease-drug associations, genetic evidence, and
safety/tractability data. Useful for translational context in grant writing.

API: https://api.platform.opentargets.org/api/v4/graphql
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import requests

from .base import AgentConfig, BaseAgent, FetchResult

_SEARCH_QUERY = """
query Search($q: String!, $n: Int!) {
  search(queryString: $q, page: {index: 0, size: $n}) {
    total
    hits {
      id
      entity
      name
      description
      score
    }
  }
}
"""

_DISEASE_ASSOC_QUERY = """
query DiseaseAssoc($diseaseId: String!, $n: Int!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: {index: 0, size: $n}) {
      count
      rows {
        target {
          id
          approvedSymbol
          approvedName
          biotype
        }
        score
        datatypeScores {
          componentId
          score
        }
      }
    }
  }
}
"""

_TARGET_ASSOC_QUERY = """
query TargetAssoc($targetId: String!, $n: Int!) {
  target(ensemblId: $targetId) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: 0, size: $n}) {
      count
      rows {
        disease {
          id
          name
          therapeuticAreas { id name }
        }
        score
      }
    }
    knownDrugs(size: $n) {
      count
      rows {
        prefName
        drugType
        mechanismOfAction
        phase
        disease { name }
        urls { url }
      }
    }
  }
}
"""


class OpenTargetsAgent(BaseAgent):
    """Query Open Targets Platform for target-disease-drug associations."""

    GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent":   "BioVoice/1.0",
            "Content-Type": "application/json",
        })
        self._delay = float(config.extra_params.get("delay", 0.5))

    def get_capabilities(self) -> List[str]:
        return ["target", "disease", "drug", "translational"]

    def get_default_prompt(self) -> str:
        return (
            "You are a translational scientist. Based on these Open Targets "
            "associations for {topic}, describe the target-disease evidence, "
            "clinical pipeline status, and tractability for drug development."
        )

    async def fetch(self, query: str, limit: int = 50, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search_and_fetch, query, limit)
        context = self._build_context(items)
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    def _gql(self, query: str, variables: Dict) -> Optional[Dict]:
        try:
            time.sleep(self._delay)
            resp = self._session.post(
                self.GRAPHQL_URL,
                json={"query": query, "variables": variables},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("data")
        except Exception as e:
            print(f"[OpenTargets] GraphQL failed: {e}")
            return None

    def _search_and_fetch(self, query: str, limit: int) -> List[Dict]:
        # Step 1: free-text search to find relevant targets/diseases
        data = self._gql(_SEARCH_QUERY, {"q": query, "n": min(limit, 25)})
        if not data:
            return []

        hits   = data.get("search", {}).get("hits", [])
        items  = []
        target_ids  = []
        disease_ids = []

        for hit in hits:
            if hit["entity"] == "target":
                target_ids.append(hit["id"])
            elif hit["entity"] == "disease":
                disease_ids.append(hit["id"])

        # Step 2: for each target, get disease associations + known drugs
        for tid in target_ids[:5]:
            assoc = self._gql(_TARGET_ASSOC_QUERY, {"targetId": tid, "n": 10})
            if not assoc:
                continue
            target = assoc.get("target", {})
            drugs  = target.get("knownDrugs", {}).get("rows", [])
            dis    = target.get("associatedDiseases", {}).get("rows", [])
            items.append({
                "entity":          "target",
                "id":              tid,
                "symbol":          target.get("approvedSymbol", ""),
                "name":            target.get("approvedName", ""),
                "diseases":        [r["disease"]["name"] for r in dis[:5]],
                "drugs":           [d["prefName"] for d in drugs[:5]],
                "drug_phases":     [d.get("phase", "") for d in drugs[:5]],
                "source":          "opentargets",
                "title":           target.get("approvedName", ""),
                "abstract":        self._target_abstract(target, dis, drugs),
                "year":            "",
                "citation_count":  0,
                "pmid":            "",
            })

        # Step 3: for each disease, get top associated targets
        for did in disease_ids[:3]:
            assoc = self._gql(_DISEASE_ASSOC_QUERY, {"diseaseId": did, "n": 10})
            if not assoc:
                continue
            disease = assoc.get("disease", {})
            rows    = disease.get("associatedTargets", {}).get("rows", [])
            items.append({
                "entity":         "disease",
                "id":             did,
                "name":           disease.get("name", ""),
                "top_targets":    [r["target"]["approvedSymbol"] for r in rows[:8]],
                "source":         "opentargets",
                "title":          disease.get("name", ""),
                "abstract":       self._disease_abstract(disease, rows),
                "year":           "",
                "citation_count": 0,
                "pmid":           "",
            })

        print(f"[OpenTargets] {len(items)} entities")
        return items[:limit]

    def _target_abstract(self, target, diseases, drugs) -> str:
        sym  = target.get("approvedSymbol", "")
        name = target.get("approvedName", "")
        dis  = ", ".join(r["disease"]["name"] for r in diseases[:3])
        dr   = "; ".join(
            f"{d['prefName']} (Phase {d.get('phase','?')})"
            for d in drugs[:3]
        )
        return (
            f"Open Targets: {name} ({sym}). "
            f"Associated diseases: {dis or 'none listed'}. "
            f"Known drugs in pipeline: {dr or 'none listed'}."
        )

    def _disease_abstract(self, disease, rows) -> str:
        name    = disease.get("name", "")
        targets = ", ".join(r["target"]["approvedSymbol"] for r in rows[:5])
        return (
            f"Open Targets disease: {name}. "
            f"Top associated targets: {targets or 'none listed'}."
        )

    def _build_context(self, items: List[Dict]) -> str:
        lines = []
        for it in items:
            if it["entity"] == "target":
                lines.append(
                    f"[OT Target:{it['symbol']}] {it['name']}\n"
                    f"  Diseases: {', '.join(it.get('diseases',[])[:3])}\n"
                    f"  Drugs: {', '.join(it.get('drugs',[])[:3])}"
                )
            else:
                lines.append(
                    f"[OT Disease:{it['id']}] {it['name']}\n"
                    f"  Top targets: {', '.join(it.get('top_targets',[])[:5])}"
                )
        return "\n\n".join(lines)
