"""
biovoice/agents/alphafold_agent.py
AlphaFold EBI agent — fetches predicted protein structures for proteins
relevant to the research query.

Strategy
--------
1. Extract candidate UniProt accessions from the query by searching the
   UniProt REST API for proteins matching the query terms.
2. For each accession, hit the AlphaFold prediction API to retrieve
   structure metadata (mean pLDDT, PDB/mmCIF download URLs, model version).
3. Return structured items that include:
     - accession, gene, organism
     - mean_plddt (overall confidence 0-100)
     - fragment count (multidomain proteins may have multiple models)
     - pdb_url, cif_url  (direct download links)
     - af_version (AF2 vs AF3)
   plus a text context string for LLM synthesis.

API reference
-------------
  GET https://alphafold.ebi.ac.uk/api/prediction/{accession}
  Returns a list of model entries (one per fragment / version).

  GET https://alphafold.ebi.ac.uk/api/search
      ?search_term=<term>&type=PROTEIN&start=0&rows=<n>
  Full-text search over protein names in the AlphaFold DB.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import requests

from .base import AgentConfig, BaseAgent, FetchResult


class AlphaFoldAgent(BaseAgent):
    """
    Fetch AlphaFold structure predictions for proteins related to the query.

    Two-step process:
      1. Search AlphaFold DB by keyword to discover relevant accessions.
      2. Fetch per-accession prediction metadata.
    """

    SEARCH_URL  = "https://alphafold.ebi.ac.uk/api/search"
    PREDICT_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    AF_PAGE_URL = "https://alphafold.ebi.ac.uk/entry/{accession}"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        email = config.extra_params.get("email", "biovoice@example.com")
        self._session.headers.update({
            "User-Agent": f"BioVoice/1.0 ({email})",
            "Accept":     "application/json",
        })
        self._delay = float(config.extra_params.get("delay", 0.5))

    def get_capabilities(self) -> List[str]:
        return ["structure", "alphafold", "protein", "plddt"]

    def get_default_prompt(self) -> str:
        return (
            "You are a structural biologist specialising in AI-predicted "
            "protein structures. Based on these AlphaFold predictions about "
            "{topic}, describe the structural confidence, disordered regions, "
            "and implications for antibody epitope accessibility."
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

    # ── Internal methods ──────────────────────────────────────────────────────

    def _search_and_fetch(self, query: str, limit: int) -> List[Dict]:
        accessions = self._search_accessions_via_uniprot(query, min(limit, 50))
        if not accessions:
            print(f"[AlphaFold] No UniProt accessions found for: {query!r}")
            return []

        results = []
        for acc in accessions:
            time.sleep(self._delay)
            entry = self._fetch_prediction(acc)
            if entry:
                results.append(entry)

        print(f"[AlphaFold] {len(results)}/{len(accessions)} accessions "
              f"have AlphaFold models")
        return results

    def _search_accessions_via_uniprot(self, query: str, limit: int) -> List[str]:
        """
        Use the UniProt REST API (which is reliable) to discover relevant
        protein accessions, then look each one up in AlphaFold.

        Filters to reviewed (Swiss-Prot) entries first for quality, falls back
        to TrEMBL if fewer than `limit` reviewed hits exist.
        """
        UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
        accessions: List[str] = []

        for reviewed in ("true", "false"):
            if len(accessions) >= limit:
                break
            want = limit - len(accessions)
            params = {
                "query":   f"({query}) AND (reviewed:{reviewed})",
                "format":  "json",
                "fields":  "accession",
                "size":    min(want, 500),
            }
            try:
                time.sleep(self._delay)
                resp = self._session.get(UNIPROT_URL, params=params, timeout=20)
                resp.raise_for_status()
                for result in resp.json().get("results", []):
                    acc = result.get("primaryAccession")
                    if acc and acc not in accessions:
                        accessions.append(acc)
            except Exception as e:
                print(f"[AlphaFold] UniProt accession search failed: {e}")

        print(f"[AlphaFold] {len(accessions)} UniProt accessions to query")
        return accessions

    def _fetch_prediction(self, accession: str) -> Optional[Dict]:
        """
        Fetch AlphaFold prediction metadata for a single UniProt accession.
        Returns None if no prediction exists.
        """
        url = self.PREDICT_URL.format(accession=accession)
        try:
            resp = self._session.get(url, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            models = resp.json()   # list of model entries per fragment

            if not models:
                return None

            # Take the highest-version model (last in list)
            m = models[-1]

            # Aggregate mean pLDDT across fragments
            plddt_values = [
                entry.get("globalMetricValue")
                for entry in models
                if entry.get("globalMetricValue") is not None
            ]
            mean_plddt = (
                round(sum(plddt_values) / len(plddt_values), 1)
                if plddt_values else None
            )

            return {
                # Identifiers
                "accession":            accession,
                "source":               "alphafold",
                "gene":                 m.get("gene", ""),
                "protein_name":         m.get("uniprotDescription", ""),
                "organism":             m.get("organismScientificName", ""),
                "tax_id":               m.get("taxId"),
                # Structure quality
                "mean_plddt":           mean_plddt,
                "frac_very_high":       m.get("fractionPlddtVeryHigh"),   # >90
                "frac_confident":       m.get("fractionPlddtConfident"),  # 70-90
                "frac_low":             m.get("fractionPlddtLow"),        # 50-70
                "frac_very_low":        m.get("fractionPlddtVeryLow"),    # <50
                "fragment_count":       len(models),
                "af_version":           m.get("latestVersion", ""),
                "tool_used":            m.get("toolUsed", ""),            # AF2/AF3
                "model_created":        m.get("modelCreatedDate", ""),
                # Download URLs
                "pdb_url":              m.get("pdbUrl", ""),
                "cif_url":              m.get("cifUrl", ""),
                "pae_image_url":        m.get("paeImageUrl", ""),
                "pae_doc_url":          m.get("paeDocUrl", ""),
                "af_page":              self.AF_PAGE_URL.format(accession=accession),
                # Corpus-ranker compatible fields
                "title":                m.get("uniprotDescription", ""),
                "abstract":             self._make_abstract(m, mean_plddt, len(models)),
                "year":                 (m.get("modelCreatedDate") or "")[:4] or None,
                "citation_count":       0,
            }
        except Exception as e:
            print(f"[AlphaFold] prediction fetch failed for {accession}: {e}")
            return None

    def _make_abstract(self, model: Dict, mean_plddt: Optional[float],
                       fragment_count: int) -> str:
        """Synthesise a text description for LLM context and corpus ranking."""
        gene    = model.get("gene", "unknown gene")
        org     = model.get("organismScientificName", "unknown organism")
        name    = model.get("uniprotDescription", "")
        conf    = (
            f"mean pLDDT {mean_plddt:.1f}/100"
            if mean_plddt is not None else "confidence not reported"
        )
        frags   = (
            f"{fragment_count} fragment model(s)"
            if fragment_count > 1 else "single-chain model"
        )
        return (
            f"AlphaFold predicted structure for {name} ({gene}) from {org}. "
            f"Overall structural confidence: {conf}. {frags}. "
            f"pLDDT > 90 indicates high confidence (ordered regions); "
            f"pLDDT 70-90 generally confident; < 70 suggests disordered loops "
            f"or regions with limited template coverage."
        )

    def _build_context(self, items: List[Dict]) -> str:
        lines = []
        for it in items:
            plddt = it.get("mean_plddt")
            conf  = f"{plddt:.1f}" if plddt else "N/A"
            lines.append(
                f"[AlphaFold:{it['accession']}] {it.get('protein_name','')} "
                f"({it.get('gene','')}, {it.get('organism','')})\n"
                f"  mean pLDDT={conf}  fragments={it.get('fragment_count',1)}  "
                f"version={it.get('af_version','')}  "
                f"PDB: {it.get('pdb_url','N/A')}"
            )
        return "\n\n".join(lines)
