"""
biovoice/agents/pubchem_agent.py
PubChem small-molecule agent.

PubChem is the world's largest collection of freely accessible chemical
information, covering 115M+ compounds, bioassay data, and target interactions.
Useful for inhibitor scaffolds, drug candidates, and mechanistic context for
antiviral research.

API: https://pubchem.ncbi.nlm.nih.gov/rest/pug/
  compound/name/{name}/JSON           — compound lookup
  compound/fastsimilarity_2d/...      — similarity search
  bioassay/target/genesymbol/{gene}   — assays for a gene target
  compound/cid/{cid}/assaysummary/JSON — bioassay data for a compound

Also uses the PubChem PUG-REST compound search by keyword:
  GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/cids/JSON
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import requests

from .base import AgentConfig, BaseAgent, FetchResult


class PubChemAgent(BaseAgent):
    """
    Fetch small-molecule and bioassay data from PubChem.
    Returns compound records with bioactivity summary useful for
    antiviral compound screening context.
    """

    BASE_URL    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    SUGGEST_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "BioVoice/1.0"})
        self._delay     = float(config.extra_params.get("delay", 0.5))
        self._max_cids  = int(config.extra_params.get("max_cids", 100))

    def get_capabilities(self) -> List[str]:
        return ["chemistry", "drug", "compound", "bioassay", "smallmolecule"]

    def get_default_prompt(self) -> str:
        return (
            "You are a medicinal chemist analysing compound activity data. "
            "Based on these PubChem compound records about {topic}, describe "
            "the chemical scaffolds, bioactivity profiles, and their relevance "
            "to antiviral drug development."
        )

    async def fetch(self, query: str, limit: int = 50, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search_compounds, query, limit)
        context = self._build_context(items)
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _search_compounds(self, query: str, limit: int) -> List[Dict]:
        """
        Two-stage approach:
        1. PUG-REST keyword → CIDs
        2. Per-CID property + bioassay fetch
        """
        cids = self._keyword_to_cids(query, limit)
        if not cids:
            # Fallback: try gene-target assay search
            cids = self._gene_target_cids(query, limit)

        items: List[Dict] = []
        for cid in cids[:self._max_cids]:
            record = self._fetch_compound(cid)
            if record:
                items.append(record)
            if len(items) >= limit:
                break

        print(f"[PubChem] {len(items)} compound records")
        return items

    def _keyword_to_cids(self, query: str, limit: int) -> List[int]:
        """
        PUG-REST compound name/keyword → list of CIDs.
        Falls back to autocomplete if direct name lookup fails.
        """
        # Try direct name search first
        url = f"{self.BASE_URL}/compound/name/{requests.utils.quote(query)}/cids/JSON"
        try:
            time.sleep(self._delay)
            resp = self._session.get(url, timeout=20)
            if resp.ok:
                data = resp.json()
                return data.get("IdentifierList", {}).get("CID", [])[:limit]
        except Exception as e:
            print(f"[PubChem] name search failed: {e}")

        # Fallback: use PubChem's full-text compound search
        return self._fulltext_search(query, limit)

    def _fulltext_search(self, query: str, limit: int) -> List[int]:
        """
        PubChem PUG-REST full-text search for compound CIDs.
        Endpoint: /compound/fastsubstructure/smarts or keyword search via pugrest.
        We use the dedicated compound text search endpoint.
        """
        url = f"{self.BASE_URL}/compound/name/{requests.utils.quote(query)}/cids/JSON"
        params = {"name_type": "word"}
        try:
            time.sleep(self._delay)
            resp = self._session.get(url, params=params, timeout=20)
            if resp.ok:
                data = resp.json()
                return data.get("IdentifierList", {}).get("CID", [])[:limit]
        except Exception as e:
            print(f"[PubChem] fulltext search failed: {e}")
        return []

    def _gene_target_cids(self, query: str, limit: int) -> List[int]:
        """
        Search bioassays by gene symbol and extract active compound CIDs.
        PubChem gene target assay endpoint.
        """
        # Extract likely gene/protein name (first word or two)
        gene = query.split()[0].upper() if query else ""
        if not gene:
            return []

        url = f"{self.BASE_URL}/bioassay/target/genesymbol/{gene}/aids/JSON"
        try:
            time.sleep(self._delay)
            resp = self._session.get(url, timeout=20)
            if not resp.ok:
                return []
            aids = resp.json().get("IdentifierList", {}).get("AID", [])[:5]
        except Exception as e:
            print(f"[PubChem] gene target search failed: {e}")
            return []

        cids: List[int] = []
        for aid in aids:
            url2 = f"{self.BASE_URL}/assay/aid/{aid}/cids/JSON"
            params = {"cids_type": "active", "list_return": "listkey"}
            try:
                time.sleep(self._delay)
                r = self._session.get(url2, timeout=20)
                if r.ok:
                    batch = r.json().get("IdentifierList", {}).get("CID", [])
                    cids.extend(batch[:10])
                    if len(cids) >= limit:
                        break
            except Exception:
                continue
        return cids[:limit]

    def _fetch_compound(self, cid: int) -> Optional[Dict]:
        """
        Fetch compound properties from PubChem for a given CID.
        Returns a corpus-compatible record.
        """
        props = [
            "IUPACName", "MolecularFormula", "MolecularWeight",
            "IsomericSMILES", "CanonicalSMILES",
            "XLogP", "HBondDonorCount", "HBondAcceptorCount",
            "RotatableBondCount", "TPSA",
        ]
        url = (
            f"{self.BASE_URL}/compound/cid/{cid}/property/"
            f"{','.join(props)}/JSON"
        )
        try:
            time.sleep(self._delay)
            resp = self._session.get(url, timeout=20)
            if not resp.ok:
                return None
            data = resp.json()
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if not props_list:
                return None
            p = props_list[0]
        except Exception as e:
            print(f"[PubChem] compound {cid} fetch failed: {e}")
            return None

        # Optionally fetch synonym (common name)
        synonym = self._fetch_synonym(cid)

        iupac  = p.get("IUPACName", "")
        name   = synonym or iupac or f"CID {cid}"
        formula = p.get("MolecularFormula", "")
        mw      = p.get("MolecularWeight", "")
        smiles  = p.get("IsomericSMILES") or p.get("CanonicalSMILES", "")
        xlogp   = p.get("XLogP", "")
        tpsa    = p.get("TPSA", "")
        hbd     = p.get("HBondDonorCount", "")
        hba     = p.get("HBondAcceptorCount", "")

        abstract = (
            f"PubChem CID {cid}: {name}. "
            f"Formula: {formula}, MW: {mw} g/mol. "
            f"SMILES: {smiles[:80]}{'...' if len(smiles) > 80 else ''}. "
            + (f"XLogP: {xlogp}, TPSA: {tpsa} Å², HBD: {hbd}, HBA: {hba}." if xlogp else "")
        )

        return {
            "cid":              cid,
            "pmid":             "",
            "doi":              "",
            "title":            name,
            "abstract":         abstract,
            "iupac_name":       iupac,
            "molecular_formula": formula,
            "molecular_weight": mw,
            "smiles":           smiles,
            "xlogp":            xlogp,
            "tpsa":             tpsa,
            "hbd":              hbd,
            "hba":              hba,
            "year":             "",
            "citation_count":   0,
            "source":           "pubchem",
            "fulltext_available": False,
        }

    def _fetch_synonym(self, cid: int) -> Optional[str]:
        """Return the first (most common) synonym for a CID."""
        url = f"{self.BASE_URL}/compound/cid/{cid}/synonyms/JSON"
        try:
            time.sleep(self._delay * 0.5)
            resp = self._session.get(url, timeout=10)
            if resp.ok:
                syns = (
                    resp.json()
                    .get("InformationList", {})
                    .get("Information", [{}])[0]
                    .get("Synonym", [])
                )
                if syns:
                    return syns[0]
        except Exception:
            pass
        return None

    def _build_context(self, items: List[Dict]) -> str:
        lines = []
        for it in items:
            lines.append(
                f"[PubChem CID:{it.get('cid','')}] {it['title']}\n"
                f"  Formula: {it.get('molecular_formula','')}  "
                f"MW: {it.get('molecular_weight','')}  "
                f"XLogP: {it.get('xlogp','')}  TPSA: {it.get('tpsa','')}\n"
                f"  SMILES: {str(it.get('smiles',''))[:60]}"
            )
        return "\n\n".join(lines)
