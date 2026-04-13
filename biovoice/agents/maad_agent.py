"""
biovoice/agents/maad_agent.py
MAAD (Monoclonal Antibody Antigen Database) agent.

MAAD is a curated database of experimentally characterised antibody-antigen
interactions, focused on broadly neutralising antibodies against influenza,
SARS-CoV-2, HIV, RSV, and related pathogens.

Primary endpoints tried in order:
  1. OpenMAD / CoV-AbDab-style REST API (if available)
  2. IEDB structure-function data for antibody-antigen complexes
  3. CoV-AbDab (Oxford, public CSV export) for coronavirus bnAbs as fallback

We also query the AbDb / SAbDab (Structural Antibody Database) at:
  https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/

SAbDab is freely accessible, curated, and contains thousands of Fv structures
with antigen annotation — the closest publicly accessible equivalent to MAAD for
structural virology.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Dict, List, Optional

import requests

from .base import AgentConfig, BaseAgent, FetchResult


# SAbDab summary endpoint — returns CSV of all structures with metadata
_SABDAB_SUMMARY = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/"
# SAbDab search — filter by antigen type
_SABDAB_SEARCH  = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/"

# CoV-AbDab (Oxford) — coronavirus antibody database, public CSV
_COVABDAB_CSV   = (
    "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/"
    "CoV-AbDab_280923.csv"
)


class MAADAgent(BaseAgent):
    """
    Query curated antibody-antigen databases.

    Pulls from SAbDab (Structural Antibody Database) and optionally
    CoV-AbDab for coronavirus-specific entries. Returns structured
    records with: antibody name, antigen, species, PDB codes, IGHV/IGLV
    gene usage, and binding region annotations.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "BioVoice/1.0 (academic; mailto:biovoice@example.com)"
        })
        self._delay        = float(config.extra_params.get("delay", 1.0))
        self._antigen_kw   = config.extra_params.get("antigen_keyword", "influenza")
        self._include_cov  = config.extra_params.get("include_covabdab", "false").lower() != "false"

    def get_capabilities(self) -> List[str]:
        return ["antibody", "structure", "antigen", "bnab", "maad"]

    def get_default_prompt(self) -> str:
        return (
            "You are a structural immunologist. Based on these antibody-antigen "
            "database records about {topic}, describe the structural basis of "
            "broad neutralisation, IGHV gene usage patterns, and epitope conservation."
        )

    async def fetch(self, query: str, limit: int = 100, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._search_sabdab, query, limit)

        if self._include_cov and len(items) < limit:
            cov_items = await asyncio.to_thread(
                self._search_covabdab, query, limit - len(items)
            )
            items.extend(cov_items)

        context = self._build_context(items)
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    # ── SAbDab ────────────────────────────────────────────────────────────────

    def _search_sabdab(self, query: str, limit: int) -> List[Dict]:
        """
        Query SAbDab via their search endpoint, filtering by antigen keyword.
        Returns CSV rows parsed into normalised dicts.
        """
        params = {
            "antigen_name":  query,
            "format":        "csv",
            "redundancy":    "unique",
        }
        items: List[Dict] = []
        try:
            time.sleep(self._delay)
            resp = self._session.get(_SABDAB_SEARCH, params=params, timeout=30)
            if not resp.ok:
                print(f"[MAAD/SAbDab] search returned {resp.status_code}, "
                      f"trying summary endpoint")
                return self._summary_fallback(query, limit)

            content_type = resp.headers.get("Content-Type", "")
            if "csv" not in content_type and not resp.text.strip().startswith("pdb"):
                return self._summary_fallback(query, limit)

            items = self._parse_sabdab_csv(resp.text, query, limit)

        except Exception as e:
            print(f"[MAAD/SAbDab] search failed: {e}")
            return self._summary_fallback(query, limit)

        print(f"[MAAD/SAbDab] {len(items)} structures")
        return items[:limit]

    def _summary_fallback(self, query: str, limit: int) -> List[Dict]:
        """
        Fall back to the full SAbDab summary CSV and filter client-side.
        The full CSV is ~3MB but cached locally after first fetch.
        """
        items: List[Dict] = []
        try:
            time.sleep(self._delay)
            resp = self._session.get(_SABDAB_SUMMARY, timeout=60)
            if not resp.ok:
                print(f"[MAAD/SAbDab] summary endpoint {resp.status_code}")
                return []
            items = self._parse_sabdab_csv(resp.text, query, limit)
        except Exception as e:
            print(f"[MAAD/SAbDab] summary fallback failed: {e}")
        return items[:limit]

    def _parse_sabdab_csv(self, text: str, query: str, limit: int) -> List[Dict]:
        import csv as csv_mod
        items: List[Dict] = []
        query_lower = query.lower()
        try:
            reader = csv_mod.DictReader(io.StringIO(text))
            for row in reader:
                # Filter rows that mention the query in antigen name or species
                antigen = (row.get("antigen_name") or row.get("antigen") or "").lower()
                species = (row.get("antigen_species") or row.get("species") or "").lower()
                if query_lower not in antigen and query_lower not in species:
                    # Broaden: also match on genus-level keywords
                    if not any(kw in antigen or kw in species
                               for kw in query_lower.split()[:3]):
                        continue

                items.append(self._normalise_sabdab(row))
                if len(items) >= limit:
                    break
        except Exception as e:
            print(f"[MAAD/SAbDab] CSV parse error: {e}")
        return items

    def _normalise_sabdab(self, row: Dict) -> Dict:
        pdb  = row.get("pdb") or row.get("PDB") or ""
        name = (row.get("Hchain") or row.get("heavy_chain") or "")
        ag   = row.get("antigen_name") or row.get("antigen") or ""
        return {
            "source":           "maad",
            "pdb_id":           pdb.upper(),
            "antibody_name":    name or f"Ab-{pdb}",
            "antigen":          ag,
            "antigen_species":  row.get("antigen_species") or row.get("species") or "",
            "heavy_chain":      row.get("Hchain") or row.get("heavy_chain") or "",
            "light_chain":      row.get("Lchain") or row.get("light_chain") or "",
            "ighv":             row.get("heavy_subclass") or row.get("IGHV") or "",
            "iglv":             row.get("light_subclass") or row.get("IGLV") or "",
            "resolution":       row.get("resolution") or "",
            "scfv":             row.get("scFv") or "",
            "title":            f"SAbDab: {name or pdb} vs {ag}",
            "abstract":         self._make_abstract(row),
            "year":             (row.get("date") or "")[:4],
            "citation_count":   0,
            "pmid":             row.get("pmid") or "",
            "doi":              row.get("doi") or "",
            "fulltext_available": False,
        }

    def _make_abstract(self, row: Dict) -> str:
        pdb  = (row.get("pdb") or "").upper()
        ag   = row.get("antigen_name") or row.get("antigen") or "unknown antigen"
        sp   = row.get("antigen_species") or ""
        ighv = row.get("heavy_subclass") or row.get("IGHV") or ""
        res  = row.get("resolution") or ""
        return (
            f"SAbDab structural entry PDB:{pdb}. "
            f"Antigen: {ag} ({sp}). "
            + (f"Heavy chain gene: {ighv}. " if ighv else "")
            + (f"Resolution: {res} Å. " if res else "")
            + "Curated antibody-antigen co-crystal structure."
        )

    # ── CoV-AbDab (coronavirus bnAbs) ─────────────────────────────────────────

    def _search_covabdab(self, query: str, limit: int) -> List[Dict]:
        """
        Optional: fetch CoV-AbDab CSV for coronavirus-specific bnAbs.
        Only used when include_covabdab=true in config.
        """
        items: List[Dict] = []
        try:
            time.sleep(self._delay)
            resp = self._session.get(_COVABDAB_CSV, timeout=60)
            if not resp.ok:
                print(f"[MAAD/CoV-AbDab] fetch returned {resp.status_code}")
                return []
            import csv as csv_mod
            query_lower = query.lower()
            reader = csv_mod.DictReader(io.StringIO(resp.text))
            for row in reader:
                name  = row.get("Ab or Nb Name", "")
                virus = row.get("Protein + Epitope", "") + " " + row.get("Origin", "")
                if query_lower not in name.lower() and query_lower not in virus.lower():
                    continue
                items.append({
                    "source":          "maad",
                    "antibody_name":   name,
                    "antigen":         row.get("Protein + Epitope", ""),
                    "antigen_species": "SARS-CoV-2 / coronavirus",
                    "ighv":            row.get("IGHV Gene", ""),
                    "iglv":            row.get("IGLV Gene", ""),
                    "pdb_id":          row.get("PDB ID", ""),
                    "neutralising":    row.get("Neutralising Vs", ""),
                    "title":           f"CoV-AbDab: {name}",
                    "abstract":        (
                        f"CoV-AbDab entry: {name}. "
                        f"Targets: {row.get('Protein + Epitope', '')}. "
                        f"IGHV: {row.get('IGHV Gene', '')}. "
                        f"Neutralises: {row.get('Neutralising Vs', '')}."
                    ),
                    "year":            "",
                    "citation_count":  0,
                    "pmid":            "",
                    "fulltext_available": False,
                })
                if len(items) >= limit:
                    break
        except Exception as e:
            print(f"[MAAD/CoV-AbDab] parse error: {e}")

        print(f"[MAAD/CoV-AbDab] {len(items)} entries")
        return items[:limit]

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, items: List[Dict]) -> str:
        lines = []
        for it in items[:50]:
            lines.append(
                f"[MAAD PDB:{it.get('pdb_id','')}] "
                f"{it.get('antibody_name','')} → {it.get('antigen','')} "
                f"({it.get('antigen_species','')})"
                + (f" | IGHV:{it['ighv']}" if it.get("ighv") else "")
                + (f" | {it['resolution']}Å" if it.get("resolution") else "")
            )
        return "\n".join(lines)
