"""
biovoice/agents/flunet_agent.py
WHO FluNet surveillance agent.

FluNet is WHO's global influenza virological surveillance database.
Provides weekly strain distribution data, subtype proportions, and
hemispheric trends.  GISAID requires individual registration and SFTP
access; we use FluNet as the publicly accessible substitute.

API: https://apps.who.int/flumart/Default?ReportNo=12
     (CSV download endpoint, no auth required)

Also uses WHO's FluID JSON feed for processed summaries.
"""

from __future__ import annotations

import asyncio
import csv
import io
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List

import requests

from .base import AgentConfig, BaseAgent, FetchResult


class FluNetAgent(BaseAgent):
    """
    Fetch WHO FluNet strain surveillance data.
    Returns aggregated strain records useful for understanding
    current circulating subtypes and vaccine match context.
    """

    # FluNet CSV export
    FLUNET_URL = "https://apps.who.int/flumart/Default"

    # WHO FluID API (JSON summary)
    FLUID_URL  = "https://flunewseurope.org/api/v1/surveillance"   # Europe fallback
    WHO_API    = "https://www.who.int/tools/flunet/data-download"

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "BioVoice/1.0"})
        self._delay    = float(config.extra_params.get("delay", 1.0))
        self._weeks    = int(config.extra_params.get("weeks_back", 104))   # 2 years

    def get_capabilities(self) -> List[str]:
        return ["surveillance", "epidemiology", "strain", "flu"]

    def get_default_prompt(self) -> str:
        return (
            "You are an epidemiologist analysing influenza surveillance data. "
            "Based on these WHO FluNet records about {topic}, describe the "
            "current circulating strains, subtype distribution, and their "
            "relevance to broadly neutralising antibody coverage."
        )

    async def fetch(self, query: str, limit: int = 500, **kwargs) -> FetchResult:
        items = await asyncio.to_thread(self._fetch_flunet, query, limit)
        context = self._build_context(items)
        return FetchResult(
            source=self.name,
            items=items,
            metadata={"total": len(items), "query": query},
            prompt_context=context,
        )

    def _fetch_flunet(self, query: str, limit: int) -> List[Dict]:
        end   = date.today()
        start = end - timedelta(weeks=self._weeks)

        params = {
            "ReportNo": 12,
            "WHORegion": "AllWHORegions",
            "Year_from": start.year,
            "Week_from": start.isocalendar()[1],
            "Year_to":   end.year,
            "Week_to":   end.isocalendar()[1],
        }

        try:
            time.sleep(self._delay)
            resp = self._session.get(
                self.FLUNET_URL, params=params, timeout=30
            )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")

            if "csv" in content_type or resp.text.startswith("Country"):
                return self._parse_csv(resp.text, limit)
            else:
                # Response is HTML — fall back to summary stats
                print("[FluNet] CSV export returned HTML, using summary fallback")
                return self._fallback_summary(query, limit)

        except Exception as e:
            print(f"[FluNet] fetch failed: {e} — using summary fallback")
            return self._fallback_summary(query, limit)

    def _parse_csv(self, text: str, limit: int) -> List[Dict]:
        items: List[Dict] = []
        try:
            reader = csv.DictReader(io.StringIO(text))
            subtypes = defaultdict(int)
            records  = []
            for row in reader:
                records.append(row)
                for col in ["AH1", "AH1N12009", "AH3", "AH5", "B", "BVIC", "BYAM"]:
                    val = row.get(col, "").strip()
                    if val and val.isdigit():
                        subtypes[col] += int(val)

            # Aggregate into weekly summary items
            for row in records[:limit]:
                items.append({
                    "country":        row.get("Country", ""),
                    "year":           row.get("Year", ""),
                    "week":           row.get("Week", ""),
                    "ah1":            row.get("AH1", "0"),
                    "ah1n1_2009":     row.get("AH1N12009", "0"),
                    "ah3":            row.get("AH3", "0"),
                    "ah5":            row.get("AH5", "0"),
                    "influenza_b":    row.get("B", "0"),
                    "b_victoria":     row.get("BVIC", "0"),
                    "b_yamagata":     row.get("BYAM", "0"),
                    "source":         "flunet",
                    "title":          f"FluNet W{row.get('Week','')}/{row.get('Year','')} {row.get('Country','')}",
                    "abstract":       self._row_abstract(row, subtypes),
                    "citation_count": 0,
                    "pmid":           "",
                })
        except Exception as e:
            print(f"[FluNet] CSV parse failed: {e}")
        return items

    def _fallback_summary(self, query: str, limit: int) -> List[Dict]:
        """Return a static summary item when the live API is unavailable."""
        return [{
            "source":         "flunet",
            "title":          "WHO FluNet Global Surveillance Summary",
            "abstract": (
                "WHO FluNet tracks global influenza virological surveillance. "
                "In recent seasons, H3N2 and H1N1pdm09 have been co-circulating "
                "in the Northern Hemisphere, with B/Victoria predominating in some "
                "regions. H5N1 sporadic zoonotic cases continue to be reported. "
                "Vaccine effectiveness varies by season and subtype, motivating "
                "broadly neutralising antibody approaches that cover multiple subtypes."
            ),
            "year":           str(date.today().year),
            "citation_count": 0,
            "pmid":           "",
        }]

    def _row_abstract(self, row: Dict, subtypes: Dict) -> str:
        country = row.get("Country", "")
        year    = row.get("Year", "")
        week    = row.get("Week", "")
        dominant = max(subtypes, key=lambda k: subtypes[k]) if subtypes else "unknown"
        return (
            f"WHO FluNet report: {country}, Week {week}/{year}. "
            f"Influenza A H1N1pdm09: {row.get('AH1N12009','0')}, "
            f"H3N2: {row.get('AH3','0')}, H5: {row.get('AH5','0')}. "
            f"Influenza B Victoria: {row.get('BVIC','0')}, "
            f"Yamagata: {row.get('BYAM','0')}. "
            f"Dominant subtype in window: {dominant}."
        )

    def _build_context(self, items: List[Dict]) -> str:
        if not items:
            return ""
        # Aggregate totals
        totals: Dict[str, int] = defaultdict(int)
        countries = set()
        years     = set()
        for it in items:
            for col in ["ah1n1_2009", "ah3", "ah5", "influenza_b", "b_victoria", "b_yamagata"]:
                try:
                    totals[col] += int(it.get(col) or 0)
                except ValueError:
                    pass
            countries.add(it.get("country", ""))
            years.add(it.get("year", ""))

        dominant = max(totals, key=lambda k: totals[k]) if totals else "N/A"
        return (
            f"WHO FluNet surveillance summary ({', '.join(sorted(years))}):\n"
            f"  Countries: {len(countries)}\n"
            f"  H1N1pdm09: {totals['ah1n1_2009']:,}\n"
            f"  H3N2: {totals['ah3']:,}\n"
            f"  H5: {totals['ah5']:,}\n"
            f"  Influenza B: {totals['influenza_b']:,}\n"
            f"  Dominant subtype: {dominant}\n"
            f"  Total weekly reports: {len(items)}"
        )
