"""
scripts/show_corpus.py
Fetch and display the ranked paper corpus for a query without running synthesis.
Usage: python3 scripts/show_corpus.py "broadly neutralizing antibodies influenza hemagglutinin"
"""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from biovoice.config.settings import BioVoiceSettings
from biovoice.core.grant_config import GrantConfig
from biovoice.core.orchestrator import BioVoiceOrchestrator, _token_set, _jaccard
from domain.virology.schemas.antibody_schema import antibody_schema
from datetime import date

query = sys.argv[1] if len(sys.argv) > 1 else "broadly neutralizing antibodies influenza hemagglutinin"

settings = BioVoiceSettings()
config   = settings.to_orchestrator_config()
gc       = GrantConfig(research_question=query, max_ranked_papers=30)
orch     = BioVoiceOrchestrator(config, use_rag=False)


async def go():
    frs = await orch._fetch_all(gc.research_question, ['pubmed', 'europe_pmc', 'uniprot'], None)

    seen, merged = set(), []
    for fr in frs:
        for item in fr.items:
            pmid = str(item.get('pmid') or item.get('PMID') or '')
            key  = pmid or item.get('doi') or item.get('title') or str(id(item))
            if key not in seen:
                seen.add(key)
                item = dict(item)
                item.setdefault('_agent_source', fr.source)
                merged.append(item)

    cy = date.today().year
    mc = max((int(p.get('citation_count') or 0) for p in merged), default=1) or 1
    dv = _token_set(query + ' ' + ' '.join(antibody_schema.keys()), set())

    def sc(p):
        yr = int(p.get('year') or p.get('pub_year') or cy)
        return (
            0.5 * max(0, 1 - (cy - yr) / 20)
            + 0.3 * int(p.get('citation_count') or 0) / mc
            + 0.2 * _jaccard(
                _token_set((p.get('title') or '') + ' ' + (p.get('abstract') or ''), set()), dv
            )
        )

    # Cap UniProt protein-DB entries at 3; rest must be literature with PMIDs.
    MAX_UNIPROT = 3
    all_sorted = sorted(merged, key=sc, reverse=True)
    ranked, uniprot_count = [], 0
    for p in all_sorted:
        if len(ranked) >= gc.max_ranked_papers:
            break
        if p.get('_agent_source') == 'uniprot':
            if uniprot_count >= MAX_UNIPROT:
                continue
            uniprot_count += 1
        ranked.append(p)

    src_counts = {}
    for fr in frs:
        src_counts[fr.source] = fr.count

    print(f"\nQuery: {query}")
    print(f"Sources fetched: " + ", ".join(f"{s}={n}" for s, n in src_counts.items()))
    print(f"Total unique after dedup: {len(merged)}")
    print(f"Sent to synthesis: top {len(ranked)}")
    print()

    print(f"{'#':>2}  {'PMID':<12} {'Year':>4}  {'Cites':>6}  {'Score':>5}  {'FT':<2}  {'Source':<11}  Title")
    print('─' * 115)
    for i, p in enumerate(ranked, 1):
        pmid  = str(p.get('pmid') or p.get('PMID') or p.get('accession') or '—')[:11]
        title = (p.get('title') or p.get('name') or '—')[:65]
        yr    = str(p.get('year') or p.get('pub_year') or '?')
        cites = str(p.get('citation_count') or 0)
        src   = str(p.get('_agent_source', '?'))[:11]
        ft    = 'Y' if p.get('fulltext_available') else '·'
        s     = sc(p)
        print(f"{i:2}.  {pmid:<12} {yr:>4}  {cites:>6}  {s:.3f}  {ft:<2}  {src:<11}  {title}")

    print()
    print('─' * 115)
    print("ABSTRACTS")
    print('─' * 115)

    for i, p in enumerate(ranked, 1):
        pmid  = str(p.get('pmid') or p.get('PMID') or p.get('accession') or '—')
        title = p.get('title') or p.get('name') or '—'
        yr    = str(p.get('year') or p.get('pub_year') or '?')
        doi   = p.get('doi', '')
        abst  = p.get('abstract') or p.get('function') or '(no abstract)'
        src   = p.get('_agent_source', '?')

        au = p.get('authors') or []
        if isinstance(au, list) and au:
            first = au[0]
            name  = first.get('last_name') or str(first) if isinstance(first, dict) else str(first)
            auth  = name[:25] + (' et al.' if len(au) > 1 else '')
        elif isinstance(au, str):
            auth = au.split(';')[0][:30]
        else:
            auth = ''

        print(f"\n[{i}] PMID:{pmid}  {yr}  [{src}]  {auth}")
        print(f"    {title}")
        if doi:
            print(f"    DOI: {doi}")
        print(f"    {abst[:400]}")

asyncio.run(go())
