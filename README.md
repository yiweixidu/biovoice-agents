# BioVoice — Virology Skill for FluBroad

> **BioVoice** is the official virology Skill package for the [FluBroad Agent Framework](https://github.com/yiweixidu/flubroad).
> It brings influenza bnAb research, universal vaccine literature, and structural immunology
> into the FluBroad pipeline with zero framework code.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FluBroad Skill](https://img.shields.io/badge/flubroad--skill-virology-teal)](https://github.com/yiweixidu/flubroad)

Multi-agent biomedical research assistant. Give it a research question; it fetches from 17 databases, synthesizes a literature review, and produces a Word document, presentation slides, and a narrated video — all in one command.

```bash
biovoice run "broadly neutralizing antibodies influenza hemagglutinin"
```

Output in `./output/`:
- `review.docx` — literature review with verified citations
- `biovoice_output.pptx` — presentation slides
- `biovoice_presentation.mp4` — narrated video walkthrough

**Fully local, zero API cost:**
```bash
biovoice run --llm ollama/llama3.1:8b "your research question"
```

---

## What it does

1. **Fetches in parallel** from 15 databases: PubMed, Europe PMC, UniProt, AlphaFold, Semantic Scholar, bioRxiv/medRxiv, IEDB, Open Targets, Crossref, WHO FluNet, PubChem, PDB, ChEMBL, ClinicalTrials.gov, local files.
2. **Ranks** results by recency, citation count, and domain relevance (Jaccard).
3. **Builds a RAG index** (Chroma + bge-small embeddings, runs on CPU).
4. **Synthesizes** a 6-section review via LLM. Every claim gets a PMID citation. Suspicious citations are flagged, never silently suppressed.
5. **Renders outputs**: Word doc, PPT slides, narrated MP4 video.

For grant-writing specifically, use `biovoice grant`:
```bash
biovoice grant "broadly neutralizing antibodies influenza hemagglutinin"
```

Produces NIH-formatted Specific Aims + Research Strategy with Vancouver citations.

---

## Quickstart

### Requirements
- Python 3.10+
- FFmpeg (for video): `sudo apt install ffmpeg` / `brew install ffmpeg`
- Ollama (for local LLM, optional): https://ollama.ai

### Install

```bash
pip install git+https://github.com/YOUR_USERNAME/biovoice-agents.git
```

Or for development:
```bash
git clone https://github.com/YOUR_USERNAME/biovoice-agents.git
cd biovoice-agents
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Edit .env — add OPENAI_API_KEY, or set LLM_TYPE=ollama for local
```

Or use the interactive setup:
```bash
biovoice config
```

### Run

```bash
# OpenAI (default)
biovoice run "broadly neutralizing antibodies influenza hemagglutinin"

# Fully local, zero cost
biovoice run --llm ollama/llama3.1:8b "broadly neutralizing antibodies influenza"

# Grant mode (NIH format)
biovoice grant "broadly neutralizing antibodies influenza hemagglutinin"

# Show registered agents
biovoice list-agents
```

---

## Agents

| Agent | Source | Data type |
|-------|--------|-----------|
| `pubmed` | NCBI PubMed | Peer-reviewed literature |
| `europe_pmc` | Europe PMC | Literature + preprints, full-text |
| `semantic_scholar` | Semantic Scholar | Citation graph, influential papers |
| `biorxiv` | bioRxiv / medRxiv | Preprints (2 years) |
| `crossref` | Crossref | DOI-registered works, citation counts |
| `iedb` | IEDB | Experimentally validated epitopes, B/T-cell assays |
| `opentargets` | Open Targets | Target-disease-drug associations |
| `uniprot` | UniProt | Protein sequences, function, binding sites |
| `alphafold` | AlphaFold / UniProt | Structure predictions, pLDDT scores |
| `pdb` | RCSB PDB | Experimental 3D structures |
| `pubchem` | PubChem | Small molecules, Lipinski properties, bioassays |
| `flunet` | WHO FluNet | Influenza strain surveillance (weekly) |
| `chembl` | ChEMBL | Bioactive compounds |
| `clinicaltrials` | ClinicalTrials.gov | Clinical trial status and results |
| `local_data` | Local files | PDFs, JSON datasets |

Run a subset:
```bash
biovoice run --agents pubmed,europe_pmc,semantic_scholar "your query"
```

---

## Output formats

```bash
biovoice run --output review,word,ppt,video "your query"
```

| Format | Description |
|--------|-------------|
| `review` | Plain text |
| `word` | `.docx`, NIH-compatible |
| `ppt` | `.pptx`, widescreen, programmatic design |
| `video` | Narrated MP4 (edge-tts + moviepy) |

---

## LLM options

| Flag | Provider | Cost | Notes |
|------|----------|------|-------|
| `--llm openai/gpt-4o-mini` | OpenAI | ~$0.05/run | Default |
| `--llm openai/gpt-4o` | OpenAI | ~$0.50/run | Higher quality |
| `--llm ollama/llama3.1:8b` | Local | Free | Requires Ollama |
| `--llm ollama/llama3.2:3b` | Local | Free | Faster |

---

## Architecture

```
Input query
    │
    ├── [parallel asyncio.gather — 15 agents]
    │     ├── pubmed, europe_pmc, semantic_scholar, biorxiv
    │     ├── iedb, opentargets, crossref, flunet, pubchem
    │     └── uniprot, alphafold, pdb, chembl, clinicaltrials, local_data
    │
    ├── Merge + deduplicate by PMID/DOI
    │   Rank: 0.5×recency + 0.3×citations + 0.2×domain_Jaccard
    │
    ├── RAG index (Chroma + BAAI/bge-small-en-v1.5, CPU)
    │
    ├── Section synthesis (LLM, 6 sections)
    │   Every claim requires [PMID] — suspicious citations flagged
    │
    └── Output pipeline
          ├── review.docx  (python-docx)
          ├── slides.pptx  (python-pptx, programmatic design)
          └── video.mp4    (edge-tts + moviepy)
```

```
biovoice/
├── agents/          # 15 data source agents
├── cli/main.py      # Click CLI
├── config/          # Pydantic settings
├── core/            # Orchestrator, task state machine
├── models/          # LLM client (OpenAI + Ollama)
├── output/          # Word, PPT, video renderers
└── ui/              # Gradio web UI (biovoice serve)
core/
├── rag/             # Chroma vector store + retrieval
└── presentation/    # PPT generator primitives
domain/virology/     # Antibody schemas + prompt templates
```

---

## Grant writing mode

Skips RAG; synthesizes directly from ranked abstracts (faster, ~$0.05/run).

```bash
biovoice grant "broadly neutralizing antibodies influenza hemagglutinin" \
    --max-papers 30 \
    --output-dir ./grant_output
```

Output:
- `grant_specific_aims.docx` — Specific Aims, NIH format, Vancouver citations
- `grant_slides.pptx` — 6-slide supporting deck

Every claim verified against source abstract. Suspicious citations surfaced to user.

---

## Web UI

```bash
biovoice serve  # Gradio at http://localhost:7860
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/               # unit tests (no API calls)
pytest -m integration       # requires NCBI API access
```

---

## FluBroad Skill ecosystem

BioVoice is a **Skill package** in the [FluBroad Agent Framework](https://github.com/yiweixidu/flubroad). The framework provides the infrastructure (RAG, orchestration, PPT, knowledge graph, Q&A, fine-tuning). BioVoice provides the domain expertise:

```python
# biovoice/skill.py — the SkillManifest
from biovoice.skill import manifest

print(manifest.summary())
# Skill(virology v1.0.0) | 17 agents | 6 sections

print(manifest.default_agents)
# ['pubmed', 'europe_pmc', 'semantic_scholar', 'iedb', 'uniprot', 'pdb', 'flunet', 'maad']
```

Once the `flubroad` framework package is published, usage becomes:

```python
from flubroad.skill import SkillLoader
from flubroad.core.orchestrator import FluBroadOrchestrator

skill = SkillLoader.load("virology")   # finds this package via entry points
orch  = FluBroadOrchestrator(config, skill=skill)
```

The Skill manifest (`biovoice/skill.py`) contains all domain knowledge — section queries, synthesis instructions, antibody extraction schema, knowledge graph patterns, grant templates — in one place. No virology logic lives in the framework.

Other planned Skills: `flubroad-skill-oncology`, `flubroad-skill-immunology`. See the [FluBroad Skill Specification](https://github.com/yiweixidu/flubroad/blob/main/docs/skill-spec.md) to build your own.

---

## License

**CC BY-NC 4.0** — free for academic and non-commercial use. Commercial use requires a license.
See [LICENSE](LICENSE) for details.

The FluBroad core framework ([github.com/yiweixidu/flubroad](https://github.com/yiweixidu/flubroad)) is MIT licensed.
