"""
biovoice/cli/main.py   — Click CLI  (`biovoice` command)
biovoice/bots/gateway.py — FastAPI webhook gateway for Feishu / DingTalk
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json

import click
from dotenv import load_dotenv
load_dotenv()  # load .env before settings are instantiated

from biovoice.agents.registry import AgentRegistry
from biovoice.config.settings import BioVoiceSettings
from biovoice.core.orchestrator import BioVoiceOrchestrator


@click.group()
def cli():
    """BioVoice-Agents: multi-agent biomedical research assistant."""


@cli.command("list-agents")
def list_agents():
    """List all registered data source agents."""
    AgentRegistry.load_plugins()
    agents = AgentRegistry.list_agents()
    if not agents:
        click.echo("No agents registered.")
        return
    click.echo("Available agents:")
    for name in agents:
        cls = AgentRegistry.available()[name]
        caps = cls.__init__.__doc__ or ""
        click.echo(f"  {name:20s}  {cls.__name__}")


@cli.command("list-models")
def list_models():
    """List supported model types."""
    rows = [
        ("openai/gpt-4o-mini",            "LLM",       "Recommended default"),
        ("openai/gpt-4o",                 "LLM",       "Higher quality"),
        ("ollama/llama3.2:3b",            "LLM",       "Local, fast, lower quality"),
        ("ollama/llama3.1:8b",            "LLM",       "Local, better quality"),
        ("openai/text-embedding-3-small", "Embedding", "OpenAI cloud"),
        ("huggingface/BAAI/bge-large",    "Embedding", "Local HuggingFace"),
        ("edge-tts",                      "TTS",       "Free, good quality"),
    ]
    click.echo(f"{'Model':<35} {'Type':<12} {'Notes'}")
    click.echo("-" * 70)
    for model, mtype, note in rows:
        click.echo(f"{model:<35} {mtype:<12} {note}")


@cli.command("run")
@click.argument("query_arg", required=False, default=None, metavar="QUERY")
@click.option("--query", "-q", default=None, help="Search query (alternative to positional arg)")
@click.option(
    "--agents", "-a",
    default=(
        "pubmed,europe_pmc,uniprot,alphafold,"
        "semantic_scholar,biorxiv,iedb,opentargets,"
        "crossref,flunet,pubchem"
    ),
    show_default=True,
    help="Comma-separated agent names",
)
@click.option(
    "--output", "-o",
    default="review,word,ppt,video",
    show_default=True,
    help="Comma-separated output types: review,word,ppt,video",
)
@click.option(
    "--max-papers", default=0, show_default=True,
    help="Max papers per agent (0 = no limit)",
)
@click.option("--topic", default="flu_bnabs", show_default=True)
@click.option(
    "--llm", default=None,
    help=(
        "LLM backend in provider/model format.  Examples:\n"
        "  openai/gpt-4o-mini  (default, needs OPENAI_API_KEY)\n"
        "  openai/gpt-4o\n"
        "  ollama/llama3.1:8b  (local, zero cost)\n"
        "  ollama/llama3.2:3b  (fast local)"
    ),
    metavar="PROVIDER/MODEL",
)
def run(query_arg: str, query: str, agents: str, output: str, max_papers: int, topic: str, llm: str):
    query = query or query_arg
    if not query:
        raise click.UsageError("Provide a query: biovoice run \"your query\" or --query \"your query\"")
    """Run the full BioVoice pipeline for a query."""
    settings    = BioVoiceSettings()
    config      = settings.to_orchestrator_config()
    if llm:
        parts = llm.split("/", 1)
        if len(parts) != 2:
            raise click.UsageError("--llm must be in PROVIDER/MODEL format, e.g. ollama/llama3.1:8b")
        config["llm_type"]  = parts[0]
        config["llm_model"] = parts[1]
    if max_papers:
        config["max_papers_per_agent"] = max_papers
    else:
        config["max_papers_per_agent"] = 9999   # no meaningful cap
    orch        = BioVoiceOrchestrator(config)
    agent_list  = [a.strip() for a in agents.split(",")]
    output_list = [o.strip() for o in output.split(",")]

    llm_display = llm or f"{config['llm_type']}/{config['llm_model']}"
    click.echo(f"Query   : {query}")
    click.echo(f"LLM     : {llm_display}")
    click.echo(f"Agents  : {agent_list}")
    click.echo(f"Outputs : {output_list}")
    click.echo(f"Limit   : {'unlimited' if not max_papers else max_papers} per agent")
    click.echo("")

    def progress(stage, current, total):
        click.echo(f"  [{current}/{total}] {stage}")

    result = asyncio.run(
        orch.run(
            query=query,
            agent_names=agent_list,
            output_types=output_list,
            topic=topic,
            progress_cb=progress,
        )
    )
    click.echo("\n=== Review ===")
    click.echo(result.get("review", "")[:1000] + "...\n")
    if result.get("word_file"):
        click.echo(f"Word doc  : {result['word_file']}")
    if result.get("ppt_file"):
        click.echo(f"PPT saved : {result['ppt_file']}")
    if result.get("video_file"):
        click.echo(f"Video     : {result['video_file']}")
    ab_count = len(result.get("antibodies", []))
    click.echo(f"Antibodies extracted: {ab_count}")


@cli.command("config")
@click.option("--openai-key", prompt="OpenAI API key (sk-...)", hide_input=True)
@click.option("--pubmed-key", default="", prompt="PubMed API key (leave blank to skip)")
@click.option("--email",      prompt="Your email (for PubMed/Unpaywall)")
def config_cmd(openai_key: str, pubmed_key: str, email: str):
    """Interactive setup — writes a .env file."""
    lines = [
        f"OPENAI_API_KEY={openai_key}",
        f"PUBMED_API_KEY={pubmed_key}",
        f"EMAIL={email}",
        "LLM_TYPE=openai",
        "LLM_MODEL=gpt-4o-mini",
        "LLM_TEMPERATURE=0.1",
        "COLLECTION_NAME=biovoice",
        "PERSIST_DIR=./data/vector_db",
        "CACHE_FILE=data/cache/articles.json",
        "OUTPUT_DIR=./output",
    ]
    with open(".env", "w") as f:
        f.write("\n".join(lines) + "\n")
    click.echo(".env written. Run `biovoice run --query '...'` to start.")


@cli.command("serve")
@click.option("--port", default=7860, show_default=True)
def serve(port: int):
    """Launch the Gradio web UI."""
    from biovoice.ui.app import launch_ui
    launch_ui(port=port)


@cli.command("grant")
@click.argument("query_arg", required=False, default=None, metavar="QUERY")
@click.option(
    "--query", "-q",
    default=None,
    help='Virology research question (alternative to positional arg)',
)
@click.option(
    "--output-dir", "-o",
    default="./output",
    show_default=True,
    help="Directory for .docx and .pptx output files.",
)
@click.option(
    "--max-papers",
    default=30,
    show_default=True,
    help="Max papers to synthesize after ranking (abstracts only).",
)
def grant(query_arg: str, query: str, output_dir: str, max_papers: int):
    query = query or query_arg
    if not query:
        raise click.UsageError("Provide a query: biovoice grant \"your question\" or --query \"your question\"")
    """
    Run the Virology Grant Copilot.

    Fetches from PubMed + Europe PMC + UniProt, synthesizes NIH-formatted
    grant sections (Specific Aims, Significance, Innovation, Background),
    and writes a Word doc + PPT slide deck to OUTPUT_DIR.

    Every citation is verified against the source abstract. Suspicious
    citations are flagged in the output — never silently suppressed.

    Example:

      biovoice grant --query "broadly neutralizing antibodies influenza"
    """
    import time
    from biovoice.core.grant_config import GrantConfig

    settings = BioVoiceSettings()
    config   = settings.to_orchestrator_config()

    gc = GrantConfig(
        research_question=query,
        max_ranked_papers=max_papers,
        output_dir=output_dir,
    )

    # Grant mode: skip RAG, use abstracts-only synthesis
    orch = BioVoiceOrchestrator(config, use_rag=False)

    click.echo(f"\nQuery      : {query}")
    click.echo(f"Max papers : {max_papers}")
    click.echo(f"Output dir : {output_dir}")
    click.echo("\nFetching from PubMed, Europe PMC, UniProt ...\n")

    t0 = time.time()
    result = asyncio.run(orch.run_grant(gc))
    elapsed = time.time() - t0

    click.echo(f"\n{'='*60}")
    click.echo(f"Completed in {elapsed:.0f}s")
    click.echo(f"Sections generated : {len(result.sections)}")
    click.echo(f"Citations verified : {len(result.all_citations)}")

    if result.has_warnings:
        click.echo(
            click.style(
                f"\n{result.warning_count} citation warning(s) — review before submission:",
                fg="yellow",
            )
        )
        for w in result.citation_warnings:
            click.echo(f"  {w}")

    click.echo("")
    if result.word_file:
        click.echo(f"Word doc : {result.word_file}")
    if result.ppt_file:
        click.echo(f"PPT      : {result.ppt_file}")

    # Print Specific Aims preview
    sa = result.section("specific_aims")
    if sa:
        click.echo(f"\n--- Specific Aims (preview) ---")
        click.echo(sa.text[:600] + ("..." if len(sa.text) > 600 else ""))


if __name__ == "__main__":
    cli()
