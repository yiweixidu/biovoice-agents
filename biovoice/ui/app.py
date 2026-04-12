"""
biovoice/ui/app.py
Gradio Web UI for BioVoice-Agents.

Layout:
  Left  panel  — agent selection, model selection, prompt editor, API key mgmt
  Centre panel — query + config + run controls + progress log
  Right  panel — tabbed output: review, antibody table, PPT download, video
"""

from __future__ import annotations

import asyncio
import json
import os

import gradio as gr
from dotenv import load_dotenv

from biovoice.agents.registry import AgentRegistry
from biovoice.config.settings import BioVoiceSettings
from biovoice.core.orchestrator import BioVoiceOrchestrator

load_dotenv()
AgentRegistry.load_plugins()

settings  = BioVoiceSettings()
_orch: BioVoiceOrchestrator | None = None


def _get_orch() -> BioVoiceOrchestrator:
    global _orch
    if _orch is None:
        _orch = BioVoiceOrchestrator(settings.to_orchestrator_config())
    return _orch


# ── Run pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    query:           str,
    agent_selection: list,
    llm_model:       str,
    max_papers:      int,
    do_ppt:          bool,
    do_video:        bool,
    custom_prompt:   str,
):
    if not query.strip():
        yield "Please enter a query.", "<p>No output.</p>", None, None, ""
        return

    log_lines: list = []

    def log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines)

    yield log("Starting pipeline..."), "<p>Working...</p>", None, None, ""

    # Rebuild orchestrator if model changed
    cfg = settings.to_orchestrator_config()
    cfg["llm_model"]            = llm_model
    cfg["max_papers_per_agent"] = int(max_papers)
    orch = BioVoiceOrchestrator(cfg)

    output_types = ["review"]
    if do_ppt:
        output_types.append("ppt")
    if do_video:
        output_types.append("video")

    def progress(stage, current, total):
        log_lines.append(f"  [{current}/{total}] {stage}")

    try:
        result = asyncio.run(
            orch.run(
                query=query,
                agent_names=agent_selection or ["pubmed"],
                output_types=output_types,
                progress_cb=progress,
            )
        )
    except Exception as exc:
        yield log(f"\nERROR: {exc}"), "<p>Failed.</p>", None, None, ""
        return

    review     = result.get("review", "")
    antibodies = result.get("antibodies", [])
    ppt_file   = result.get("ppt_file")
    video_file = result.get("video_file")

    if antibodies:
        import pandas as pd
        table_html = pd.DataFrame(antibodies).to_html(
            classes="table table-striped", index=False
        )
    else:
        table_html = "<p>No antibodies extracted.</p>"

    task    = result.get("task", {})
    summary = (
        f"Status: {task.get('status','done')} | "
        f"Agents: {', '.join(agent_selection)} | "
        f"Antibodies found: {len(antibodies)}"
    )
    yield log(f"\nDone!\n{summary}"), table_html, ppt_file, video_file, review


# ── RAG Q&A ───────────────────────────────────────────────────────────────────

def answer_question(question: str, history):
    orch = _get_orch()
    docs = orch.rag.similarity_search(question, k=4)
    if not docs:
        return "No relevant documents found. Run a pipeline first."
    context = "\n\n".join(d.page_content for d in docs)
    system  = "You are a helpful biomedical research assistant."
    human   = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    answer  = orch.model.chat(system, human)
    refs    = "\n\n**References:**\n" + "\n".join(
        f"- {d.metadata.get('title','N/A')} (PMID: {d.metadata.get('pmid','N/A')})"
        for d in docs
    )
    return answer + refs


# ── Gradio layout ─────────────────────────────────────────────────────────────

ALL_AGENTS = AgentRegistry.list_agents() or [
    "pubmed", "europe_pmc", "pdb", "uniprot",
    "clinicaltrials", "chembl", "local_data",
]

DEFAULT_PROMPT = (
    "You are a senior biomedical researcher. "
    "Write a critical, structured literature review with PMIDs for every claim."
)

with gr.Blocks(title="BioVoice-Agents") as demo:
    gr.Markdown("# BioVoice-Agents")
    gr.Markdown(
        "Multi-agent biomedical research assistant · "
        f"Model: **{settings.llm_model}**"
    )

    with gr.Tabs():

        # ── Tab 1: Run pipeline ───────────────────────────────────────────────
        with gr.TabItem("Run pipeline"):
            with gr.Row():
                # Left panel
                with gr.Column(scale=1):
                    gr.Markdown("### Agents")
                    agent_sel = gr.CheckboxGroup(
                        choices=ALL_AGENTS,
                        value=["pubmed"],
                        label="Data source agents",
                    )
                    gr.Markdown("### Model")
                    model_sel = gr.Dropdown(
                        choices=[
                            "gpt-4o-mini", "gpt-4o",
                            "ollama/llama3.2:3b", "ollama/llama3.1:8b",
                        ],
                        value=settings.llm_model,
                        label="LLM",
                    )
                    max_papers = gr.Slider(
                        label="Max papers per agent",
                        minimum=10, maximum=500, value=100, step=10,
                    )
                    gr.Markdown("### Outputs")
                    do_ppt   = gr.Checkbox(label="Generate PPT",   value=True)
                    do_video = gr.Checkbox(label="Generate video",  value=False)
                    gr.Markdown("### Prompt override")
                    custom_prompt = gr.Textbox(
                        label="System prompt (optional)",
                        value=DEFAULT_PROMPT,
                        lines=4,
                    )

                # Centre + right panels
                with gr.Column(scale=3):
                    query_box = gr.Textbox(
                        label="Query",
                        value="broadly neutralizing antibodies against influenza HA stem",
                        lines=2,
                    )
                    run_btn = gr.Button("Run", variant="primary")
                    log_box = gr.Textbox(
                        label="Progress log", lines=10, interactive=False
                    )

                    with gr.Tabs():
                        with gr.TabItem("Review"):
                            review_box = gr.Textbox(
                                label="Generated review",
                                lines=25,
                                interactive=False,
                            )
                        with gr.TabItem("Antibody table"):
                            ab_html = gr.HTML(label="Extracted antibodies")
                        with gr.TabItem("Downloads"):
                            ppt_dl   = gr.File(label="PPT",   interactive=False)
                            video_dl = gr.File(label="Video", interactive=False)

            run_btn.click(
                fn=run_pipeline,
                inputs=[
                    query_box, agent_sel, model_sel,
                    max_papers, do_ppt, do_video, custom_prompt,
                ],
                outputs=[log_box, ab_html, ppt_dl, video_dl, review_box],
            )

        # ── Tab 2: RAG Q&A ────────────────────────────────────────────────────
        with gr.TabItem("RAG Q&A"):
            gr.Markdown("Ask questions about the indexed literature.")
            gr.ChatInterface(
                fn=answer_question,
                title="BioVoice Q&A",
            )

        # ── Tab 3: Agent info ─────────────────────────────────────────────────
        with gr.TabItem("Agent info"):
            gr.Markdown("### Registered agents")
            agent_rows = []
            for name in ALL_AGENTS:
                try:
                    cls  = AgentRegistry.available().get(name)
                    caps = ", ".join(cls(
                        __import__(
                            "biovoice.agents.base",
                            fromlist=["AgentConfig"]
                        ).AgentConfig(name=name)
                    ).get_capabilities()) if cls else "N/A"
                except Exception:
                    caps = "N/A"
                agent_rows.append([name, caps])
            gr.Dataframe(
                value=agent_rows,
                headers=["Agent", "Capabilities"],
                interactive=False,
            )


def launch_ui(port: int = 7860):
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    launch_ui()