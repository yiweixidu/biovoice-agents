"""
biovoice/qa/engine.py
Multi-turn Q&A engine with source attribution.

Answers biomedical questions about a loaded corpus using RAG retrieval +
an LLM, keeping conversation history so follow-up questions work naturally.

Design
------
- QAEngine wraps any BioVoice ModelClient (OpenAI, Ollama, ...).
- Each answer cites the retrieved chunks as numbered references with PMID / DOI.
- Conversation history is kept as a plain list of (role, text) pairs — no
  LangChain Memory objects — so the engine stays framework-agnostic.
- Thread-safe: each request gets a fresh context window built from history +
  retrieved docs. No global mutable state between concurrent requests.

Usage
-----
    from core.rag.vector_store import FluBroadRAG
    from biovoice.qa.engine import QAEngine
    from biovoice.models.base import build_model_client

    rag = FluBroadRAG(...)
    rag.load()
    model = build_model_client(config)
    qa = QAEngine(rag=rag, model=model, k=5)

    answer = qa.ask("What epitope does CR6261 target?")
    print(answer.text)
    print(answer.references)

    # Follow-up
    answer2 = qa.ask("How does this compare to FI6?")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from biovoice.models.base import ModelClient

SYSTEM_PROMPT = """\
You are BioVoice, an expert biomedical research assistant specialising in \
virology, broadly neutralising antibodies, vaccine design, and related fields.

You answer questions strictly from the provided literature context.
Rules:
1. Cite every factual claim with [N] (corresponding to the references below the context).
2. If the context does not contain enough information, say so explicitly rather \
   than guessing.
3. Maintain awareness of prior conversation turns — if the user refers to \
   "it", "they", "that antibody", etc., resolve the reference from the history.
4. Be concise but complete. Use markdown for clarity when appropriate.
5. Do NOT fabricate PMIDs, titles, or author names.
"""

_HISTORY_WINDOW = 6   # number of prior turns to include in context (3 exchanges)


@dataclass
class QAResult:
    """The result of a single Q&A turn."""
    text:       str                    # answer text with [N] citations
    references: List[Dict]             # [{n, title, pmid, doi, source}]
    query:      str                    # the original question
    history_len: int = 0              # how many prior turns were used


@dataclass
class QAEngine:
    """
    Multi-turn Q&A engine over a RAG corpus.

    Parameters
    ----------
    rag     : any object with .similarity_search(query, k) → List[Document]
    model   : ModelClient with .chat(system, human) → str
    k       : number of chunks to retrieve per turn
    """
    rag:    object
    model:  ModelClient
    k:      int = 5

    # Conversation history as (role, text) pairs.
    # Populated by ask(); clear with reset().
    _history: List[Tuple[str, str]] = field(default_factory=list, init=False)

    def ask(self, query: str) -> QAResult:
        """
        Answer a question, using conversation history for context.

        Parameters
        ----------
        query : the user's question (natural language)

        Returns
        -------
        QAResult with answer text, numbered references, and metadata
        """
        # 1. Retrieve relevant chunks
        docs = self.rag.similarity_search(query, k=self.k)

        # 2. Build numbered context block
        context_lines = []
        refs: List[Dict] = []
        for i, doc in enumerate(docs, start=1):
            meta   = doc.metadata if hasattr(doc, "metadata") else {}
            pmid   = meta.get("pmid") or meta.get("PMID") or ""
            doi    = meta.get("doi") or ""
            title  = meta.get("title") or doc.page_content[:80]
            source = meta.get("source") or ""
            refs.append({"n": i, "title": title, "pmid": pmid, "doi": doi, "source": source})
            context_lines.append(
                f"[{i}] (PMID:{pmid or 'N/A'}) {title}\n{doc.page_content[:800]}"
            )
        context_block = "\n\n".join(context_lines)

        # 3. Build conversation history section (last _HISTORY_WINDOW turns)
        recent = self._history[-_HISTORY_WINDOW:]
        history_block = ""
        if recent:
            history_lines = []
            for role, text in recent:
                prefix = "User" if role == "user" else "Assistant"
                history_lines.append(f"{prefix}: {text[:600]}")
            history_block = "CONVERSATION HISTORY:\n" + "\n\n".join(history_lines) + "\n\n"

        # 4. Compose the human prompt
        human = (
            f"{history_block}"
            f"LITERATURE CONTEXT:\n{context_block}\n\n"
            f"User question: {query}\n\n"
            f"Answer (use [N] citations, be concise):"
        )

        # 5. Generate
        raw_answer = self.model.chat(SYSTEM_PROMPT, human)

        # 6. Append to history
        self._history.append(("user",      query))
        self._history.append(("assistant", raw_answer))

        return QAResult(
            text=raw_answer,
            references=refs,
            query=query,
            history_len=len(recent) // 2,
        )

    def format_response(self, result: QAResult) -> str:
        """
        Format a QAResult into a markdown string suitable for display.
        Appends a numbered reference list below the answer.
        """
        lines = [result.text.strip(), ""]
        if result.references:
            lines.append("---")
            lines.append("**References**")
            for ref in result.references:
                pmid_part = f" · PMID: {ref['pmid']}" if ref["pmid"] else ""
                doi_part  = f" · DOI: {ref['doi']}"  if ref["doi"]  else ""
                lines.append(
                    f"[{ref['n']}] *{ref['title'][:100]}*{pmid_part}{doi_part}"
                )
        return "\n".join(lines)

    def reset(self):
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> List[Tuple[str, str]]:
        """Read-only view of current conversation history."""
        return list(self._history)

    def history_as_gradio_chatbot(self) -> List[List[Optional[str]]]:
        """
        Convert history to the [[user, bot], ...] format expected by
        Gradio's gr.Chatbot component.
        """
        turns: List[List[Optional[str]]] = []
        i = 0
        while i < len(self._history) - 1:
            role_a, text_a = self._history[i]
            role_b, text_b = self._history[i + 1]
            if role_a == "user" and role_b == "assistant":
                turns.append([text_a, text_b])
                i += 2
            else:
                i += 1
        return turns
