"""
biovoice/finetuning/feedback_store.py
Feedback collection and LoRA training data export.

Workflow
--------
1. During normal usage the BioVoice orchestrator logs every (prompt, response)
   pair to feedback_store.log_inference().

2. A domain expert opens the Gradio "Feedback" tab, sees the generated text,
   and submits a corrected version + quality rating (1-5).

3. feedback_store.submit_correction() saves the correction alongside the
   original.

4. feedback_store.export_jsonl() writes a JSONL file in ShareGPT format:
   {"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
   This is directly compatible with Unsloth, LLaMA-Factory, and Axolotl.

5. Fine-tune:
   # Unsloth (recommended for single GPU):
   python scripts/finetune_lora.py --data data/feedback.jsonl --base ollama/llama3.1:8b

   # Ollama Modelfile (no-code path for Ollama users):
   ollama create biovoice-finetuned -f scripts/Modelfile

Storage
-------
Records are stored as newline-delimited JSON (JSONL) at feedback_dir/feedback.jsonl.
Thread-safe: each write acquires a file lock (portalocker if available, else a
threading.Lock for single-process use).

Design notes
------------
- No database required — flat JSONL is readable, diffable, and git-trackable.
- Keeps the inference log separate from corrections: log_inference() appends to
  inferences.jsonl; submit_correction() appends to corrections.jsonl.
- export_jsonl() merges corrected inference pairs into a deduplicated training
  set, preferring the expert-corrected response over the original.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    """A single feedback instance."""
    id:              str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:       str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Inputs
    query:           str   = ""          # user query / topic
    section:         str   = ""          # e.g. "results", "mechanisms", "grant_sa"
    prompt_context:  str   = ""          # RAG context provided to the LLM (truncated)

    # Outputs
    original_text:   str   = ""          # LLM-generated text
    corrected_text:  str   = ""          # expert-corrected text (empty if uncorrected)

    # Rating
    rating:          int   = 0           # 1-5 (0 = not rated)
    rating_reason:   str   = ""          # free-text explanation

    # Metadata
    model:           str   = ""          # e.g. "gpt-4o-mini", "llama3.1:8b"
    source:          str   = "biovoice"  # which pipeline generated this


# ── Store ─────────────────────────────────────────────────────────────────────

class FeedbackStore:
    """
    Append-only store for inference logs and expert corrections.

    Parameters
    ----------
    feedback_dir : directory for JSONL files (created if absent)
    """

    def __init__(self, feedback_dir: str = "data/feedback"):
        self._dir   = feedback_dir
        self._lock  = threading.Lock()
        os.makedirs(feedback_dir, exist_ok=True)
        self._inf_path  = os.path.join(feedback_dir, "inferences.jsonl")
        self._corr_path = os.path.join(feedback_dir, "corrections.jsonl")

    # ── Write API ─────────────────────────────────────────────────────────────

    def log_inference(
        self,
        query:          str,
        original_text:  str,
        section:        str = "",
        prompt_context: str = "",
        model:          str = "",
    ) -> str:
        """
        Log an LLM-generated synthesis output.

        Returns the record ID (use it to submit a correction later).
        """
        record = FeedbackRecord(
            query=query,
            section=section,
            prompt_context=prompt_context[:2000],   # cap to 2K chars
            original_text=original_text,
            model=model,
        )
        self._append(self._inf_path, asdict(record))
        return record.id

    def submit_correction(
        self,
        record_id:      str,
        corrected_text: str,
        rating:         int = 5,
        rating_reason:  str = "",
    ) -> FeedbackRecord:
        """
        Submit an expert correction for a previously logged inference.

        Parameters
        ----------
        record_id      : ID returned by log_inference()
        corrected_text : the improved version of the synthesis
        rating         : 1-5 quality score for the ORIGINAL text
        rating_reason  : optional free-text explanation
        """
        # Load the original record
        orig = self._find_by_id(self._inf_path, record_id)
        if orig is None:
            # Record not found — create a minimal correction-only entry
            orig = {}

        record = FeedbackRecord(
            id=record_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=orig.get("query", ""),
            section=orig.get("section", ""),
            prompt_context=orig.get("prompt_context", ""),
            original_text=orig.get("original_text", ""),
            corrected_text=corrected_text,
            rating=max(1, min(5, int(rating))),
            rating_reason=rating_reason,
            model=orig.get("model", ""),
        )
        self._append(self._corr_path, asdict(record))
        return record

    # ── Export API ────────────────────────────────────────────────────────────

    def export_jsonl(
        self,
        output_path:         str  = "data/feedback/training_data.jsonl",
        format:              str  = "sharegpt",   # "sharegpt" | "alpaca"
        min_rating:          int  = 3,            # only include corrections rated ≥ this
        include_uncorrected: bool = False,        # include original if no correction exists
        system_prompt:       str  = (
            "You are BioVoice, an expert biomedical research assistant. "
            "Write accurate, well-cited literature synthesis sections."
        ),
    ) -> int:
        """
        Export training data in fine-tuning format.

        Parameters
        ----------
        output_path         : destination JSONL file
        format              : "sharegpt" (Unsloth/LLaMA-Factory) or "alpaca"
        min_rating          : minimum expert rating to include a correction
        include_uncorrected : if True, include original (model) outputs that
                              were NOT corrected as positive examples
        system_prompt       : system message prepended to every conversation

        Returns
        -------
        Number of training examples written.
        """
        # Load corrections (expert-preferred outputs)
        corrections: Dict[str, dict] = {}
        for rec in self._iter_jsonl(self._corr_path):
            if int(rec.get("rating", 0)) >= min_rating and rec.get("corrected_text"):
                corrections[rec["id"]] = rec

        # Load all inferences
        inferences = list(self._iter_jsonl(self._inf_path))

        examples: List[dict] = []
        seen_ids: set = set()

        # Priority 1: corrected pairs
        for rec_id, corr in corrections.items():
            query         = corr.get("query", "")
            context       = corr.get("prompt_context", "")
            corrected     = corr.get("corrected_text", "")
            section       = corr.get("section", "")
            if not query or not corrected:
                continue
            human = self._build_human_prompt(query, context, section)
            examples.append(
                self._format_example(human, corrected, system_prompt, format)
            )
            seen_ids.add(rec_id)

        # Priority 2 (optional): uncorrected high-quality originals
        if include_uncorrected:
            for rec in inferences:
                if rec.get("id") in seen_ids:
                    continue
                original = rec.get("original_text", "")
                query    = rec.get("query", "")
                if not query or not original:
                    continue
                human = self._build_human_prompt(
                    query, rec.get("prompt_context", ""), rec.get("section", "")
                )
                examples.append(
                    self._format_example(human, original, system_prompt, format)
                )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"[FeedbackStore] Exported {len(examples)} training examples → {output_path}")
        return len(examples)

    def stats(self) -> Dict:
        """Return store statistics."""
        inferences  = list(self._iter_jsonl(self._inf_path))
        corrections = list(self._iter_jsonl(self._corr_path))
        ratings = [int(r.get("rating", 0)) for r in corrections if r.get("rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        return {
            "total_inferences":  len(inferences),
            "total_corrections": len(corrections),
            "avg_rating":        round(avg_rating, 2),
            "correction_rate":   round(
                len(corrections) / max(len(inferences), 1), 3
            ),
        }

    def list_corrections(self, limit: int = 20) -> List[FeedbackRecord]:
        """Return the most recent corrections (newest first)."""
        all_recs = list(self._iter_jsonl(self._corr_path))
        all_recs.reverse()
        result = []
        for r in all_recs[:limit]:
            result.append(FeedbackRecord(**{
                k: r.get(k, v)
                for k, v in asdict(FeedbackRecord()).items()
            }))
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _append(self, path: str, record: dict):
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _iter_jsonl(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass

    def _find_by_id(self, path: str, record_id: str) -> Optional[dict]:
        for rec in self._iter_jsonl(path):
            if rec.get("id") == record_id:
                return rec
        return None

    @staticmethod
    def _build_human_prompt(query: str, context: str, section: str) -> str:
        parts = []
        if section:
            parts.append(f"Section: {section}")
        parts.append(f"Research topic: {query}")
        if context:
            parts.append(f"Literature context (excerpt):\n{context[:1500]}")
        parts.append("Write this synthesis section now:")
        return "\n\n".join(parts)

    @staticmethod
    def _format_example(
        human:         str,
        assistant:     str,
        system_prompt: str,
        format:        str,
    ) -> dict:
        if format == "sharegpt":
            convs = []
            if system_prompt:
                convs.append({"from": "system", "value": system_prompt})
            convs.append({"from": "human",  "value": human})
            convs.append({"from": "gpt",    "value": assistant})
            return {"conversations": convs}
        # Alpaca format
        return {
            "instruction": system_prompt,
            "input":       human,
            "output":      assistant,
        }
