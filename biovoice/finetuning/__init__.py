"""
biovoice/finetuning — feedback collection and LoRA fine-tuning data export.

Usage pattern
-------------
1. Run BioVoice normally.
2. Expert reviews synthesis output and submits corrections via Gradio UI
   (or programmatically via FeedbackStore).
3. Export: FeedbackStore.export_jsonl() writes a sharegpt / alpaca-format
   JSONL file ready for LoRA fine-tuning with Unsloth / HuggingFace PEFT.
4. Fine-tune: see scripts/finetune_lora.py for an Unsloth recipe.
"""

from .feedback_store import FeedbackStore, FeedbackRecord

__all__ = ["FeedbackStore", "FeedbackRecord"]
