"""
biovoice/core/task.py     — Task state machine
biovoice/config/settings.py — Pydantic settings loaded from .env
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ── Task ──────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    FETCHING   = "fetching"
    INDEXING   = "indexing"
    GENERATING = "generating"
    OUTPUT     = "output"
    DONE       = "done"
    FAILED     = "failed"


class Task:
    """Lightweight in-memory task object. For production use Redis/Celery."""

    def __init__(
        self,
        query:        str,
        agents:       List[str],
        output_types: List[str],
        task_id:      Optional[str] = None,
    ):
        self.task_id     = task_id or str(uuid.uuid4())[:8]
        self.query       = query
        self.agents      = agents
        self.output_types = output_types
        self.status      = TaskStatus.PENDING
        self.error:  Optional[str] = None
        self.created_at  = datetime.utcnow().isoformat()
        self.updated_at  = self.created_at

    def set_status(self, status: TaskStatus, error: Optional[str] = None) -> None:
        self.status     = status
        self.error      = error
        self.updated_at = datetime.utcnow().isoformat()
        print(f"[Task {self.task_id}] {status.value}"
              + (f" — {error}" if error else ""))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":     self.task_id,
            "query":       self.query,
            "agents":      self.agents,
            "status":      self.status.value,
            "error":       self.error,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
        }


