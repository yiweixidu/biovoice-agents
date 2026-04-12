"""
biovoice/cli/main.py   — Click CLI  (`biovoice` command)
biovoice/bots/gateway.py — FastAPI webhook gateway for Feishu / DingTalk
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# Bot gateway (FastAPI)
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel

_app = FastAPI(title="BioVoice-Agents Bot Gateway")
_tasks: dict = {}  # in-memory; replace with Redis for production


class TaskSummary(PydanticModel):
    task_id:    str
    status:     str
    query:      str
    error:      str | None = None


def _parse_command(text: str) -> dict:
    """
    Parse a bot command like:
      /review broadly neutralizing antibodies influenza --agents pubmed,pdb
    Returns {"query": "...", "agents": [...], "output": [...]}
    """
    import re
    text = text.strip().lstrip("/review").strip()
    agents_match = re.search(r"--agents\s+([\w,]+)", text)
    output_match = re.search(r"--output\s+([\w,]+)", text)
    agents = agents_match.group(1).split(",") if agents_match else ["pubmed"]
    output = output_match.group(1).split(",") if output_match else ["review", "ppt"]
    query  = re.sub(r"--\w+\s+[\w,]+", "", text).strip()
    return {"query": query, "agents": agents, "output": output}


async def _run_task(task_id: str, command: dict, platform: str, channel_id: str):
    """Background worker — executed via BackgroundTasks."""
    settings = BioVoiceSettings()
    orch = BioVoiceOrchestrator(settings.to_orchestrator_config())
    _tasks[task_id] = {"status": "running", "query": command["query"]}
    try:
        result = await orch.run(
            query=command["query"],
            agent_names=command["agents"],
            output_types=command["output"],
        )
        _tasks[task_id]["status"] = "done"
        _tasks[task_id]["review_snippet"] = result.get("review", "")[:400]
        _tasks[task_id]["ppt_file"]       = result.get("ppt_file")
        _tasks[task_id]["video_file"]     = result.get("video_file")
    except Exception as exc:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"]  = str(exc)


@_app.post("/webhook/{platform}")
async def handle_webhook(
    platform: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Receives bot events from Feishu or DingTalk."""
    import uuid
    payload = await request.json()

    # Extract text — both platforms put it in different keys
    text = (
        payload.get("event", {}).get("message", {}).get("content", "{}")
    )
    if isinstance(text, str):
        try:
            text = json.loads(text).get("text", text)
        except json.JSONDecodeError:
            pass

    if not text or "/review" not in text:
        return JSONResponse({"code": 0, "msg": "not a review command"})

    command = _parse_command(text)
    task_id = str(uuid.uuid4())[:8]
    channel_id = payload.get("event", {}).get("message", {}).get("chat_id", "")

    background_tasks.add_task(
        _run_task, task_id, command, platform, channel_id
    )
    return JSONResponse({
        "code": 0,
        "msg": f"Task {task_id} started for: {command['query'][:60]}",
    })


@_app.get("/api/task/{task_id}/status", response_model=TaskSummary)
def get_task_status(task_id: str):
    t = _tasks.get(task_id)
    if not t:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return TaskSummary(
        task_id=task_id,
        status=t.get("status", "unknown"),
        query=t.get("query", ""),
        error=t.get("error"),
    )


@_app.get("/api/task/{task_id}/output")
def get_task_output(task_id: str):
    t = _tasks.get(task_id)
    if not t:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return {
        "task_id":       task_id,
        "review_snippet": t.get("review_snippet", ""),
        "ppt_file":      t.get("ppt_file"),
        "video_file":    t.get("video_file"),
    }


@_app.get("/api/agent/list")
def list_available_agents():
    AgentRegistry.load_plugins()
    return {"agents": AgentRegistry.list_agents()}


def start_bot_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(_app, host=host, port=port)