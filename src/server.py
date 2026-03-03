"""FastAPI server — multi-user pipeline API.

Exposes the tool-generation pipeline as a REST API so client machines
(no GPU, no Ollama, no Docker) can submit queries and poll for results.

Architecture
------------
GPU machine
  ├── Ollama          :11434  (DeepSeek-R1, Qwen2.5-Coder)
  └── Pipeline server :8000   (this file)

Client machine
  └── HTTP  →  GPU machine:8000   (plain requests, no GPU needed)

Concurrency model
-----------------
- Each job gets its own LangGraph thread_id (isolated state).
- asyncio.Semaphore(max_concurrent_pipelines) gates simultaneous GPU runs.
  Extra jobs queue automatically and start when a slot frees.
- The sync pipeline runs in a ThreadPoolExecutor so the event loop stays free.

Endpoints
---------
  POST /api/generate          submit job  →  202 + {job_id}
  GET  /api/jobs/{job_id}     poll status / result
  GET  /api/jobs              list all jobs
  GET  /api/health            server + queue health
"""

import uuid
import time
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .pipeline import run_pipeline
from .logger_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Config
# ============================================================================

def _load_server_config() -> Dict[str, Any]:
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("server", {})
    return {}


_server_cfg = _load_server_config()
MAX_CONCURRENT: int = _server_cfg.get("max_concurrent_pipelines", 2)
JOB_TTL: int = _server_cfg.get("job_ttl_seconds", 3600)


# ============================================================================
# In-memory job store
# ============================================================================

class JobStatus:
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


_jobs: Dict[str, Dict[str, Any]] = {}
_semaphore: Optional[asyncio.Semaphore] = None
_executor: Optional[ThreadPoolExecutor] = None


# ============================================================================
# FastAPI app
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore, _executor
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    _executor  = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    logger.info(
        "Pipeline API started — max concurrent pipelines: %d, job TTL: %ds",
        MAX_CONCURRENT, JOB_TTL,
    )
    yield
    if _executor:
        _executor.shutdown(wait=False)


app = FastAPI(
    title="Code Generator Pipeline API",
    description=(
        "Multi-user REST API for the LangGraph tool-generation pipeline. "
        "Submit a query and data path, receive a job_id, then poll for results."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================================
# Request / Response models
# ============================================================================

class GenerateRequest(BaseModel):
    """Payload for submitting a new pipeline job."""
    query: str
    data_path: str


class JobResponse(BaseModel):
    """Returned for every job status query."""
    job_id:       str
    query:        str
    status:       str
    queued_at:    float
    started_at:   Optional[float] = None
    completed_at: Optional[float] = None
    result:       Optional[Dict[str, Any]] = None
    error:        Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect browser root hits to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    raise HTTPException(status_code=204)


@app.get("/api/tools/{tool_name}")
async def get_tool_code(tool_name: str) -> Dict[str, Any]:
    """Return the source code of a promoted tool by name.

    ``tool_name`` is the value returned in ``result.tool_name``, e.g.
    ``chi_square_weather_crash_independence_test_20260303_142409``.
    """
    # Strip .py if the caller included it
    tool_name = tool_name.removesuffix(".py")
    tool_path = Path("tools/active") / f"{tool_name}.py"
    if not tool_path.exists():
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found in active registry")
    return {
        "tool_name": tool_name,
        "tool_path": str(tool_path),
        "code": tool_path.read_text(encoding="utf-8"),
    }


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Server health + current queue state."""
    active = sum(1 for j in _jobs.values() if j["status"] == JobStatus.RUNNING)
    queued = sum(1 for j in _jobs.values() if j["status"] == JobStatus.QUEUED)
    return {
        "status": "ok",
        "active_pipelines":  active,
        "queued_pipelines":  queued,
        "max_concurrent":    MAX_CONCURRENT,
        "total_jobs_stored": len(_jobs),
    }


@app.post("/api/generate", response_model=JobResponse, status_code=202)
async def submit_job(req: GenerateRequest) -> Dict[str, Any]:
    """Submit a pipeline job.

    Returns immediately with ``status=queued`` and a ``job_id``.
    Poll ``GET /api/jobs/{job_id}`` until status is ``completed`` or ``failed``.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":       job_id,
        "query":        req.query,
        "data_path":    req.data_path,
        "status":       JobStatus.QUEUED,
        "queued_at":    time.time(),
        "started_at":   None,
        "completed_at": None,
        "result":       None,
        "error":        None,
    }
    asyncio.create_task(_run_job(job_id, req.query, req.data_path))
    logger.info("Job %s queued: %s", job_id, req.query[:80])
    return _as_response(job_id)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> Dict[str, Any]:
    """Poll a job's status and (when done) its result."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    _gc_old_jobs()
    return _as_response(job_id)


@app.get("/api/jobs")
async def list_jobs() -> list:
    """List all stored jobs with their current status."""
    _gc_old_jobs()
    return [_as_response(jid) for jid in _jobs]


# ============================================================================
# Background runner
# ============================================================================

async def _run_job(job_id: str, query: str, data_path: str) -> None:
    """Wait for a semaphore slot, then run the pipeline in the thread pool."""
    global _semaphore, _executor

    assert _semaphore is not None, "Semaphore not initialised — startup() must run first"
    async with _semaphore:                          # queue here if GPU is busy
        _jobs[job_id]["status"]     = JobStatus.RUNNING
        _jobs[job_id]["started_at"] = time.time()
        logger.info("Job %s started", job_id)

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _executor,
                run_pipeline,   # sync — runs in thread, event loop stays free
                query,
                data_path,
                job_id,         # used as LangGraph thread_id for checkpointer isolation
            )
            _jobs[job_id]["result"] = _extract_result(result)
            _jobs[job_id]["status"] = JobStatus.COMPLETED

        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
            _jobs[job_id]["error"]  = str(exc)
            _jobs[job_id]["status"] = JobStatus.FAILED

        finally:
            _jobs[job_id]["completed_at"] = time.time()
            logger.info(
                "Job %s -> %s (%.1fs)",
                job_id,
                _jobs[job_id]["status"],
                _jobs[job_id]["completed_at"] - _jobs[job_id]["started_at"],
            )


# ============================================================================
# Helpers
# ============================================================================

def _extract_result(state: Dict[str, Any]) -> Dict[str, Any]:
    """Distil the final LangGraph state into a compact, JSON-safe summary."""
    promoted = state.get("promoted_tool") or {}
    exec_out = state.get("execution_output") or {}
    errors   = state.get("projected_errors") or state.get("errors") or []

    return {
        "promoted":           bool(promoted),
        "tool_name":          promoted.get("name"),
        "tool_path":          promoted.get("path"),
        "repair_attempts":    state.get("repair_attempts", 0),
        "draft_output_path":  state.get("draft_output_path"),
        "execution": {
            "success":            exec_out.get("success", not exec_out.get("error")),
            "execution_time_ms":  exec_out.get("execution_time_ms"),
            "error":              exec_out.get("error"),
            "output":             exec_out.get("result"),        # full analysis result
            "summary":            exec_out.get("summary_markdown"),
        },
        "errors": errors,
    }


def _as_response(job_id: str) -> Dict[str, Any]:
    j = _jobs[job_id]
    return {
        "job_id":       j["job_id"],
        "query":        j["query"],
        "status":       j["status"],
        "queued_at":    j["queued_at"],
        "started_at":   j["started_at"],
        "completed_at": j["completed_at"],
        "result":       j["result"],
        "error":        j["error"],
    }


def _gc_old_jobs() -> None:
    """Remove completed/failed jobs older than JOB_TTL seconds."""
    cutoff = time.time() - JOB_TTL
    stale = [
        jid for jid, j in _jobs.items()
        if j["status"] in (JobStatus.COMPLETED, JobStatus.FAILED)
        and (j["completed_at"] or 0) < cutoff
    ]
    for jid in stale:
        del _jobs[jid]


# ============================================================================
# Entrypoint (python -m src.server)  or  uvicorn src.server:app
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = _server_cfg.get("host", "0.0.0.0")
    port = _server_cfg.get("port", 8000)
    uvicorn.run("src.server:app", host=host, port=port, reload=False, workers=1)
