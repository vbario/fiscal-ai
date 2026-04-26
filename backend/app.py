"""FastAPI app: kick off reports, stream progress via SSE, return final tables."""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import cost, pipeline

app = FastAPI(title="Fiscal AI")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


class ReportRequest(BaseModel):
    companies: list[str]


# In-memory job registry. Each job has: companies, queues for SSE, results.
class Job:
    def __init__(self, companies: list[str]):
        self.id = uuid.uuid4().hex[:12]
        self.companies = companies
        self.events: list[dict[str, Any]] = []
        self.subscribers: list[asyncio.Queue] = []
        self.results: dict[str, dict] = {}
        self.done: bool = False
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit(self, event: dict[str, Any]):
        with self.lock:
            self.events.append(event)
            subs = list(self.subscribers)
            loop = self.loop
        if loop:
            for q in subs:
                # Schedule a put onto the queue from the worker thread.
                asyncio.run_coroutine_threadsafe(q.put(event), loop)


JOBS: dict[str, Job] = {}


def _run_job(job: Job):
    def make_progress(company: str):
        def cb(stage: str, data: dict):
            job.emit({"type": "progress", "company": company, "stage": stage, "data": data})
        return cb

    for name in job.companies:
        try:
            result = pipeline.run_company(name, make_progress(name))
            job.results[name] = result
            job.emit({"type": "company_done", "company": name})
        except Exception as e:
            job.results[name] = {"error": str(e), "trace": traceback.format_exc(limit=4)}
            job.emit({"type": "company_error", "company": name, "error": str(e)})

    job.done = True
    job.emit({"type": "all_done"})


@app.post("/api/reports")
async def create_report(req: ReportRequest):
    if not req.companies:
        raise HTTPException(400, "companies must be a non-empty list")
    job = Job([c.strip() for c in req.companies if c.strip()])
    job.loop = asyncio.get_running_loop()
    JOBS[job.id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job.id, "companies": job.companies}


@app.get("/api/reports/{job_id}/events")
async def stream_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    queue: asyncio.Queue = asyncio.Queue()
    # Replay history first.
    for e in list(job.events):
        await queue.put(e)
    with job.lock:
        job.subscribers.append(queue)

    async def event_gen():
        try:
            while True:
                event = await queue.get()
                yield {"event": event["type"], "data": json.dumps(event)}
                if event["type"] == "all_done":
                    break
        finally:
            with job.lock:
                if queue in job.subscribers:
                    job.subscribers.remove(queue)

    return EventSourceResponse(event_gen())


@app.get("/api/reports/{job_id}")
async def get_report(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JSONResponse({"job_id": job.id, "done": job.done, "results": job.results})


@app.get("/api/cost")
async def get_cost():
    return JSONResponse(cost.snapshot())


@app.post("/api/cost/reset")
async def reset_cost():
    cost.reset()
    return JSONResponse(cost.snapshot())


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
