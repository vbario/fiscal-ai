"""FastAPI app: kick off reports, stream progress via SSE, return final tables."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import fitz
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import config, cost, llm, pipeline

app = FastAPI(title="Fiscal AI")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
REPORT_DIR = ROOT / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class ReportRequest(BaseModel):
    companies: list[str]
    api_key: Optional[str] = None
    model: Optional[str] = None


class RetryRequest(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None


# In-memory job registry. Each job has: companies, queues for SSE, results.
class Job:
    def __init__(self, companies: list[str], job_id: str | None = None, created_at: str | None = None):
        self.id = job_id or uuid.uuid4().hex[:12]
        self.companies = companies
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.events: list[dict[str, Any]] = []
        self.subscribers: list[asyncio.Queue] = []
        self.results: dict[str, dict] = {}
        self.done: bool = False
        self.last_cost_usd: float = 0.0
        self.running_companies: set[str] = set()
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        # User-supplied overrides (held in memory only — never persisted).
        self.api_key: str | None = None
        self.model: str | None = None

    def emit(self, event: dict[str, Any]):
        self.updated_at = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.events.append(event)
            subs = list(self.subscribers)
            loop = self.loop
        if loop:
            for q in subs:
                # Schedule a put onto the queue from the worker thread.
                asyncio.run_coroutine_threadsafe(q.put(event), loop)
        save_job(self)

    def to_record(self, include_events: bool = True) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "companies": self.companies,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "done": self.done,
            "results": self.results,
            "events": self.events if include_events else [],
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Job":
        job = cls(record.get("companies", []), job_id=record.get("job_id"), created_at=record.get("created_at"))
        job.updated_at = record.get("updated_at") or job.created_at
        job.done = bool(record.get("done"))
        job.results = record.get("results", {})
        job.events = record.get("events", [])
        return job


JOBS: dict[str, Job] = {}


def _job_path(job_id: str) -> Path:
    return REPORT_DIR / f"{job_id}.json"


def save_job(job: Job) -> None:
    tmp = REPORT_DIR / f"{job.id}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(job.to_record(), indent=2), encoding="utf-8")
    tmp.replace(_job_path(job.id))


def load_jobs() -> None:
    for path in REPORT_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            job = Job.from_record(record)
            if not job.done:
                now = datetime.now(timezone.utc).isoformat()
                for company in job.companies:
                    if company not in job.results:
                        job.results[company] = {"error": "Interrupted by server restart."}
                        job.events.append(
                            {
                                "type": "company_error",
                                "company": company,
                                "error": "Interrupted by server restart.",
                            }
                        )
                job.done = True
                job.updated_at = now
                job.events.append({"type": "all_done"})
                save_job(job)
            JOBS[job.id] = job
        except Exception:
            continue


load_jobs()


def _emit_cost_if_changed(job: Job) -> None:
    snap = cost.snapshot()
    usd = float(snap.get("usd") or 0.0)
    if usd > job.last_cost_usd:
        delta = round(usd - job.last_cost_usd, 6)
        job.last_cost_usd = usd
        job.emit({"type": "cost_update", "delta_usd": delta, "cost": snap})


def _make_progress(job: Job, company: str):
    def cb(stage: str, data: dict):
        job.emit({"type": "progress", "company": company, "stage": stage, "data": data})
        _emit_cost_if_changed(job)

    return cb


def _run_company_once(job: Job, name: str) -> None:
    try:
        llm.set_overrides(api_key=job.api_key, model=job.model)
        result = pipeline.run_company(name, _make_progress(job, name))
        job.results[name] = result
        save_job(job)
        _emit_cost_if_changed(job)
        job.emit({"type": "company_done", "company": name})
    except Exception as e:
        job.results[name] = {"error": str(e), "trace": traceback.format_exc(limit=4)}
        save_job(job)
        _emit_cost_if_changed(job)
        job.emit({"type": "company_error", "company": name, "error": str(e)})


def _attach_cost_listener(job: Job):
    def listener(_model: str, _delta_usd: float) -> None:
        _emit_cost_if_changed(job)

    cost.subscribe(listener)
    return listener


def _run_job(job: Job):
    listener = _attach_cost_listener(job)
    try:
        for name in job.companies:
            _run_company_once(job, name)
        job.done = True
        save_job(job)
        job.emit({"type": "all_done"})
    finally:
        cost.unsubscribe(listener)


def _run_retry(job: Job, name: str) -> None:
    listener = _attach_cost_listener(job)
    try:
        _run_company_once(job, name)
        with job.lock:
            job.running_companies.discard(name)
            still_running = bool(job.running_companies)
        if not still_running:
            job.done = True
            save_job(job)
            job.emit({"type": "all_done"})
    finally:
        cost.unsubscribe(listener)


def _normalize_overrides(api_key: str | None, model: str | None) -> tuple[str | None, str | None]:
    key = (api_key or "").strip() or None
    mdl = (model or "").strip() or None
    if mdl is not None and mdl not in config.ALLOWED_MODEL_IDS:
        raise HTTPException(400, f"model must be one of: {sorted(config.ALLOWED_MODEL_IDS)}")
    return key, mdl


@app.get("/api/models")
async def list_models():
    return JSONResponse({"models": config.ALLOWED_MODELS, "default": config.MODEL})


@app.post("/api/reports")
async def create_report(req: ReportRequest):
    if not req.companies:
        raise HTTPException(400, "companies must be a non-empty list")
    api_key, model = _normalize_overrides(req.api_key, req.model)
    job = Job([c.strip() for c in req.companies if c.strip()])
    job.api_key = api_key
    job.model = model
    job.loop = asyncio.get_running_loop()
    JOBS[job.id] = job
    save_job(job)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job.id, "companies": job.companies}


@app.post("/api/reports/{job_id}/companies/{company_name}/retry")
async def retry_company(job_id: str, company_name: str, body: Optional[RetryRequest] = None):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    decoded = unquote(company_name)
    if decoded not in job.companies and decoded not in job.results:
        raise HTTPException(404, "company not found in report")
    if not job.done:
        raise HTTPException(409, "report is still running")

    api_key, model = _normalize_overrides(
        body.api_key if body else None, body.model if body else None
    )

    with job.lock:
        if decoded in job.running_companies:
            raise HTTPException(409, "company retry is already running")
        event_count = len(job.events)
        job.running_companies.add(decoded)
        job.done = False
        job.results.pop(decoded, None)
        job.loop = asyncio.get_running_loop()
        # Refresh overrides if the caller supplied them; otherwise keep prior.
        if api_key is not None:
            job.api_key = api_key
        if model is not None:
            job.model = model

    save_job(job)
    job.emit({"type": "retry_started", "company": decoded})
    threading.Thread(target=_run_retry, args=(job, decoded), daemon=True).start()
    return {"job_id": job.id, "company": decoded, "event_count": event_count}


@app.get("/api/reports")
async def list_reports():
    reports = []
    for job in sorted(JOBS.values(), key=lambda j: j.updated_at, reverse=True):
        reports.append(
            {
                "job_id": job.id,
                "companies": job.companies,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "done": job.done,
                "result_companies": list(job.results.keys()),
            }
        )
    return JSONResponse({"reports": reports})


@app.get("/api/reports/{job_id}/events")
async def stream_events(job_id: str, since: int = 0):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    queue: asyncio.Queue = asyncio.Queue()
    # Replay history first.
    history = list(job.events)
    since = max(0, min(int(since or 0), len(history)))
    for e in history[since:]:
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
async def get_report(job_id: str, include_events: bool = False):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JSONResponse(job.to_record(include_events=include_events))


def _fmt_pdf_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        n = float(value)
    except Exception:
        return str(value)
    if abs(n) >= 1000:
        return f"{n:,.0f}"
    return f"{n:,.2f}".rstrip("0").rstrip(".")


def _draw_wrapped(page: fitz.Page, text: str, rect: fitz.Rect, fontsize: float = 8, bold: bool = False, align: int = 0) -> None:
    page.insert_textbox(
        rect,
        text,
        fontsize=fontsize,
        fontname="hebo" if bold else "helv",
        align=align,
        color=(0.05, 0.06, 0.08),
    )


def _company_pdf(company_name: str, data: dict) -> bytes:
    doc = fitz.open()
    company = data.get("company", {})
    title = company.get("name") or company_name
    statements = data.get("statements") or {}
    labels = {
        "income_statement": "Income Statement",
        "balance_sheet": "Balance Sheet",
        "cash_flow": "Cash Flow Statement",
    }

    for stmt_key, stmt_label in labels.items():
        stmt = statements.get(stmt_key) or {}
        periods = stmt.get("periods") or []
        rows = stmt.get("rows") or []
        page = doc.new_page(width=842, height=595)
        margin = 28
        y = margin
        _draw_wrapped(page, title, fitz.Rect(margin, y, 814, y + 24), fontsize=14, bold=True)
        y += 26
        meta = f"{stmt_label} | Currency: {stmt.get('currency') or '?'} | Unit: {stmt.get('unit') or ''}"
        _draw_wrapped(page, meta, fitz.Rect(margin, y, 814, y + 16), fontsize=8)
        y += 22

        label_w = 250
        col_count = max(len(periods), 1)
        col_w = (842 - margin * 2 - label_w) / col_count
        row_h = 14

        def header(current_y: float) -> float:
            page.draw_rect(fitz.Rect(margin, current_y, 814, current_y + row_h), color=(0.78, 0.82, 0.88), fill=(0.92, 0.94, 0.97), width=0.4)
            _draw_wrapped(page, "", fitz.Rect(margin + 2, current_y + 2, margin + label_w - 4, current_y + row_h), fontsize=7, bold=True)
            for i, p in enumerate(periods):
                x = margin + label_w + i * col_w
                _draw_wrapped(page, p.get("label") or p.get("period_end") or "", fitz.Rect(x + 2, current_y + 2, x + col_w - 2, current_y + row_h), fontsize=7, bold=True, align=2)
            return current_y + row_h

        y = header(y)
        for row in rows:
            if y + row_h > 565:
                page = doc.new_page(width=842, height=595)
                y = margin
                _draw_wrapped(page, f"{title} - {stmt_label}", fitz.Rect(margin, y, 814, y + 20), fontsize=11, bold=True)
                y += 24
                y = header(y)

            fill = (0.97, 0.98, 0.99) if row.get("is_section_header") else None
            if fill:
                page.draw_rect(fitz.Rect(margin, y, 814, y + row_h), color=(0.9, 0.92, 0.95), fill=fill, width=0.2)
            indent = min(int(row.get("indent_level") or 0), 2) * 10
            _draw_wrapped(page, row.get("label") or "", fitz.Rect(margin + 2 + indent, y + 2, margin + label_w - 4, y + row_h), fontsize=7, bold=bool(row.get("is_section_header") or row.get("is_subtotal")))
            for i, value in enumerate(row.get("values") or []):
                x = margin + label_w + i * col_w
                _draw_wrapped(page, _fmt_pdf_value(value), fitz.Rect(x + 2, y + 2, x + col_w - 2, y + row_h), fontsize=7, align=2)
            y += row_h

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue()


@app.get("/api/reports/{job_id}/companies/{company_name}/export.pdf")
async def export_company_pdf(job_id: str, company_name: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    decoded = unquote(company_name)
    data = job.results.get(decoded)
    if not data:
        # Fall back to matching by resolved company name.
        for key, value in job.results.items():
            if (value.get("company") or {}).get("name") == decoded:
                decoded, data = key, value
                break
    if not data or data.get("error"):
        raise HTTPException(404, "company report not found")
    pdf = _company_pdf(decoded, data)
    filename = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in decoded) + "_financials.pdf"
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
