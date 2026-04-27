# Fiscal AI Report Generation App

Type a public European company name; get a clean, comparable 10-year table of its Income Statement, Balance Sheet, and Cash Flow — extracted directly from the company's own annual reports.

## What it does

Given a company name (e.g. `Heineken, Adyen, ASML`):

1. **Resolves** the company to its legal name, ticker, exchange, country, IR site URL, and listing history (IPO year, years public).
2. **Finds** up to the last 10 annual report PDFs from the company's own investor relations site.
3. **Extracts** the three primary financial statements from each PDF using a hybrid text + vision pipeline.
4. **Merges** the per-year extractions into a single canonical table per statement, normalizing line-item names, units, and currencies.
5. **Renders** an interactive table per statement, downloadable as PDF.

Reports are streamed back over Server-Sent Events with live progress and live OpenAI cost tracking.

## How it works

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐      ┌───────────┐
│   resolve   │ ───▶ │  find PDFs   │ ───▶ │   extract    │ ───▶ │   merge   │
│ (web search)│      │ (crawl + LLM │      │ (text/vision │      │  (LLM     │
│             │      │  fallback)   │      │  per stmt)   │      │  schema-  │
└─────────────┘      └──────────────┘      └──────────────┘      │  align)   │
                                                                  └───────────┘
```

### 1. Company resolution — `backend/company.py`

A single OpenAI Responses call with the built-in `web_search` tool returns a JSON object containing:

- `name`, `ticker`, `exchange`, `country`
- `ir_url`, optional `reports_url`
- `is_public`, `ipo_year`, `ipo_date`, `public_status_note`

The `ipo_year` drives how many annual reports we'll look for: a 4-year-old IPO means at most 5 reports, not 10. The result is cached in SQLite keyed by company name.

### 2. PDF discovery — `backend/pdf_finder.py`

Two-tier strategy:

**(a) HTTP crawler** — `crawl_pdfs()` walks the IR site (BFS, max 12 pages) collecting `.pdf` / `/download/` / `/dam/` links by both anchor parsing and raw-text URL scraping (handles JSON/Nuxt payloads). Same-site is computed across sibling subdomains so corporate IR redirects work.

  - **Age-gate / consent-gate handling**: when a fetched page looks like an age-verification form (path matches `/age-gate/...` or page contains an age-gate `<form>`), the crawler programmatically POSTs the form with adult defaults (DOB 1980-01-01, country=NL, all checkboxes ticked, hidden Drupal `form_build_id` preserved) so the resulting cookie (`age_gate_access=...`) lets subsequent fetches see real content. One attempt per host, persisted on a shared `httpx.Client` cookie jar.

**(b) LLM web-search fallback** — `web_search_annual_reports()` asks the model (with `web_search` enabled) for direct PDF URLs of a company's annual reports for specific missing years. Each returned URL is verified via HEAD then a ranged GET (handles CDNs that 405 HEAD; checks the `%PDF` magic bytes).

The pipeline (`backend/pipeline.py`) computes the expected 10-year window after classification and triggers the fallback for any missing years — most IR sites only link to the 2-3 most recent reports; the fallback fills the rest.

**Classification** (`classify_pdfs()`): all crawled candidates are batched (15 per chunk) into an LLM call that tags each as `annual_report | interim_report | results_presentation | press_release | sustainability | governance | other` plus fiscal year. Annual reports are deduped to one per fiscal year, English preferred.

### 3. Statement extraction — `backend/extractor.py`

Per annual-report PDF:

1. **Locate statement pages** by regex-matching multilingual headings (`consolidated income statement`, `geconsolideerde winst- en verliesrekening`, `compte de résultat consolidé`, `Konzern-Gewinn- und Verlustrechnung`, etc.).
2. **Score candidate pages** (`_statement_page_score()`) to prefer the actual primary statement over table-of-contents pages and note-section reprints. TOCs and note pages are penalized; "as at"/"for the years ended"/"total assets" anchors are rewarded.
3. **Render the chosen pages** at 180 DPI with PyMuPDF.
4. **Send the page images** to a vision-capable model (`llm.vision_json`) with a strict JSON schema asking for `periods`, `currency`, `unit`, and a list of `line_items` with `label`, `values[]`, `indent_level`, `is_section_header`, `is_subtotal`.

This avoids fragile PDF-to-text table parsing — vision handles multi-column European report layouts that defeat `pdfplumber` alone.

### 4. Merging — `backend/merger.py`

Per statement, across N years of extractions:

- The **most recent report's line items** define the canonical schema (labels and ordering).
- For each older report's line items, an LLM call maps them to the canonical labels (`rapidfuzz` pre-filters obvious matches to keep the prompt small) or marks them `new` — `new` rows are appended in their original position.
- For each `(canonical_label, period_end)` cell, the value from the most-recent report containing that period wins (latest restatement beats earlier filings).
- Units are normalized to millions per company; currency is preserved.

### 5. API + streaming — `backend/app.py`

FastAPI app, single SSE endpoint per job:

- `POST /api/reports` → `{ job_id }`. Companies run sequentially; per-PDF extraction runs in a `ThreadPoolExecutor(max_workers=3)`.
- `GET /api/reports/{job_id}/stream` → SSE stream of `progress`, `cost_update`, `company_done`, `company_error`, `all_done` events.
- `GET /api/reports/{job_id}` → final JSON snapshot.
- `GET /api/reports/{job_id}/companies/{name}/export.pdf` → server-rendered PDF table (PyMuPDF).
- `GET /api/cost` / `POST /api/cost/reset` → session-wide LLM spend.

Saved jobs land in `data/reports/*.json`, replayable via the "Load saved report" dropdown.

### Cost tracking — `backend/cost.py`

Every LLM call records token usage by model into a thread-safe accumulator. A pub/sub `subscribe()` API lets `app.py` attach a per-job listener that emits a `cost_update` SSE event the moment any call returns — so the spend ticks up in real time, not just at progress checkpoints. Pricing for `gpt-5*`, `gpt-4.1*`, `gpt-4o*` lives in the `PRICING` table.

### Caching — `backend/cache.py`

Single SQLite KV store at `data/fiscal.sqlite`. Wraps:

- Company resolution (`company:v2:<name>`)
- PDF classification chunks (`classify:<sha>`)
- Per-statement extractions (`extract:<sha-of-pages>`)
- LLM merge mappings
- Web-search fallback results (`web_pdfs:<sha>`)

Re-running the same company is essentially free after the first pass.

## Tech stack

- **Backend**: FastAPI · `sse-starlette` · `httpx` · `BeautifulSoup4` · `pdfplumber` · `PyMuPDF (fitz)` · `pypdf` · `rapidfuzz`
- **LLM**: OpenAI Responses API (`gpt-5` by default) — used in three modes: web-search, vision, and structured JSON
- **Frontend**: a single `frontend/index.html` (no build step) — vanilla JS, light/dark theme, SSE consumer
- **Storage**: SQLite KV cache + JSON snapshots on disk

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8010
```

Then open <http://127.0.0.1:8010>.

Optional env vars:

- `OPENAI_MODEL` (default `gpt-5`) — used for resolution, classification, merging
- `OPENAI_VISION_MODEL` (default = `OPENAI_MODEL`) — used for statement extraction

## Repo layout

```
backend/
  app.py          FastAPI + SSE + per-job runner + PDF export
  pipeline.py     Orchestration: resolve → discover → extract → merge
  company.py      LLM company resolution (incl. listing history)
  pdf_finder.py   Crawler + age-gate pass + LLM web-search fallback
  extractor.py    Page selection + vision-based statement extraction
  merger.py       Cross-year canonical-schema alignment
  llm.py          OpenAI Responses helpers (web-search, vision, text)
  cost.py         Token/$ accumulator + pub/sub for live cost streaming
  cache.py        SQLite KV
  config.py       Env + paths
frontend/
  index.html      Single-file UI (SSE consumer, statement tabs, theme toggle)
data/             (gitignored) PDFs, cache DB, saved reports
```

## Limitations

- European focus: the resolver is prompted for European companies and the heading regex covers EN/NL/FR/DE.
- Annual reports only — half-year/interim filings are explicitly excluded.
- Statement extraction quality depends on the report's layout; deeply nested or non-standard formats can lose indent levels.
- Reports older than the company's IPO are not fetched; private companies abort early.
