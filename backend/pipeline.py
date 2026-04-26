"""End-to-end pipeline for one company: company info → PDFs → extractions → merged statements."""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from . import company as company_mod
from . import extractor, merger, pdf_finder

ProgressFn = Callable[[str, dict], None]
MAX_REPORTS = 10  # 10 years of annual reports per company


def run_company(name: str, on_progress: ProgressFn) -> dict:
    def emit(stage: str, **fields):
        on_progress(stage, {"company": name, **fields})

    emit("resolving", message=f"Looking up {name}…")
    info = company_mod.resolve(name)
    emit("resolved", info=info)

    emit("crawling", message="Finding PDFs on IR site…")
    candidates = pdf_finder.crawl_pdfs(info["ir_url"], info.get("reports_url"))
    emit("crawled", candidate_count=len(candidates))

    annuals: list[dict] = []
    if candidates:
        emit("classifying", message=f"Classifying {len(candidates)} PDFs…")
        classified = pdf_finder.classify_pdfs(
            info["name"],
            candidates,
            on_progress=lambda stage, fields: emit(stage, **fields),
        )
        annuals = pdf_finder.annual_reports(classified)[:MAX_REPORTS]
        emit("classified", annual_count=len(annuals), annuals=[{"year": a.get("fiscal_year"), "url": a["url"]} for a in annuals])

    if not annuals:
        emit("web_fallback", message="Crawler found no annual reports — searching the web…")
        web_hits = pdf_finder.web_search_annual_reports(info["name"], info.get("ir_url"), max_years=MAX_REPORTS)
        verified = [h for h in web_hits if pdf_finder.verify_pdf_url(h["url"])]
        annuals = pdf_finder.annual_reports(verified)[:MAX_REPORTS]
        emit("web_fallback_done", found=len(web_hits), verified=len(verified), kept=len(annuals))

    if not annuals:
        emit("done", error="No annual reports found.", statements={})
        return {"company": info, "statements": {}}

    extractions: dict[int, dict[str, dict | None]] = {}

    def _process(a: dict):
        year = a["fiscal_year"]
        emit("downloading", year=year, message=f"Downloading annual report {year}…", url=a["url"])
        path = pdf_finder.download_pdf(a["url"])
        emit("downloaded", year=year, message=f"Downloaded annual report {year}.")
        emit("extracting", year=year, message=f"Finding and extracting statements for {year}…")
        ext = extractor.extract_all(path)
        found = {
            stmt: bool(data and data.get("line_items"))
            for stmt, data in ext.items()
        }
        emit("extracted", year=year, found=found)
        return year, ext

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_process, a): a for a in annuals}
        for fut in as_completed(futures):
            a = futures[fut]
            try:
                year, ext = fut.result()
                extractions[year] = ext
                emit("year_complete", year=year)
            except Exception as e:
                emit("extract_error", year=a.get("fiscal_year"), error=str(e), trace=traceback.format_exc(limit=2))

    emit("merging", message="Merging into 10-year tables…")
    merged: dict[str, dict] = {}
    for stmt in ("income_statement", "balance_sheet", "cash_flow"):
        per_year = {y: ex[stmt] for y, ex in extractions.items() if ex.get(stmt) and ex[stmt].get("line_items")}
        merged[stmt] = merger.merge_statement(stmt, per_year, info["name"])

    result = {"company": info, "annuals": annuals, "statements": merged}
    emit("done", statements_summary={k: len(v.get("rows", [])) for k, v in merged.items()})
    return result
