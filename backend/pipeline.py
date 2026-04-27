"""End-to-end pipeline for one company: company info → PDFs → extractions → merged statements."""
from __future__ import annotations

import contextvars
import datetime as dt
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from . import company as company_mod
from . import extractor, merger, pdf_finder

ProgressFn = Callable[[str, dict], None]
MAX_REPORTS = 10  # 10 years of annual reports per company


def _history_message(name: str, history: dict) -> str:
    if not history["is_public"]:
        return f"{name} is not publicly listed."
    yrs = history.get("years_public")
    ipo = history.get("ipo_year")
    if ipo and yrs is not None:
        plural = "s" if yrs != 1 else ""
        if yrs >= MAX_REPORTS:
            return f"{name} has been public since {ipo} ({yrs} year{plural}); fetching the last 10 years."
        return (
            f"{name} has been public since {ipo} ({yrs} year{plural}); fetching all available "
            f"annual reports (up to {history['max_years']})."
        )
    return f"{name} is publicly listed; listing year unknown — fetching the last 10 years."


def _public_history(info: dict) -> dict:
    """Compute years-public window and a human-readable note."""
    is_public = bool(info.get("is_public"))
    ipo_year = info.get("ipo_year") if isinstance(info.get("ipo_year"), int) else None
    today = dt.date.today()
    years_public: int | None = None
    note: str | None = None
    if not is_public:
        note = "Not currently a publicly listed company."
        max_years = 0
    elif ipo_year is None:
        # Public but listing year unknown — fall back to the full 10-year window.
        max_years = MAX_REPORTS
        note = "Listing year not confirmed; defaulting to the last 10 years of reports."
    else:
        years_public = max(0, today.year - ipo_year)
        if years_public >= MAX_REPORTS:
            max_years = MAX_REPORTS
        else:
            max_years = max(1, years_public + 1)  # include partial current year
            note = (
                f"Public since {ipo_year} ({years_public} year"
                f"{'s' if years_public != 1 else ''}); fewer than 10 years of annual reports available."
            )
    return {
        "is_public": is_public,
        "ipo_year": ipo_year,
        "ipo_date": info.get("ipo_date"),
        "years_public": years_public,
        "max_years": max_years,
        "public_status_note": info.get("public_status_note") or note,
    }


def run_company(name: str, on_progress: ProgressFn) -> dict:
    def emit(stage: str, **fields):
        on_progress(stage, {"company": name, **fields})

    emit("resolving", message=f"Looking up {name}…")
    info = company_mod.resolve(name)
    emit("resolved", info=info)

    history = _public_history(info)
    info["public_history"] = history
    emit("public_history", history=history, message=_history_message(info["name"], history))

    if not history["is_public"]:
        emit("done", error="Company is not publicly listed.", statements={})
        return {"company": info, "statements": {}}

    max_years = history["max_years"]

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
        annuals = pdf_finder.annual_reports(classified)[:max_years]
        emit("classified", annual_count=len(annuals), annuals=[{"year": a.get("fiscal_year"), "url": a["url"]} for a in annuals])

    today_year = dt.date.today().year
    expected_years = set()
    if max_years:
        # Most recent fiscal year is usually last year; current year only if a report has been filed.
        latest_filed = max((a.get("fiscal_year") for a in annuals if isinstance(a.get("fiscal_year"), int)), default=today_year - 1)
        latest_year = max(latest_filed, today_year - 1)
        expected_years = {latest_year - i for i in range(max_years)}
    have_years = {a.get("fiscal_year") for a in annuals if isinstance(a.get("fiscal_year"), int)}
    missing_years = sorted(expected_years - have_years, reverse=True)

    if missing_years:
        reason = "no annual reports" if not annuals else f"missing {len(missing_years)} year(s): {missing_years}"
        emit("web_fallback", message=f"Crawler {reason} — searching the web…")
        web_hits = pdf_finder.web_search_annual_reports(info["name"], info.get("ir_url"), max_years=max_years)
        new_hits = [h for h in web_hits if h.get("fiscal_year") in missing_years]
        verified = [h for h in new_hits if pdf_finder.verify_pdf_url(h["url"])]
        merged = pdf_finder.annual_reports(annuals + verified)[:max_years]
        added = len(merged) - len(annuals)
        annuals = merged
        emit("web_fallback_done", found=len(web_hits), verified=len(verified), added=added, total=len(annuals))

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

    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(ctx.run, _process, a): a for a in annuals}
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
