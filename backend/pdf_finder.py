"""Discover annual-report PDFs on a company's IR site, then classify them with the LLM."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from . import cache, llm
from .config import PDF_DIR

UA = "Mozilla/5.0 (compatible; FiscalAI/0.1; +https://example.com)"
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/pdf,*/*"}

# Max number of distinct pages to crawl per company while looking for PDFs.
MAX_PAGES = 12
# Max PDFs we'll bother classifying.
MAX_CANDIDATES = 60


def _same_site(a: str, b: str) -> bool:
    return urlparse(a).netloc.split(":")[0].lower().endswith(
        urlparse(b).netloc.split(":")[0].lower().split("www.")[-1]
    )


def _is_pdf_link(href: str) -> bool:
    return ".pdf" in href.lower()


def _fetch(client: httpx.Client, url: str) -> httpx.Response | None:
    try:
        r = client.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
        if r.status_code == 200:
            return r
    except Exception:
        return None
    return None


def crawl_pdfs(ir_url: str, reports_url: str | None = None) -> list[dict]:
    """Return a list of {url, anchor_text, source_page} for every PDF we found."""
    seeds = [u for u in [reports_url, ir_url] if u]
    if not seeds:
        return []

    seen_pages: set[str] = set()
    seen_pdfs: dict[str, dict] = {}
    queue: list[str] = list(dict.fromkeys(seeds))

    keywords = re.compile(
        r"(annual|integrated|results|reports?|investors?|financ|interim|half-year|q[1-4])",
        re.I,
    )

    with httpx.Client() as client:
        while queue and len(seen_pages) < MAX_PAGES:
            url = queue.pop(0)
            if url in seen_pages:
                continue
            seen_pages.add(url)
            r = _fetch(client, url)
            if not r or "html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                text = " ".join(a.get_text(" ", strip=True).split())[:200]
                if _is_pdf_link(href):
                    if href not in seen_pdfs:
                        seen_pdfs[href] = {
                            "url": href,
                            "anchor_text": text,
                            "source_page": url,
                        }
                elif _same_site(href, seeds[0]) and keywords.search(href + " " + text):
                    if href not in seen_pages and len(seen_pages) + len(queue) < MAX_PAGES * 2:
                        queue.append(href)

    return list(seen_pdfs.values())[:MAX_CANDIDATES]


CLASSIFY_SCHEMA = {
    "type": "object",
    "required": ["pdfs"],
    "properties": {
        "pdfs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["url", "kind", "fiscal_year"],
                "properties": {
                    "url": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "annual_report",
                            "interim_report",
                            "results_presentation",
                            "press_release",
                            "sustainability",
                            "governance",
                            "other",
                        ],
                    },
                    "fiscal_year": {
                        "type": ["integer", "null"],
                        "description": "Fiscal year covered (e.g. 2024). Null if not applicable.",
                    },
                    "language": {"type": ["string", "null"]},
                    "title": {"type": ["string", "null"]},
                },
            },
        }
    },
}


def classify_pdfs(company_name: str, candidates: list[dict]) -> list[dict]:
    """Ask the LLM to classify each candidate PDF based on its URL + anchor text."""
    if not candidates:
        return []

    key = "classify:" + hashlib.sha256(
        (company_name + "|" + "|".join(sorted(c["url"] for c in candidates))).encode()
    ).hexdigest()

    listing = "\n".join(
        f"- url: {c['url']}\n  text: {c['anchor_text']}" for c in candidates
    )
    prompt = (
        f"For the public company '{company_name}', classify each of the following PDF links "
        "from its investor relations site. Use the URL and anchor text to determine the kind "
        "(annual_report, interim_report, results_presentation, press_release, sustainability, "
        "governance, other) and the fiscal year. "
        "Annual reports include 'Annual Report', 'Integrated Annual Report', 'Annual Review' "
        "(but NOT sustainability reports or interim reports). "
        "Half-year and quarterly results are interim_report. "
        "If the year cannot be inferred from the URL/text, set fiscal_year to null."
        f"\n\nCandidates:\n{listing}"
    )
    out = llm.text_json(prompt, CLASSIFY_SCHEMA, cache_key=key)
    by_url = {p["url"]: p for p in out.get("pdfs", [])}
    enriched = []
    for c in candidates:
        meta = by_url.get(c["url"])
        if not meta:
            continue
        enriched.append({**c, **meta})
    return enriched


def annual_reports(classified: list[dict]) -> list[dict]:
    """Return one annual_report per fiscal_year (most-recent-looking URL preferred)."""
    by_year: dict[int, dict] = {}
    for p in classified:
        if p.get("kind") != "annual_report":
            continue
        fy = p.get("fiscal_year")
        if not isinstance(fy, int):
            continue
        # Prefer English when language is known.
        prev = by_year.get(fy)
        if prev is None:
            by_year[fy] = p
            continue
        if (p.get("language") or "").lower().startswith("en") and not (
            (prev.get("language") or "").lower().startswith("en")
        ):
            by_year[fy] = p
    return [by_year[y] for y in sorted(by_year.keys(), reverse=True)]


def download_pdf(url: str) -> Path:
    """Download a PDF (if not already cached) and return its local path."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", urlparse(url).path.rsplit("/", 1)[-1] or "file.pdf")
    path = PDF_DIR / f"{h}_{name}"
    if not path.exists():
        with httpx.Client() as client:
            r = client.get(url, headers=HEADERS, follow_redirects=True, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
    return path
