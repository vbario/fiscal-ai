"""Discover annual-report PDFs on a company's IR site, then classify them with the LLM."""
from __future__ import annotations

import html
import hashlib
import re
import time
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urljoin, urlparse

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
# Keep classification prompts small enough that one slow response does not kill
# the whole report generation.
CLASSIFY_CHUNK_SIZE = 15
CLASSIFY_RETRIES = 0

ProgressFn = Callable[[str, dict], None]


def _host(url: str) -> str:
    return urlparse(url).netloc.split(":")[0].lower().removeprefix("www.")


def _site_root(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _same_site(a: str, b: str) -> bool:
    """Allow IR pages to link across sibling corporate subdomains."""
    ha, hb = _host(a), _host(b)
    return ha == hb or ha.endswith("." + hb) or hb.endswith("." + ha) or _site_root(ha) == _site_root(hb)


def _is_document_link(href: str) -> bool:
    low = href.lower()
    return (
        ".pdf" in low
        or "/download" in low
        or "/dam/" in low
        or "/asset/" in low
    )


def _context_text(raw_html: str, start: int, end: int, radius: int = 450) -> str:
    snippet = raw_html[max(0, start - radius) : min(len(raw_html), end + radius)]
    snippet = re.sub(r"<[^>]+>", " ", snippet)
    snippet = html.unescape(unquote(snippet))
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet[:500]


def _add_pdf_candidate(seen_pdfs: dict[str, dict], href: str, text: str, source_page: str) -> None:
    if href not in seen_pdfs:
        seen_pdfs[href] = {
            "url": href,
            "anchor_text": text[:500],
            "source_page": source_page,
        }


def _page_context(soup: BeautifulSoup, url: str) -> str:
    parts: list[str] = []
    if soup.title and soup.title.string:
        parts.append(soup.title.string.strip())
    h1 = soup.find("h1")
    if h1:
        parts.append(" ".join(h1.get_text(" ", strip=True).split()))
    parts.append(url)
    return " | ".join(p for p in parts if p)


def _prioritize_page(url: str) -> bool:
    path = urlparse(url).path.lower().rstrip("/")
    return bool(re.search(r"/financials/\d{4}$", path) or re.search(r"/reports?/\d{4}$", path))


def _candidate_score(c: dict) -> tuple[int, str]:
    text = (c.get("anchor_text") or "") + " " + c.get("url", "") + " " + c.get("source_page", "")
    low = text.lower()
    score = 0
    if "annual report" in low or "annual-report" in low:
        score += 100
    if re.search(r"/financials/\d{4}\b", low) or re.search(r"/reports?/\d{4}\b", low):
        score += 50
    if "consolidated financial statements" in low:
        score += 30
    if "ixbrl" in low or "pillar 3" in low or "transparency and disclosure" in low:
        score -= 80
    if re.search(r"\bh[12]\b|half[- ]year|interim|business update|press release", low):
        score -= 20
    return (-score, c.get("url", ""))


_GATE_PATH_RE = re.compile(r"/(age[-_ ]?gate|age[-_ ]?verification|age[-_ ]?check)\b", re.I)
_GATE_HINTS = ("age-gate", "age_gate", "age verification", "date of birth", "verify your age")


def _looks_like_gate(response: httpx.Response) -> bool:
    if _GATE_PATH_RE.search(str(response.url)):
        return True
    body = response.text.lower()
    if "<form" not in body:
        return False
    return any(h in body for h in _GATE_HINTS)


def _pass_gate(client: httpx.Client, response: httpx.Response) -> bool:
    """If the response is an age/consent gate, submit it with adult defaults."""
    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return False
    form = None
    for f in soup.find_all("form"):
        cls = " ".join(f.get("class") or []).lower()
        action = (f.get("action") or "").lower()
        fid = (f.get("id") or "").lower()
        haystack = f"{cls} {action} {fid}"
        if "age" in haystack or "gate" in haystack or "verify" in haystack:
            form = f
            break
    if form is None:
        # Fall back to any form on a gate-path URL.
        if not _GATE_PATH_RE.search(str(response.url)):
            return False
        form = soup.find("form")
        if form is None:
            return False

    action_url = urljoin(str(response.url), form.get("action") or str(response.url))
    method = (form.get("method") or "post").lower()
    data: dict[str, str] = {}
    for el in form.find_all(["input", "select", "textarea"]):
        name = el.get("name")
        if not name:
            continue
        low = name.lower()
        if el.name == "select":
            opts = el.find_all("option")
            chosen = next((o.get("value") for o in opts if o.has_attr("selected")), None)
            if chosen is None:
                chosen = next(
                    (o.get("value") for o in opts if (o.get("value") or "").strip()),
                    "",
                )
            if "country" in low:
                chosen = "NL"
            data[name] = chosen
            continue
        itype = (el.get("type") or "text").lower()
        if itype in ("button", "image", "reset"):
            continue
        if "day" in low:
            data[name] = "1"
        elif "month" in low:
            data[name] = "1"
        elif "year" in low and "form" not in low:
            data[name] = "1980"
        elif itype == "checkbox":
            data[name] = el.get("value") or "1"
        elif itype == "radio":
            if el.has_attr("checked") and name not in data:
                data[name] = el.get("value") or ""
        elif itype == "submit":
            if "op" not in data:
                data[name] = el.get("value") or "Enter"
        else:
            value = el.get("value") or ""
            if value.lower().startswith("replace with"):
                value = ""
            data[name] = value
    try:
        if method == "post":
            client.post(action_url, data=data, headers=HEADERS, follow_redirects=True, timeout=30)
        else:
            client.get(action_url, params=data, headers=HEADERS, follow_redirects=True, timeout=30)
        return True
    except Exception:
        return False


def _fetch(client: httpx.Client, url: str, _gates_passed: set[str] | None = None) -> httpx.Response | None:
    try:
        r = client.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
        if r.status_code != 200:
            return None
        if _gates_passed is not None and _looks_like_gate(r):
            host = _host(str(r.url))
            if host not in _gates_passed:
                _gates_passed.add(host)
                if _pass_gate(client, r):
                    r2 = client.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
                    if r2.status_code == 200:
                        return r2
        return r
    except Exception:
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
        r"(annual|integrated|results|reports?|financ|interim|half-year|q[1-4])",
        re.I,
    )

    gates_passed: set[str] = set()
    with httpx.Client(cookies=httpx.Cookies()) as client:
        while queue and len(seen_pages) < MAX_PAGES:
            url = queue.pop(0)
            if url in seen_pages:
                continue
            seen_pages.add(url)
            r = _fetch(client, url, gates_passed)
            if not r or "html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            page_context = _page_context(soup, url)
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                text = " ".join(a.get_text(" ", strip=True).split())[:200]
                text_with_context = f"{text} | {page_context}" if page_context else text
                if _is_document_link(href):
                    _add_pdf_candidate(seen_pdfs, href, text_with_context, url)
                elif _same_site(href, seeds[0]) and keywords.search(href + " " + text):
                    if href not in seen_pages and (
                        _prioritize_page(href) or len(seen_pages) + len(queue) < MAX_PAGES * 2
                    ):
                        if _prioritize_page(href):
                            queue.insert(0, href)
                        else:
                            queue.append(href)

            # Many modern IR sites render reports from JSON/Nuxt payloads instead
            # of plain anchors. Pull out absolute URLs and keep nearby payload text
            # so classification can infer the year/type from surrounding labels.
            for m in re.finditer(r"https?://[^\"'<>\\\s]+", r.text):
                href = html.unescape(m.group(0)).rstrip(").,;")
                if not _same_site(href, seeds[0]):
                    continue
                context = _context_text(r.text, m.start(), m.end())
                if _is_document_link(href):
                    _add_pdf_candidate(seen_pdfs, href, f"{context} | {page_context}", url)
                elif keywords.search(href + " " + context):
                    if href not in seen_pages and (
                        _prioritize_page(href) or len(seen_pages) + len(queue) < MAX_PAGES * 2
                    ):
                        if _prioritize_page(href):
                            queue.insert(0, href)
                        else:
                            queue.append(href)

    return sorted(seen_pdfs.values(), key=_candidate_score)[:MAX_CANDIDATES]


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


def _classify_chunk(company_name: str, candidates: list[dict]) -> list[dict]:
    # v2: tightened prompt to reject partial / supplementary PDFs.
    key = "classify:v2:" + hashlib.sha256(
        (company_name + "|" + "|".join(sorted(c["url"] for c in candidates))).encode()
    ).hexdigest()

    listing = "\n".join(
        f"- url: {c['url']}\n  text: {c['anchor_text']}" for c in candidates
    )
    prompt = (
        f"For the public company '{company_name}', classify each of the following PDF links "
        "from its investor relations site. Use the URL and anchor text to determine the kind "
        "(annual_report, interim_report, results_presentation, press_release, sustainability, "
        "governance, other) and the fiscal year.\n\n"
        "Mark as 'annual_report' ONLY if the document is the FULL annual report containing the "
        "complete audited consolidated financial statements (income statement, balance sheet, "
        "cash flow statement). Examples: 'Annual Report', 'Integrated Annual Report', "
        "'Annual Review', 'Universal Registration Document', '20-F', '10-K'.\n\n"
        "Do NOT mark as 'annual_report' (use 'other' or 'interim_report' instead) any of these "
        "partial / supplementary documents, even if the URL contains the words 'annual report':\n"
        "  - 'financial-performance-section', 'financial highlights', 'highlights brochure'\n"
        "  - 'Q1/Q2/Q3/Q4 financial statements', 'quarterly statements', 'X-month results'\n"
        "  - 'press release', 'fact sheet', 'shareholder letter' (unless it contains the full statements)\n"
        "  - 'sustainability report', 'ESG report', 'remuneration report'\n"
        "  - 'corporate governance', 'compensation report'\n"
        "  - 'investor presentation', 'capital markets day' deck\n"
        "  - 'annual report excerpt', 'annual report supplement'\n"
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


def _infer_year(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(20[0-3]\d|19[8-9]\d)\b", text)]
    if not years:
        return None
    currentish = [y for y in years if 1990 <= y <= 2035]
    return max(currentish) if currentish else None


def _heuristic_classify(candidates: list[dict]) -> list[dict]:
    """Best-effort local classification when an LLM chunk repeatedly fails."""
    enriched: list[dict] = []
    for c in candidates:
        text = f"{c.get('url', '')} {c.get('anchor_text', '')} {c.get('source_page', '')}"
        low = text.lower()
        kind = "other"
        if re.search(r"sustainability|esg|impact|responsibility", low):
            kind = "sustainability"
        elif re.search(r"governance|remuneration", low):
            kind = "governance"
        elif re.search(r"press[- ]release|media[- ]release", low):
            kind = "press_release"
        elif re.search(r"presentation|slides|deck", low):
            kind = "results_presentation"
        elif re.search(r"half[- ]year|interim|q[1-4]|quarter", low):
            kind = "interim_report"
        elif re.search(
            r"annual[- ]report|integrated[- ]report|annual[- ]review|universal[- ]registration|"
            r"registration[- ]document|full[- ]year|year[- ]end|shareholder[- ]letter",
            low,
        ):
            kind = "annual_report"

        enriched.append(
            {
                **c,
                "kind": kind,
                "fiscal_year": _infer_year(text),
                "language": "en" if re.search(r"\b(en|english)\b", low) else None,
                "title": c.get("anchor_text") or None,
            }
        )
    return enriched


def classify_pdfs(company_name: str, candidates: list[dict], on_progress: ProgressFn | None = None) -> list[dict]:
    """Classify candidate PDFs in small, retried chunks."""
    if not candidates:
        return []

    chunks = [
        candidates[i : i + CLASSIFY_CHUNK_SIZE]
        for i in range(0, len(candidates), CLASSIFY_CHUNK_SIZE)
    ]
    enriched: list[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        if on_progress:
            on_progress(
                "classifying_chunk",
                {"chunk": idx, "chunks": len(chunks), "count": len(chunk)},
            )

        last_error: Exception | None = None
        for attempt in range(1, CLASSIFY_RETRIES + 2):
            try:
                part = _classify_chunk(company_name, chunk)
                enriched.extend(part)
                if on_progress:
                    on_progress(
                        "classified_chunk",
                        {
                            "chunk": idx,
                            "chunks": len(chunks),
                            "classified_count": len(part),
                        },
                    )
                break
            except Exception as exc:
                last_error = exc
                if attempt <= CLASSIFY_RETRIES:
                    if on_progress:
                        on_progress(
                            "classify_retry",
                            {
                                "chunk": idx,
                                "chunks": len(chunks),
                                "attempt": attempt + 1,
                                "error": str(exc),
                            },
                        )
                    time.sleep(1.5 * attempt)
                else:
                    fallback = _heuristic_classify(chunk)
                    enriched.extend(fallback)
                    if on_progress:
                        on_progress(
                            "classify_fallback",
                            {
                                "chunk": idx,
                                "chunks": len(chunks),
                                "count": len(fallback),
                                "error": str(last_error) if last_error else "classification failed",
                            },
                        )
    return enriched


WEB_FALLBACK_SCHEMA = {
    "type": "object",
    "required": ["pdfs"],
    "properties": {
        "pdfs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["url", "fiscal_year"],
                "properties": {
                    "url": {"type": "string"},
                    "fiscal_year": {"type": "integer"},
                    "title": {"type": ["string", "null"]},
                    "language": {"type": ["string", "null"]},
                },
            },
        }
    },
}


def web_search_annual_reports(company_name: str, ir_url: str | None = None, max_years: int = 10) -> list[dict]:
    """Ask the LLM to find direct annual-report PDF URLs when the crawler fails."""
    site_hint = f" Their investor relations site is {ir_url}." if ir_url else ""
    prompt = (
        f"Find direct PDF download URLs for the most recent {max_years} annual reports of the public company "
        f"'{company_name}'.{site_hint} Each URL MUST point to the FULL annual report that contains the "
        "complete audited consolidated financial statements: income statement / statement of profit or loss, "
        "balance sheet / statement of financial position, AND cash flow statement. "
        "Acceptable document types: Annual Report, Integrated Annual Report, Annual Review, "
        "Universal Registration Document, Form 20-F, Form 10-K. "
        "Return one PDF per fiscal year, English version when available. "
        "Each url MUST be a direct .pdf link that downloads a file.\n\n"
        "Do NOT return ANY of these partial / supplementary documents, even if the URL contains "
        "'annual report':\n"
        "  - Financial highlights / performance-section excerpts\n"
        "  - Q1/Q2/Q3/Q4 quarterly financial statements\n"
        "  - Half-year / interim reports\n"
        "  - Sustainability-only or ESG-only reports\n"
        "  - Investor presentations, press releases, fact sheets\n"
        "  - Corporate governance / remuneration / compensation reports\n"
        "If the latest fiscal year only has a partial document published so far, SKIP that year rather "
        "than returning the partial document."
    )
    out = llm.web_search_json(
        prompt,
        WEB_FALLBACK_SCHEMA,
        cache_key="web_pdfs:v2:" + hashlib.sha256(f"{company_name}|{ir_url}|{max_years}".encode()).hexdigest(),
    )
    pdfs = out.get("pdfs", []) or []
    enriched: list[dict] = []
    for p in pdfs:
        url = (p.get("url") or "").strip()
        if not url.lower().endswith(".pdf") and ".pdf" not in url.lower():
            continue
        enriched.append({
            "url": url,
            "anchor_text": p.get("title") or "",
            "source_page": "web_search",
            "kind": "annual_report",
            "fiscal_year": p.get("fiscal_year"),
            "language": p.get("language"),
            "title": p.get("title"),
        })
    return enriched


def verify_pdf_url(url: str) -> bool:
    """HEAD/GET-probe a URL to confirm it returns an actual PDF."""
    try:
        with httpx.Client() as c:
            r = c.head(url, headers=HEADERS, follow_redirects=True, timeout=20)
            if r.status_code >= 400 or "pdf" not in (r.headers.get("content-type") or "").lower():
                # Fall back to a small ranged GET — many CDNs don't answer HEAD properly.
                r = c.get(url, headers={**HEADERS, "Range": "bytes=0-1024"}, follow_redirects=True, timeout=20)
                if r.status_code >= 400:
                    return False
                ctype = (r.headers.get("content-type") or "").lower()
                if "pdf" not in ctype and not r.content.startswith(b"%PDF"):
                    return False
        return True
    except Exception:
        return False


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
