"""Extract Income Statement / Balance Sheet / Cash Flow from a PDF annual report."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from . import cache, llm
from .config import CACHE_DIR

# Multilingual headings for the three primary statements.
HEADINGS = {
    "income_statement": [
        r"consolidated\s+(income\s+statement|statement\s+of\s+(?:profit\s+or\s+loss|operations|comprehensive\s+income|income))",
        r"income\s+statement",
        r"statement\s+of\s+profit\s+or\s+loss",
        r"statement\s+of\s+operations",
        r"profit\s+and\s+loss\s+account",
        # Dutch / French / German common variants
        r"geconsolideerde\s+winst[-\s]en[-\s]verliesrekening",
        r"compte\s+de\s+r[ée]sultat\s+consolid[ée]",
        r"konzern[\s-]gewinn[\s-]und[\s-]verlustrechnung",
    ],
    "balance_sheet": [
        r"consolidated\s+(balance\s+sheet|statement\s+of\s+financial\s+position)",
        r"balance\s+sheet",
        r"statement\s+of\s+financial\s+position",
        r"geconsolideerde\s+balans",
        r"bilan\s+consolid[ée]",
        r"konzernbilanz",
    ],
    "cash_flow": [
        r"consolidated\s+(cash\s+flow\s+statement|statement\s+of\s+cash\s+flows)",
        r"cash\s+flow\s+statement",
        r"statement\s+of\s+cash\s+flows",
        r"geconsolideerd\s+kasstroomoverzicht",
        r"tableau\s+(des|de)\s+flux\s+de\s+tr[ée]sorerie",
        r"konzern[\s-]kapitalflussrechnung",
    ],
}

STATEMENT_LABELS = {
    "income_statement": "Consolidated Income Statement",
    "balance_sheet": "Consolidated Balance Sheet",
    "cash_flow": "Consolidated Cash Flow Statement",
}

# Pages of context to render around the heading match.
PAGES_PER_STATEMENT = 3
RENDER_DPI = 180


def _statement_page_score(statement_type: str, text_norm: str) -> int:
    """Prefer primary statement pages over TOCs and note references."""
    head = text_norm[:700]
    score = 0
    if statement_type == "income_statement":
        # The Profit & Loss / Income Statement / Statement of Operations is the
        # primary income statement. "Statement of Comprehensive Income" is a
        # supplementary page that often appears immediately after — it starts
        # with the bottom-line "Net income" and adds OCI items. Score the
        # primary statement higher so we don't accidentally pick the OCI page.
        if re.search(r"consolidated\s+(income\s+statement|statement\s+of\s+(profit\s+or\s+loss|operations))", head, re.I):
            score += 130
        if re.search(r"consolidated\s+statement\s+of\s+comprehensive\s+income", head, re.I):
            score += 50
        # Revenue / sales / cost-of-sales anchors are a strong signal of the
        # primary P&L page (vs. the OCI page which leads with "Net income").
        if re.search(r"\b(net\s+sales|total\s+revenue|total\s+net\s+sales|cost\s+of\s+(sales|goods\s+sold|system\s+sales)|gross\s+profit|operating\s+(income|profit))\b", head, re.I):
            score += 40
        if re.search(r"for\s+the\s+years?\s+ended|net\s+income|income\s+before", head, re.I):
            score += 25
        # If the page's first ~250 chars start with "Net income" without any
        # revenue-line preamble, it's almost certainly the OCI page.
        if re.match(r"[^a-z]*consolidated\s+statement\s+of\s+comprehensive\s+income.*?net\s+income", head[:400], re.I | re.S):
            score -= 60
    elif statement_type == "balance_sheet":
        if re.search(r"consolidated\s+(balance\s+sheet|statement\s+of\s+financial\s+position)", head, re.I):
            score += 90
        if re.search(r"as\s+at|as\s+of|total\s+assets|total\s+liabilities", head, re.I):
            score += 25
    elif statement_type == "cash_flow":
        if re.search(r"consolidated\s+(statement\s+of\s+cash\s+flows|cash\s+flow\s+statement)", head, re.I):
            score += 90
        if re.search(r"for\s+the\s+years?\s+ended|operating\s+activities|investing\s+activities|financing\s+activities", head, re.I):
            score += 25

    if re.search(r"all\s+amounts\s+(are\s+)?in|unless\s+otherwise\s+stated", head, re.I):
        score += 10
    if re.search(r"table\s+of\s+contents|contents", head, re.I):
        score -= 100
    # TOC pages often mention several primary statements and many note titles.
    statement_mentions = len(re.findall(r"Consolidated\s+Statement|Consolidated\s+Balance\s+Sheet|Company\s+Balance\s+Sheet", text_norm, re.I))
    note_mentions = len(re.findall(r"\b\d{1,2}\.\s+[A-Z]", text_norm))
    if statement_mentions >= 3 or note_mentions >= 6:
        score -= 70
    return score


def _file_id(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def find_statement_pages(pdf_path: Path) -> dict[str, list[int]]:
    """Scan PDF text and return a list of 0-based page indexes per statement type."""
    hits: dict[str, list[int]] = {k: [] for k in HEADINGS}
    compiled = {
        k: [re.compile(p, re.I | re.S) for p in patterns]
        for k, patterns in HEADINGS.items()
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            text_norm = re.sub(r"\s+", " ", text)
            for stmt, regs in compiled.items():
                if any(r.search(text_norm) for r in regs):
                    # Avoid Index/contents pages: skip if "table of contents" nearby
                    # or if page has very little text.
                    if len(text_norm) < 200:
                        continue
                    if re.search(r"table of contents|contents", text_norm[:300], re.I):
                        continue
                    hits[stmt].append(i)
    # Heuristic: keep the best-scoring primary statement page, avoiding TOCs and
    # later note references that mention statement names.
    cleaned: dict[str, list[int]] = {}
    for stmt, pages in hits.items():
        if not pages:
            cleaned[stmt] = []
            continue
        scored: list[tuple[int, int]] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx in pages:
                try:
                    text = pdf.pages[page_idx].extract_text() or ""
                except Exception:
                    text = ""
                text_norm = re.sub(r"\s+", " ", text)
                scored.append((_statement_page_score(stmt, text_norm), page_idx))
        scored.sort(key=lambda x: (-x[0], x[1]))
        anchor = scored[0][1]
        cleaned[stmt] = sorted(set(range(anchor, min(anchor + PAGES_PER_STATEMENT, anchor + 8))))
    return cleaned


def render_pages(pdf_path: Path, page_indexes: list[int]) -> list[Path]:
    """Render given pages to PNGs in CACHE_DIR; return their paths."""
    if not page_indexes:
        return []
    fid = _file_id(pdf_path)
    out: list[Path] = []
    doc = fitz.open(str(pdf_path))
    try:
        for idx in page_indexes:
            if idx >= len(doc):
                continue
            png_path = CACHE_DIR / f"{fid}_p{idx:04d}.png"
            if not png_path.exists():
                page = doc.load_page(idx)
                pix = page.get_pixmap(dpi=RENDER_DPI)
                pix.save(str(png_path))
            out.append(png_path)
    finally:
        doc.close()
    return out


EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["statement_type", "currency", "unit", "periods", "line_items"],
    "properties": {
        "statement_type": {
            "type": "string",
            "enum": ["income_statement", "balance_sheet", "cash_flow"],
        },
        "currency": {"type": "string", "description": "ISO 4217 e.g. EUR, USD"},
        "unit": {
            "type": "string",
            "enum": ["actual", "thousands", "millions", "billions"],
            "description": "Reporting unit for the figures",
        },
        "periods": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "period_end", "duration"],
                "properties": {
                    "label": {"type": "string", "description": "As shown in the report header (e.g. 'FY2024', '31 Dec 2024')"},
                    "period_end": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "duration": {
                        "type": "string",
                        "enum": ["12_months", "6_months", "3_months", "point_in_time", "other"],
                    },
                },
            },
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "values"],
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Verbatim line-item label as printed in the statement.",
                    },
                    "indent_level": {
                        "type": "integer",
                        "description": "0 for top-level / sub-headers, 1 for indented detail, 2 for further indent.",
                    },
                    "is_subtotal": {"type": "boolean"},
                    "is_section_header": {"type": "boolean"},
                    "values": {
                        "type": "array",
                        "description": "One value per period, in the same order as the 'periods' array. Use null for blanks/dashes.",
                        "items": {"type": ["number", "null"]},
                    },
                },
            },
        },
    },
}


def extract_statement(
    pdf_path: Path, statement_type: str, page_indexes: list[int]
) -> dict | None:
    """Render pages and ask the vision model to extract the statement."""
    if not page_indexes:
        return None
    images = render_pages(pdf_path, page_indexes)
    if not images:
        return None

    fid = _file_id(pdf_path)
    cache_key = f"extract:{fid}:{statement_type}:{','.join(map(str, page_indexes))}"

    label = STATEMENT_LABELS[statement_type]
    instructions = (
        f"You are extracting the {label} from an annual report. "
        "Extract every line item EXACTLY as printed (preserve wording, casing, ordering). "
        "Include section headers and subtotals. "
        "For each line item, return one numeric value per period column shown — "
        "use the same order as the report (typically most-recent year first). "
        "If a cell is blank or shown as '—', use null. "
        "Convert parentheses-negatives like (1,234) to -1234. "
        "Do NOT scale: report numbers exactly as they appear in the table; "
        "specify the global unit (millions/thousands/etc) once via the 'unit' field. "
        f"statement_type MUST be: {statement_type}. "
        "If the page does NOT contain the requested statement, return an empty line_items array."
    )
    try:
        return llm.vision_json(instructions, [str(p) for p in images], EXTRACTION_SCHEMA, cache_key=cache_key)
    except Exception as e:
        return {"error": str(e), "statement_type": statement_type, "line_items": [], "periods": [], "currency": "", "unit": "actual"}


def extract_all(pdf_path: Path) -> dict[str, dict | None]:
    pages = find_statement_pages(pdf_path)
    out: dict[str, dict | None] = {}
    for stmt in HEADINGS:
        out[stmt] = extract_statement(pdf_path, stmt, pages[stmt])
    return out
