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
    # Heuristic: keep the FIRST cluster of hits per statement (the actual primary
    # statement near the front of the financial-statements section, not later notes).
    cleaned: dict[str, list[int]] = {}
    for stmt, pages in hits.items():
        if not pages:
            cleaned[stmt] = []
            continue
        pages.sort()
        anchor = pages[0]
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
