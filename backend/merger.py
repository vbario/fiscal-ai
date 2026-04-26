"""Merge per-year extractions into a single 10-year clean table per statement.

Strategy:
- The MOST RECENT report's line items + ordering are the canonical schema.
- For each older report's line item, ask the LLM to map it to the canonical label
  (or mark it as 'new' — meaning it should be appended).
- For each (canonical_label, period_end) cell, prefer the value from the most
  recent report that contains that period (latest restatement wins).
- Convert all values to a single unit (millions) per company so columns are comparable.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from rapidfuzz import fuzz

from . import llm

UNIT_FACTOR = {
    "actual": 1.0,
    "thousands": 1e-3,        # → millions
    "millions": 1.0,
    "billions": 1000.0,
}


def _to_millions(value: Optional[float], unit: str) -> Optional[float]:
    if value is None:
        return None
    return value * UNIT_FACTOR.get(unit, 1.0)


def _canonical_from_latest(latest: dict) -> list[dict]:
    """Build canonical line-item list from the latest report (preserve ordering)."""
    items = []
    for li in latest.get("line_items", []):
        items.append(
            {
                "label": li["label"],
                "indent_level": li.get("indent_level", 0),
                "is_subtotal": li.get("is_subtotal", False),
                "is_section_header": li.get("is_section_header", False),
            }
        )
    return items


MAP_SCHEMA = {
    "type": "object",
    "required": ["mapping"],
    "properties": {
        "mapping": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_label", "canonical_label"],
                "properties": {
                    "source_label": {"type": "string"},
                    "canonical_label": {
                        "type": ["string", "null"],
                        "description": "An exact label from the canonical list, or null if no good match (will be appended as new).",
                    },
                },
            },
        }
    },
}


def _map_labels(
    statement_type: str,
    canonical_labels: list[str],
    source_labels: list[str],
    company: str,
    year: int,
) -> dict[str, Optional[str]]:
    """Map labels from an older report to canonical labels via fuzzy + LLM."""
    if not source_labels:
        return {}

    # Pre-pass: exact / very-high-similarity matches don't need the LLM.
    mapping: dict[str, Optional[str]] = {}
    needs_llm: list[str] = []
    canon_lower = {c.lower(): c for c in canonical_labels}
    for s in source_labels:
        if s in canon_lower.values() or s.lower() in canon_lower:
            mapping[s] = canon_lower.get(s.lower(), s)
            continue
        # rapidfuzz best match
        best = max(
            ((c, fuzz.token_sort_ratio(s, c)) for c in canonical_labels),
            key=lambda x: x[1],
            default=(None, 0),
        )
        if best[0] and best[1] >= 92:
            mapping[s] = best[0]
        else:
            needs_llm.append(s)

    if not needs_llm:
        return mapping

    cache_key = "labelmap:" + hashlib.sha256(
        (company + "|" + statement_type + "|" + str(year) + "|" +
         "\n".join(sorted(canonical_labels)) + "||" + "\n".join(sorted(needs_llm))).encode()
    ).hexdigest()

    prompt = (
        f"You are normalizing line items from a {company} annual report ({year}, {statement_type}).\n"
        "Below is the CANONICAL list of line items from the company's most recent report (preserve these labels exactly).\n"
        "Map each SOURCE label to the canonical label that refers to the same accounting concept.\n"
        "If no canonical label refers to the same concept, return null (the item will be appended).\n"
        "Be conservative: only map when the underlying concept is clearly the same.\n\n"
        f"CANONICAL:\n" + "\n".join(f"- {c}" for c in canonical_labels) + "\n\n"
        f"SOURCE:\n" + "\n".join(f"- {s}" for s in needs_llm)
    )
    out = llm.text_json(prompt, MAP_SCHEMA, cache_key=cache_key)
    for m in out.get("mapping", []):
        mapping[m["source_label"]] = m.get("canonical_label")
    # Anything still unmapped → null
    for s in needs_llm:
        mapping.setdefault(s, None)
    return mapping


def merge_statement(
    statement_type: str,
    extractions_by_year: dict[int, dict],
    company: str,
) -> dict:
    """Merge a list of single-report extractions for one statement type."""
    if not extractions_by_year:
        return {"statement_type": statement_type, "currency": None, "periods": [], "rows": []}

    years_desc = sorted(extractions_by_year.keys(), reverse=True)
    latest = extractions_by_year[years_desc[0]]
    canonical = _canonical_from_latest(latest)
    canonical_labels = [c["label"] for c in canonical]
    canonical_index = {c["label"]: i for i, c in enumerate(canonical)}

    # Pick currency from latest (assume all reports same currency).
    currency = latest.get("currency") or ""

    # Build the union of all period_ends across reports.
    all_periods: dict[str, dict] = {}  # period_end -> {label, period_end, duration}
    # Track which year-source provided each (period_end) so we can prefer the latest restatement.
    period_provenance: dict[str, int] = {}
    for year in years_desc:
        ext = extractions_by_year[year]
        for p in ext.get("periods", []):
            pe = p.get("period_end")
            if not pe:
                continue
            if pe not in all_periods:
                all_periods[pe] = p
            # latest year wins for the period label (they're iterating newest-first)
            if pe not in period_provenance:
                period_provenance[pe] = year

    # Prepare a value matrix: (canonical_label, period_end) -> (value_in_millions, source_year)
    cell: dict[tuple[str, str], tuple[Optional[float], int]] = {}

    for year in years_desc:  # newest first → first write wins per cell
        ext = extractions_by_year[year]
        unit = ext.get("unit", "actual")
        period_ends = [p.get("period_end") for p in ext.get("periods", [])]
        source_labels = [li["label"] for li in ext.get("line_items", [])]
        label_map = _map_labels(statement_type, canonical_labels, source_labels, company, year)

        for li in ext.get("line_items", []):
            src_label = li["label"]
            mapped = label_map.get(src_label)
            if mapped is None:
                # Append as new canonical row (insert near the end if not already present)
                if src_label not in canonical_index:
                    canonical.append(
                        {
                            "label": src_label,
                            "indent_level": li.get("indent_level", 0),
                            "is_subtotal": li.get("is_subtotal", False),
                            "is_section_header": li.get("is_section_header", False),
                        }
                    )
                    canonical_index[src_label] = len(canonical) - 1
                    canonical_labels.append(src_label)
                mapped = src_label
            for value, pe in zip(li.get("values", []), period_ends):
                if not pe:
                    continue
                key = (mapped, pe)
                if key in cell:
                    continue  # newer report already provided this cell
                cell[key] = (_to_millions(value, unit), year)

    period_ends_sorted = sorted(all_periods.keys(), reverse=True)[:10]  # 10 most recent
    periods_out = [
        {
            **all_periods[pe],
            "source_year": period_provenance[pe],
        }
        for pe in period_ends_sorted
    ]

    rows = []
    for c in canonical:
        rows.append(
            {
                "label": c["label"],
                "indent_level": c.get("indent_level", 0),
                "is_subtotal": c.get("is_subtotal", False),
                "is_section_header": c.get("is_section_header", False),
                "values": [cell.get((c["label"], pe), (None, None))[0] for pe in period_ends_sorted],
            }
        )

    return {
        "statement_type": statement_type,
        "currency": currency,
        "unit": "millions",
        "periods": periods_out,
        "rows": rows,
    }
