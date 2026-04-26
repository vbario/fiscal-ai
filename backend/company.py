"""Resolve a free-text company name to ticker, exchange, country, and IR URL."""
from __future__ import annotations

from . import llm

SCHEMA = {
    "type": "object",
    "required": ["name", "ticker", "exchange", "country", "ir_url", "is_public"],
    "properties": {
        "name": {"type": "string", "description": "Official legal company name"},
        "ticker": {"type": "string", "description": "Primary equity ticker symbol"},
        "exchange": {"type": "string", "description": "Primary listing exchange (e.g. Euronext Amsterdam)"},
        "country": {"type": "string", "description": "ISO country of headquarters"},
        "ir_url": {"type": "string", "description": "URL to the investor relations homepage"},
        "reports_url": {
            "type": ["string", "null"],
            "description": "If a separate 'Annual reports' or 'Financial reports' page exists, its URL; else null",
        },
        "is_public": {
            "type": "boolean",
            "description": "True if the company currently has equity listed on a public stock exchange.",
        },
        "ipo_year": {
            "type": ["integer", "null"],
            "description": (
                "Calendar year the company first became publicly listed on its primary exchange "
                "(IPO year, or year of direct/SPAC listing). Null if not public or unknown."
            ),
        },
        "ipo_date": {
            "type": ["string", "null"],
            "description": "Listing date in YYYY-MM-DD if known; otherwise null.",
        },
        "public_status_note": {
            "type": ["string", "null"],
            "description": (
                "One-sentence explanation of the listing history when relevant "
                "(e.g. 'Spun off from X and listed in 2018', 'Re-listed in 2020 after going private')."
            ),
        },
        "notes": {"type": "string"},
    },
}


def resolve(name: str) -> dict:
    prompt = (
        f"Find the European company '{name}'. "
        "Return its official name, primary ticker symbol, listing exchange, headquarters country, "
        "and the URL of the Investor Relations homepage. "
        "If there is a dedicated 'Annual Reports' or 'Financial Reports' / 'Results' page, "
        "include that URL as reports_url; otherwise null. "
        "Prefer the company's own corporate domain over third-party sites. "
        "Also determine whether the company is currently publicly listed (is_public). "
        "If yes, return ipo_year (the calendar year the company first became publicly listed on its primary "
        "exchange — IPO, direct listing, SPAC, or relisting after going private), and ipo_date (YYYY-MM-DD) "
        "if known. If the listing history is unusual (spin-off, relisting, change of primary exchange), "
        "summarise it in public_status_note."
    )
    return llm.web_search_json(prompt, SCHEMA, cache_key=f"company:v2:{name.lower().strip()}")
