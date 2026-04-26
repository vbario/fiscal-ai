"""Resolve a free-text company name to ticker, exchange, country, and IR URL."""
from __future__ import annotations

from . import llm

SCHEMA = {
    "type": "object",
    "required": ["name", "ticker", "exchange", "country", "ir_url"],
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
        "notes": {"type": "string"},
    },
}


def resolve(name: str) -> dict:
    prompt = (
        f"Find the public European company '{name}'. "
        "Return its official name, primary ticker symbol, listing exchange, headquarters country, "
        "and the URL of the Investor Relations homepage. "
        "If there is a dedicated 'Annual Reports' or 'Financial Reports' / 'Results' page, "
        "include that URL as reports_url; otherwise null. "
        "Prefer the company's own corporate domain over third-party sites."
    )
    return llm.web_search_json(prompt, SCHEMA, cache_key=f"company:{name.lower().strip()}")
