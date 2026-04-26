"""OpenAI helpers: web-search resolver + structured JSON extraction with vision."""
from __future__ import annotations

import base64
import json
from typing import Any, Optional

from openai import OpenAI

from . import cache, cost
from .config import MODEL, OPENAI_API_KEY, VISION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY, timeout=180.0, max_retries=2)


def web_search_json(prompt: str, schema: dict, cache_key: Optional[str] = None) -> dict:
    """Run a web-search-grounded query and return JSON matching the schema."""
    if cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    _model = MODEL
    resp = client.responses.create(
        model=_model,
        tools=[{"type": "web_search"}],
        input=[
            {
                "role": "system",
                "content": (
                    "You are a precise financial-data research assistant. "
                    "Always return valid JSON matching the requested schema. "
                    "Use the web_search tool to find authoritative sources "
                    "(company IR sites, exchanges, regulators)."
                ),
            },
            {
                "role": "user",
                "content": prompt + "\n\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(schema),
            },
        ],
    )
    cost.record(_model, getattr(resp, "usage", None))
    text = resp.output_text.strip()
    data = _parse_json(text)
    if cache_key:
        cache.put(cache_key, data)
    return data


def vision_json(
    instructions: str,
    image_paths: list[str],
    schema: dict,
    cache_key: Optional[str] = None,
) -> dict:
    """Send page images to a vision model and return JSON."""
    if cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    content: list[dict[str, Any]] = [{"type": "input_text", "text": instructions + "\n\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(schema)}]
    for path in image_paths:
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            }
        )

    resp = client.responses.create(
        model=VISION_MODEL,
        input=[{"role": "user", "content": content}],
    )
    cost.record(VISION_MODEL, getattr(resp, "usage", None))
    text = resp.output_text.strip()
    data = _parse_json(text)
    if cache_key:
        cache.put(cache_key, data)
    return data


def text_json(prompt: str, schema: dict, cache_key: Optional[str] = None) -> dict:
    """Pure-text JSON call (no web, no images)."""
    if cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": "Return ONLY a JSON object matching the requested schema."},
            {
                "role": "user",
                "content": prompt + "\n\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(schema),
            },
        ],
    )
    cost.record(MODEL, getattr(resp, "usage", None))
    data = _parse_json(resp.output_text.strip())
    if cache_key:
        cache.put(cache_key, data)
    return data


def _parse_json(text: str) -> dict:
    # Strip code fences if model wrapped output.
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        # Drop any leading 'json' language hint.
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    # If extra prose surrounds the JSON, find the outermost braces.
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1:
        t = t[start : end + 1]
    return json.loads(t)
