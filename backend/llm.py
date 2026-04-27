"""OpenAI helpers: web-search resolver + structured JSON extraction with vision."""
from __future__ import annotations

import base64
import contextvars
import json
import threading
from typing import Any, Optional

from openai import OpenAI

from . import cache, cost
from .config import MODEL, OPENAI_API_KEY, VISION_MODEL

# Per-context overrides set by the API layer when the user supplies their own
# key / preferred model in the UI. Falls back to env defaults when unset.
_api_key_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "fiscal_ai_api_key", default=None
)
_model_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "fiscal_ai_model", default=None
)

_client_cache: dict[str, OpenAI] = {}
_client_lock = threading.Lock()


def set_overrides(api_key: Optional[str] = None, model: Optional[str] = None) -> None:
    """Bind API key / model for the current contextvars context."""
    if api_key is not None:
        _api_key_var.set(api_key.strip() or None)
    if model is not None:
        _model_var.set(model.strip() or None)


def _resolved_api_key() -> str:
    key = _api_key_var.get() or OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "No OpenAI API key configured. Set OPENAI_API_KEY in .env or "
            "provide one in the app's Settings panel."
        )
    return key


def _resolved_model() -> str:
    return _model_var.get() or MODEL


def _resolved_vision_model() -> str:
    # When the user picks a model in the UI we use it for vision too — the
    # current set of allowed models all support vision.
    return _model_var.get() or VISION_MODEL


def _client() -> OpenAI:
    key = _resolved_api_key()
    with _client_lock:
        c = _client_cache.get(key)
        if c is None:
            c = OpenAI(api_key=key, timeout=90.0, max_retries=5)
            _client_cache[key] = c
        return c


def web_search_json(prompt: str, schema: dict, cache_key: Optional[str] = None) -> dict:
    """Run a web-search-grounded query and return JSON matching the schema."""
    if cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    _model = _resolved_model()
    resp = _client().responses.create(
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

    _model = _resolved_vision_model()
    resp = _client().responses.create(
        model=_model,
        input=[{"role": "user", "content": content}],
    )
    cost.record(_model, getattr(resp, "usage", None))
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

    _model = _resolved_model()
    resp = _client().responses.create(
        model=_model,
        input=[
            {"role": "system", "content": "Return ONLY a JSON object matching the requested schema."},
            {
                "role": "user",
                "content": prompt + "\n\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(schema),
            },
        ],
    )
    cost.record(_model, getattr(resp, "usage", None))
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
