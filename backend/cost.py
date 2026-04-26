"""Track cumulative OpenAI token usage and estimated USD cost across the process."""
from __future__ import annotations

import threading
from typing import Any

# USD per 1M tokens. Values can be overridden via env vars in config if needed.
# Defaults track public pricing for the gpt-5 family.
PRICING: dict[str, dict[str, float]] = {
    "gpt-5":           {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini":      {"input": 0.25, "cached_input": 0.025, "output":  2.00},
    "gpt-5-nano":      {"input": 0.05, "cached_input": 0.005, "output":  0.40},
    "gpt-4.1":         {"input": 2.00, "cached_input": 0.50,  "output":  8.00},
    "gpt-4.1-mini":    {"input": 0.40, "cached_input": 0.10,  "output":  1.60},
    "gpt-4o":          {"input": 2.50, "cached_input": 1.25,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15, "cached_input": 0.075, "output":  0.60},
}
DEFAULT_PRICING = PRICING["gpt-5"]


def _price_for(model: str) -> dict[str, float]:
    if model in PRICING:
        return PRICING[model]
    # Match by prefix (e.g. "gpt-5-2025-11-01" → "gpt-5").
    for key in sorted(PRICING.keys(), key=len, reverse=True):
        if model.startswith(key):
            return PRICING[key]
    return DEFAULT_PRICING


_lock = threading.Lock()
_listeners: list[Any] = []
_state: dict[str, Any] = {
    "input_tokens": 0,
    "cached_input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "calls": 0,
    "usd": 0.0,
    "by_model": {},  # model -> {input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, calls, usd}
}


def _zero_model() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "calls": 0,
        "usd": 0.0,
    }


def record(model: str, usage: Any) -> None:
    """Record a single API call's token usage. `usage` is the OpenAI Responses `resp.usage` object."""
    if usage is None:
        return
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cached = 0
    in_details = getattr(usage, "input_tokens_details", None)
    if in_details is not None:
        cached = int(getattr(in_details, "cached_tokens", 0) or 0)
    reasoning = 0
    out_details = getattr(usage, "output_tokens_details", None)
    if out_details is not None:
        reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)

    fresh_input = max(input_tokens - cached, 0)
    p = _price_for(model)
    cost = (
        fresh_input * p["input"]
        + cached * p.get("cached_input", p["input"])
        + output_tokens * p["output"]
    ) / 1_000_000.0

    with _lock:
        _state["input_tokens"] += input_tokens
        _state["cached_input_tokens"] += cached
        _state["output_tokens"] += output_tokens
        _state["reasoning_tokens"] += reasoning
        _state["calls"] += 1
        _state["usd"] += cost
        m = _state["by_model"].setdefault(model, _zero_model())
        m["input_tokens"] += input_tokens
        m["cached_input_tokens"] += cached
        m["output_tokens"] += output_tokens
        m["reasoning_tokens"] += reasoning
        m["calls"] += 1
        m["usd"] += cost
        listeners = list(_listeners)
    for fn in listeners:
        try:
            fn(model, cost)
        except Exception:
            pass


def subscribe(callback) -> None:
    with _lock:
        _listeners.append(callback)


def unsubscribe(callback) -> None:
    with _lock:
        try:
            _listeners.remove(callback)
        except ValueError:
            pass


def snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "input_tokens": _state["input_tokens"],
            "cached_input_tokens": _state["cached_input_tokens"],
            "output_tokens": _state["output_tokens"],
            "reasoning_tokens": _state["reasoning_tokens"],
            "calls": _state["calls"],
            "usd": round(_state["usd"], 6),
            "by_model": {
                k: {**v, "usd": round(v["usd"], 6)}
                for k, v in _state["by_model"].items()
            },
        }


def reset() -> None:
    with _lock:
        _state["input_tokens"] = 0
        _state["cached_input_tokens"] = 0
        _state["output_tokens"] = 0
        _state["reasoning_tokens"] = 0
        _state["calls"] = 0
        _state["usd"] = 0.0
        _state["by_model"] = {}
