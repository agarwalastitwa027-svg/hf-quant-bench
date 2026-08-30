"""Mock tool registry for the agentic benchmark.

Every tool has a real JSON Schema (OpenAI/Anthropic-style function schema) and
a deterministic fake implementation — same input always produces the same
output, so trajectories are reproducible across formats and across runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema "parameters" object
    fn: Callable[[dict], Any]


def _seeded_float(*parts: str, lo: float, hi: float) -> float:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return round(lo + frac * (hi - lo), 1)


def _get_weather(args: dict) -> dict:
    location = str(args.get("location", ""))
    unit = args.get("unit", "celsius")
    temp_c = _seeded_float("weather", location, lo=-5.0, hi=35.0)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32, 1)
    conditions = ["clear", "cloudy", "rainy", "windy", "snowy"]
    idx = int(hashlib.sha256(("cond|" + location).encode()).hexdigest(), 16) % len(conditions)
    return {"location": location, "temperature": temp, "unit": unit, "conditions": conditions[idx]}


def _get_calendar_events(args: dict) -> dict:
    date = str(args.get("date", ""))
    h = hashlib.sha256(("cal|" + date).encode()).hexdigest()
    n_events = int(h[:2], 16) % 3
    events = [
        {"title": f"Event {i+1} on {date}", "start": f"{9 + i * 2:02d}:00", "end": f"{10 + i * 2:02d}:00"}
        for i in range(n_events)
    ]
    return {"date": date, "events": events}


# Canned answers for queries used in chained (tool-A-output-feeds-tool-B)
# test cases, so the "correct" second tool call is well-defined and
# deterministic rather than left to the model to hallucinate coherently.
# Also doubles as the source of ground truth for `required_facts` in the
# rule-based answer_completeness metric (see bench/metrics_rule.py) --
# every web_search case in test_cases.yaml uses one of these queries
# specifically so there's a real, human-meaningful fact to check for,
# instead of a random hash fragment from the fallback path below.
_CANNED_SEARCH: dict[str, str] = {
    "capital of france": "The capital of France is Paris, a major European city.",
    "capital of japan": "The capital of Japan is Tokyo, its largest city.",
    "largest ocean": "The largest ocean on Earth is the Pacific Ocean, bordering many coastal cities including Tokyo.",
    "capital of germany": "The capital of Germany is Berlin, its largest city.",
    "capital of egypt": "The capital of Egypt is Cairo, a major city on the Nile.",
    "tallest mountain in the world": "The tallest mountain in the world is Mount Everest, standing at 8,849 meters.",
}


def _web_search(args: dict) -> dict:
    query = str(args.get("query", ""))
    canned = _CANNED_SEARCH.get(query.strip().lower())
    if canned:
        return {"query": query, "results": [{"title": f"Result for '{query}'", "snippet": canned}]}
    h = hashlib.sha256(("search|" + query).encode()).hexdigest()
    results = [
        {"title": f"Result {i+1} for '{query}'", "snippet": f"Deterministic fake snippet {h[i*4:i*4+8]} about {query}."}
        for i in range(3)
    ]
    return {"query": query, "results": results}


def _calculator(args: dict) -> dict:
    expression = str(args.get("expression", ""))
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return {"expression": expression, "error": "invalid characters in expression"}
    try:
        # eval is safe here: input is restricted to digits/operators/parens/spaces above.
        value = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        return {"expression": expression, "error": str(e)}
    return {"expression": expression, "result": value}


def _read_file(args: dict) -> dict:
    path = str(args.get("path", ""))
    fake_fs = {
        "/notes/todo.txt": "1. Buy milk\n2. Finish quant benchmark\n3. Call Alex",
        "/reports/q3.txt": "Q3 revenue: $482,000. Growth: 12% QoQ.",
        "/config/settings.json": json.dumps({"theme": "dark", "autosave": True}),
        "/reports/q4.txt": "Q4 revenue: $510,000. Growth: 5.8% QoQ.",
        "/logs/error.log": "2026-08-14 ERROR db_timeout retries=3\n2026-08-14 ERROR db_timeout retries=4\n2026-08-15 INFO recovered",
        "/config/limits.json": json.dumps({"max_users": 200, "rate_limit_per_min": 60}),
        "/notes/ideas.txt": "1. Add dark mode\n2. Ship the CLI\n3. Write docs",
    }
    if path not in fake_fs:
        return {"path": path, "error": "file not found"}
    return {"path": path, "content": fake_fs[path]}


TOOLS: dict[str, ToolSpec] = {
    "get_weather": ToolSpec(
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
        },
        fn=_get_weather,
    ),
    "get_calendar_events": ToolSpec(
        name="get_calendar_events",
        description="List calendar events on a given date.",
        parameters={
            "type": "object",
            "properties": {"date": {"type": "string", "description": "ISO date, e.g. '2026-09-01'"}},
            "required": ["date"],
        },
        fn=_get_calendar_events,
    ),
    "web_search": ToolSpec(
        name="web_search",
        description="Search the web and return top results.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        fn=_web_search,
    ),
    "calculator": ToolSpec(
        name="calculator",
        description="Evaluate a basic arithmetic expression.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '(3 + 4) * 2'"}},
            "required": ["expression"],
        },
        fn=_calculator,
    ),
    "read_file": ToolSpec(
        name="read_file",
        description="Read the contents of a file by path.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        fn=_read_file,
    ),
    # Deliberately irrelevant / distractor tool for spurious-call test cases.
    "send_email": ToolSpec(
        name="send_email",
        description="Send an email to a recipient. NOT relevant to informational queries.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
        fn=lambda args: {"status": "sent", "to": args.get("to")},
    ),
}


def schemas_for(names: list[str]) -> list[dict]:
    """Return OpenAI/Anthropic-style function schemas for the given tool names."""
    out = []
    for n in names:
        t = TOOLS[n]
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
        )
    return out


def call_tool(name: str, arguments: dict) -> dict | None:
    """Returns None (not KeyError) if the tool name doesn't exist, so the
    caller can record it as a hallucinated tool call rather than crashing.
    """
    spec = TOOLS.get(name)
    if spec is None:
        return None
    return spec.fn(arguments)
