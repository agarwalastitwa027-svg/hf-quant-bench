"""Load and normalize test cases from test_cases.yaml."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

DEFAULT_CASES_PATH = Path(__file__).parent / "test_cases.yaml"
DEFAULT_MAX_STEPS = 6

_FILLER_PARAGRAPH = (
    "The engineering team continued its routine review of system logs, noting "
    "steady throughput across the ingestion pipeline and no anomalies in the "
    "overnight batch jobs. Latency percentiles stayed within normal bounds, "
    "and the on-call rotation reported a quiet shift with no pages triggered. "
)


def _make_filler(word_target: int = 2000) -> str:
    words_per_para = len(_FILLER_PARAGRAPH.split())
    n_paras = max(1, word_target // words_per_para)
    return "\n".join(_FILLER_PARAGRAPH for _ in range(n_paras))


@dataclasses.dataclass
class ToolCallSpec:
    tool: str
    arguments: dict


@dataclasses.dataclass
class TestCase:
    id: str
    category: str
    prompt: str
    available_tools: list[str]
    expected_behavior: str  # "tool_call" | "no_tool" | "clarify"
    expected_trajectory: list[ToolCallSpec]
    judge_notes: str
    optimal_steps: int
    required_facts: list[str]
    max_steps: int = DEFAULT_MAX_STEPS


def load_cases(path: Path | None = None) -> list[TestCase]:
    path = path or DEFAULT_CASES_PATH
    raw = yaml.safe_load(path.read_text())
    filler = _make_filler()
    cases = []
    for entry in raw:
        # No default/fallback here on purpose: `optimal_steps` and
        # `required_facts` drive the rule-based path_optimization and
        # answer_completeness metrics (see metrics_rule.py). A case missing
        # either would otherwise silently score as "0 steps is optimal" or
        # "nothing is required to be a complete answer" -- both of which
        # inflate scores rather than reveal a broken test case. Fail loudly.
        for field in ("optimal_steps", "required_facts"):
            if field not in entry:
                raise KeyError(f"test case '{entry.get('id', '?')}' is missing required field '{field}'")

        prompt = entry["prompt"].replace("{{LONG_FILLER_2000_WORDS}}", filler)
        trajectory = [ToolCallSpec(tool=t["tool"], arguments=t["arguments"]) for t in entry.get("expected_trajectory", [])]
        cases.append(
            TestCase(
                id=entry["id"],
                category=entry["category"],
                prompt=prompt,
                available_tools=entry["available_tools"],
                expected_behavior=entry["expected_behavior"],
                expected_trajectory=trajectory,
                judge_notes=entry.get("judge_notes", ""),
                optimal_steps=entry["optimal_steps"],
                required_facts=entry["required_facts"],
                max_steps=entry.get("max_steps", DEFAULT_MAX_STEPS),
            )
        )
    return cases
