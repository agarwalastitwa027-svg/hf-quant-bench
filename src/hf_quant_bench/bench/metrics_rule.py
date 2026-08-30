"""Rule-based metrics: tool correctness, argument correctness, hallucination
rate, path optimization, answer completeness. All five are deterministic and
hand-rolled (no library, no model of any kind) -- there is no LLM-as-a-judge
anywhere in the primary metrics. path_optimization and answer_completeness
replaced an earlier Anthropic-API-based judge entirely: this project's
author has no cloud LLM API key and isn't getting one, and an approximate
judge scoring things this harness already knows exactly (the minimal step
count; the literal values the deterministic mock tools returned) would be
strictly worse than just computing them. See bench/local_judge.py for the
optional, off-by-default, offline local-model supplementary scoring.
"""

from __future__ import annotations

import dataclasses
import re

from hf_quant_bench.bench.cases import TestCase
from hf_quant_bench.bench.runner import Trajectory

_ARITH_RE = re.compile(r"^[0-9+\-*/(). ]+$")

# Matches an optionally-signed, optionally comma-grouped, optionally
# decimal number -- used both to detect whether a required_fact is itself
# numeric, and to scan an answer's free text for numeric tokens to compare
# against a numeric fact (see _fact_matches).
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _normalize_numeric(s: str) -> float | None:
    s = s.replace(",", "").strip().rstrip("°CF%")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fact_matches(fact: str, answer_text: str) -> bool:
    """Substring match for text facts, tolerance-based numeric match for
    numeric facts. No embeddings, no semantic similarity, no model of any
    kind -- see module docstring.

    Numeric tolerance is deliberately generous (>= 1.0 absolute, or 2%
    relative for larger magnitudes): natural-language answers routinely
    round a temperature or a dollar figure for readability ("about -4
    degrees" for a true value of -4.3), and that rounding is not the failure
    this metric exists to catch -- stating an unrelated number is. See the
    project writeup for where this tolerance could still produce a false
    negative (an answer that rounds MORE aggressively than this, or restates
    a fact in prose with no digits at all, e.g. "it's freezing" instead of
    a temperature).
    """
    fact_num = _normalize_numeric(fact)
    if fact_num is not None:
        tol = max(1.0, abs(fact_num) * 0.02)
        for m in _NUMBER_RE.finditer(answer_text):
            v = _normalize_numeric(m.group())
            if v is not None and abs(v - fact_num) <= tol:
                return True
        return False

    norm_answer = " ".join(answer_text.lower().split())
    norm_fact = " ".join(fact.lower().split())
    return norm_fact in norm_answer


def compute_path_optimization(case: TestCase, traj: Trajectory) -> tuple[float, bool]:
    """Returns (path_optimization, timed_out).

    timed_out is reported separately (see `timeout_rate` in results.csv) so
    a trajectory that never terminated can't be silently averaged in as
    merely "inefficient" -- those are different failure modes, and
    conflating them would hide a hung/looping model behind a low-but-
    nonzero-looking score.
    """
    if traj.ended_reason == "step_cap":
        return 0.0, True
    actual_steps = len(traj.steps)
    if actual_steps <= 0:
        return 0.0, False
    return max(0.0, min(1.0, case.optimal_steps / actual_steps)), False


def compute_answer_completeness(case: TestCase, traj: Trajectory, correct_behavior_class: bool) -> tuple[float, bool]:
    """Returns (answer_completeness, counts_in_aggregate).

    counts_in_aggregate is False only for no_tool/clarify cases, per spec:
    they get a real per-case score (1.0 if the model correctly declined/
    asked, else 0.0) for the trajectory JSONL, but are excluded from the
    format-level average so a pile of trivial 1.0s (nothing to check) can't
    dilute/inflate the average computed over cases that actually have facts
    to verify. tool_call cases with an empty required_facts list (a small
    number of cases where no safe deterministic literal exists -- see
    test_cases.yaml's header comment) DO count, scored as vacuously 1.0.
    """
    if case.expected_behavior in ("no_tool", "clarify"):
        return (1.0 if correct_behavior_class else 0.0), False
    if not case.required_facts:
        return 1.0, True
    matched = sum(1 for f in case.required_facts if _fact_matches(f, traj.final_text))
    return matched / len(case.required_facts), True


@dataclasses.dataclass
class RuleMetrics:
    tool_correctness: float  # 0..1
    argument_correctness: float  # 0..1, None-able if no tool calls expected/made
    hallucination_rate: float  # 0..1
    correct_behavior_class: bool  # did it correctly choose tool_call / no_tool / clarify at all
    path_optimization: float  # 0..1, deterministic: clamp(optimal_steps / actual_steps, 0, 1)
    timed_out: bool  # hit the step cap without terminating -- see path_optimization's note
    answer_completeness: float  # 0..1, deterministic: matched_required_facts / total_required_facts
    answer_completeness_counts_in_aggregate: bool  # False for no_tool/clarify -- see compute_answer_completeness
    notes: str = ""


def _values_equivalent(expected, actual) -> bool:
    if expected == actual:
        return True
    if isinstance(expected, str) and isinstance(actual, str):
        if expected.strip().lower() == actual.strip().lower():
            return True
        # Arithmetic-expression equivalence (e.g. "9 - 6" vs "3", or
        # "482000 * 1.1" vs "482000*1.10") for calculator-style arguments.
        if _ARITH_RE.match(expected) and _ARITH_RE.match(actual):
            try:
                ev = eval(expected, {"__builtins__": {}}, {})
                av = eval(actual, {"__builtins__": {}}, {})
                return abs(float(ev) - float(av)) < 1e-6
            except Exception:
                return False
    try:
        return abs(float(expected) - float(actual)) < 1e-6
    except (TypeError, ValueError):
        return False


def _args_match(expected_args: dict, actual_args: dict) -> float:
    """Fraction of expected keys whose values match (case/type-tolerant)."""
    if not expected_args:
        return 1.0
    matched = 0
    for k, v in expected_args.items():
        if k in actual_args and _values_equivalent(v, actual_args[k]):
            matched += 1
    return matched / len(expected_args)


def score_case(case: TestCase, traj: Trajectory) -> RuleMetrics:
    tool_steps = [s for s in traj.steps if s.parsed_tool_call is not None]
    made_any_tool_call = len(tool_steps) > 0
    non_hallucinated_names = [s.parsed_tool_call["name"] for s in tool_steps if not s.is_hallucinated_tool]

    total_steps = len(traj.steps)
    hallucinated_steps = sum(1 for s in traj.steps if s.is_hallucinated_tool)
    hallucination_rate = (hallucinated_steps / total_steps) if total_steps else 0.0

    path_optimization, timed_out = compute_path_optimization(case, traj)

    if case.expected_behavior in ("no_tool", "clarify"):
        correct_behavior_class = not made_any_tool_call
        tool_correctness = 1.0 if correct_behavior_class else 0.0
        argument_correctness = 1.0  # not applicable; keep scale consistent
        answer_completeness, counts = compute_answer_completeness(case, traj, correct_behavior_class)
        notes = "expected no tool call" if case.expected_behavior == "no_tool" else "expected a clarifying question, no tool call"
        return RuleMetrics(
            tool_correctness=tool_correctness,
            argument_correctness=argument_correctness,
            hallucination_rate=hallucination_rate,
            correct_behavior_class=correct_behavior_class,
            path_optimization=path_optimization,
            timed_out=timed_out,
            answer_completeness=answer_completeness,
            answer_completeness_counts_in_aggregate=counts,
            notes=notes,
        )

    # expected_behavior == "tool_call"
    expected_names = [t.tool for t in case.expected_trajectory]
    correct_behavior_class = made_any_tool_call

    if not expected_names:
        answer_completeness, counts = compute_answer_completeness(case, traj, correct_behavior_class)
        return RuleMetrics(
            tool_correctness=1.0,
            argument_correctness=1.0,
            hallucination_rate=hallucination_rate,
            correct_behavior_class=correct_behavior_class,
            path_optimization=path_optimization,
            timed_out=timed_out,
            answer_completeness=answer_completeness,
            answer_completeness_counts_in_aggregate=counts,
            notes="no expected trajectory defined",
        )

    # Tool correctness: how much of the expected tool sequence appears, in
    # order, among the non-hallucinated calls actually made.
    matches = 0
    cursor = 0
    for name in non_hallucinated_names:
        if cursor < len(expected_names) and name == expected_names[cursor]:
            matches += 1
            cursor += 1
    tool_correctness = matches / len(expected_names)

    # Argument correctness: for each expected step that WAS matched (by
    # position in the matched subsequence), compare arguments.
    arg_scores = []
    matched_tool_steps = [s for s in tool_steps if not s.is_hallucinated_tool]
    cursor = 0
    for expected in case.expected_trajectory:
        while cursor < len(matched_tool_steps) and matched_tool_steps[cursor].parsed_tool_call["name"] != expected.tool:
            cursor += 1
        if cursor >= len(matched_tool_steps):
            arg_scores.append(0.0)
            continue
        actual_args = matched_tool_steps[cursor].parsed_tool_call.get("arguments", {})
        arg_scores.append(_args_match(expected.arguments, actual_args))
        cursor += 1
    argument_correctness = sum(arg_scores) / len(arg_scores) if arg_scores else 0.0

    answer_completeness, counts = compute_answer_completeness(case, traj, correct_behavior_class)

    return RuleMetrics(
        tool_correctness=tool_correctness,
        argument_correctness=argument_correctness,
        hallucination_rate=hallucination_rate,
        correct_behavior_class=correct_behavior_class,
        path_optimization=path_optimization,
        timed_out=timed_out,
        answer_completeness=answer_completeness,
        answer_completeness_counts_in_aggregate=counts,
    )
