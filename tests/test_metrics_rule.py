"""Tests for the deterministic rule-based scoring in bench/metrics_rule.py.

These are the five numbers every row of results.csv is built from, and the
whole point of the project is that they're reproducible without a model or
an API call -- so they're testable the same way, with hand-built
Trajectory/StepRecord fixtures instead of real model output.
"""

from __future__ import annotations

from hf_quant_bench.bench.cases import TestCase, ToolCallSpec
from hf_quant_bench.bench.metrics_rule import (
    _fact_matches,
    compute_answer_completeness,
    compute_path_optimization,
    score_case,
)
from hf_quant_bench.bench.runner import StepRecord, Trajectory


def make_case(
    expected_behavior="tool_call",
    expected_trajectory=None,
    optimal_steps=1,
    required_facts=None,
) -> TestCase:
    return TestCase(
        id="test_case",
        category="test",
        prompt="irrelevant for scoring",
        available_tools=["get_weather"],
        expected_behavior=expected_behavior,
        expected_trajectory=expected_trajectory or [],
        judge_notes="",
        optimal_steps=optimal_steps,
        required_facts=required_facts or [],
    )


def make_step(parsed_tool_call=None, is_hallucinated_tool=False) -> StepRecord:
    return StepRecord(
        step_index=0,
        role_before_model="assistant",
        raw_model_output="",
        parsed_tool_call=parsed_tool_call,
        tool_result=None,
        is_hallucinated_tool=is_hallucinated_tool,
        prefill_s=0.0,
        decode_s=0.0,
        decode_tokens=0,
    )


def make_traj(steps, final_text="", ended_reason="answered") -> Trajectory:
    return Trajectory(
        case_id="test_case",
        steps=steps,
        final_text=final_text,
        ended_reason=ended_reason,
        total_wall_s=0.0,
    )


# --- _fact_matches ---------------------------------------------------------


def test_fact_matches_numeric_exact():
    assert _fact_matches("23", "The temperature is 23 degrees.")


def test_fact_matches_numeric_within_tolerance():
    # ±1.0 absolute floor -- 23 vs 23.9 is within tolerance.
    assert _fact_matches("23", "It's about 23.9 outside.")


def test_fact_matches_numeric_outside_tolerance():
    assert not _fact_matches("23", "It's about 40 outside.")


def test_fact_matches_numeric_false_positive_on_unrelated_number():
    """This is the exact ±1.0 false-positive documented in the README's
    'Known weak spots' section: a small required fact can be satisfied by an
    unrelated number that happens to land in range."""
    # required fact "2" matched by an unrelated "3" elsewhere in the answer.
    assert _fact_matches("2", "There were 3 retries logged for an unrelated error.")


def test_fact_matches_text_substring():
    assert _fact_matches("db_timeout", "the log shows a db_timeout error")
    assert not _fact_matches("db_timeout", "the log shows a generic error")


def test_fact_matches_text_case_and_whitespace_insensitive():
    assert _fact_matches("Db_Timeout", "  the log shows a   DB_TIMEOUT   error  ")


# --- compute_path_optimization ---------------------------------------------


def test_path_optimization_perfect():
    case = make_case(optimal_steps=2)
    traj = make_traj([make_step(), make_step()], ended_reason="answered")
    score, timed_out = compute_path_optimization(case, traj)
    assert score == 1.0
    assert not timed_out


def test_path_optimization_inefficient_but_terminated():
    case = make_case(optimal_steps=1)
    traj = make_traj([make_step(), make_step()], ended_reason="answered")
    score, timed_out = compute_path_optimization(case, traj)
    assert score == 0.5
    assert not timed_out


def test_path_optimization_step_cap_scores_zero_and_flags_timeout():
    """A hung/looping trajectory must not be scoreable as merely
    inefficient -- it's a distinct failure mode, counted separately in
    timeout_rate."""
    case = make_case(optimal_steps=1)
    traj = make_traj([make_step() for _ in range(6)], ended_reason="step_cap")
    score, timed_out = compute_path_optimization(case, traj)
    assert score == 0.0
    assert timed_out


# --- compute_answer_completeness --------------------------------------------


def test_answer_completeness_no_tool_correct_counts_full_but_excluded():
    case = make_case(expected_behavior="no_tool")
    traj = make_traj([], final_text="no need to call a tool here")
    score, counts = compute_answer_completeness(case, traj, correct_behavior_class=True)
    assert score == 1.0
    assert counts is False


def test_answer_completeness_no_tool_incorrect_scores_zero():
    case = make_case(expected_behavior="no_tool")
    traj = make_traj([], final_text="calling a tool anyway")
    score, counts = compute_answer_completeness(case, traj, correct_behavior_class=False)
    assert score == 0.0
    assert counts is False


def test_answer_completeness_empty_required_facts_is_vacuous_but_counted():
    case = make_case(expected_behavior="tool_call", required_facts=[])
    traj = make_traj([], final_text="anything")
    score, counts = compute_answer_completeness(case, traj, correct_behavior_class=True)
    assert score == 1.0
    assert counts is True


def test_answer_completeness_partial_match():
    case = make_case(expected_behavior="tool_call", required_facts=["23", "sunny"])
    traj = make_traj([], final_text="It is 23 degrees today.")
    score, counts = compute_answer_completeness(case, traj, correct_behavior_class=True)
    assert score == 0.5
    assert counts is True


# --- score_case: tool_call behavior -----------------------------------------


def test_score_case_tool_call_correct_sequence_and_args():
    case = make_case(
        expected_behavior="tool_call",
        expected_trajectory=[ToolCallSpec(tool="get_weather", arguments={"city": "Tokyo"})],
        optimal_steps=1,
        required_facts=["23"],
    )
    traj = make_traj(
        [make_step(parsed_tool_call={"name": "get_weather", "arguments": {"city": "Tokyo"}})],
        final_text="It's 23 degrees in Tokyo.",
        ended_reason="answered",
    )
    rule = score_case(case, traj)
    assert rule.tool_correctness == 1.0
    assert rule.argument_correctness == 1.0
    assert rule.correct_behavior_class is True
    assert rule.answer_completeness == 1.0
    assert rule.answer_completeness_counts_in_aggregate is True


def test_score_case_tool_call_wrong_tool_scores_zero_correctness():
    case = make_case(
        expected_behavior="tool_call",
        expected_trajectory=[ToolCallSpec(tool="get_weather", arguments={"city": "Tokyo"})],
    )
    traj = make_traj(
        [make_step(parsed_tool_call={"name": "get_calendar_events", "arguments": {}})],
        final_text="",
        ended_reason="answered",
    )
    rule = score_case(case, traj)
    assert rule.tool_correctness == 0.0


def test_score_case_hallucinated_tool_excluded_from_correctness_but_counted_in_rate():
    case = make_case(
        expected_behavior="tool_call",
        expected_trajectory=[ToolCallSpec(tool="get_weather", arguments={"city": "Tokyo"})],
    )
    traj = make_traj(
        [
            make_step(parsed_tool_call={"name": "fake_tool", "arguments": {}}, is_hallucinated_tool=True),
            make_step(parsed_tool_call={"name": "get_weather", "arguments": {"city": "Tokyo"}}),
        ],
        final_text="",
        ended_reason="answered",
    )
    rule = score_case(case, traj)
    assert rule.tool_correctness == 1.0  # the real call still matches, in order
    assert rule.hallucination_rate == 0.5  # 1 of 2 steps was hallucinated


def test_score_case_no_tool_expected_and_none_made_is_correct():
    case = make_case(expected_behavior="no_tool")
    traj = make_traj([], final_text="Here's a direct answer, no tool needed.", ended_reason="answered")
    rule = score_case(case, traj)
    assert rule.correct_behavior_class is True
    assert rule.tool_correctness == 1.0


def test_score_case_no_tool_expected_but_tool_called_is_incorrect():
    case = make_case(expected_behavior="no_tool")
    traj = make_traj(
        [make_step(parsed_tool_call={"name": "get_weather", "arguments": {}})],
        final_text="",
        ended_reason="answered",
    )
    rule = score_case(case, traj)
    assert rule.correct_behavior_class is False
    assert rule.tool_correctness == 0.0
