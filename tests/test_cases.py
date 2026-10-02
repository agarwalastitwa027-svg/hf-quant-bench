"""Tests for bench/cases.py's YAML loader against the real test_cases.yaml --
the ground-truth file every format is scored against. These catch the kind
of structural regression a bad hand-edit to the YAML would introduce (a
duplicate id, a missing required field, a malformed expected_trajectory)
before it silently corrupts every format's scores.
"""

from __future__ import annotations

import pytest

from hf_quant_bench.bench.cases import load_cases


def test_load_cases_returns_100_cases():
    cases = load_cases()
    assert len(cases) == 100


def test_case_ids_are_unique():
    cases = load_cases()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_every_case_has_a_valid_expected_behavior():
    cases = load_cases()
    for c in cases:
        assert c.expected_behavior in ("tool_call", "no_tool", "clarify"), c.id


def test_every_case_has_required_facts_and_optimal_steps():
    """load_cases() raises KeyError on a case missing either field (see its
    own comment: these drive two of the five primary metrics, so a missing
    value must fail loudly, not silently default)."""
    cases = load_cases()
    for c in cases:
        assert isinstance(c.required_facts, list), c.id
        assert isinstance(c.optimal_steps, int) and c.optimal_steps >= 1, c.id


def test_tool_call_cases_have_a_nonempty_expected_trajectory_or_documented_exception():
    """A tool_call case with no expected_trajectory silently short-circuits
    score_case() to a trivial 1.0/1.0 (see metrics_rule.py) -- worth
    flagging, not silently allowing, if a future case is added without one."""
    cases = load_cases()
    for c in cases:
        if c.expected_behavior == "tool_call":
            assert len(c.expected_trajectory) >= 1, f"{c.id} is tool_call but has no expected_trajectory"


def test_load_cases_raises_on_missing_required_field(tmp_path):
    bad_yaml = tmp_path / "bad_cases.yaml"
    bad_yaml.write_text(
        """
- id: broken_case
  category: test
  prompt: "hello"
  available_tools: []
  expected_behavior: no_tool
  judge_notes: ""
"""
    )
    with pytest.raises(KeyError):
        load_cases(bad_yaml)
