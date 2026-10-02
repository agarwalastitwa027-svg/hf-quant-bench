"""Tests for bench/report.py's summarize() aggregation -- in particular the
None-exclusion rule for answer_completeness, which is what implements
"no_tool/clarify cases don't count toward the format-level average" (see
compute_answer_completeness's docstring in metrics_rule.py). A regression
here would silently change every format's answer_completeness column
without touching any of the per-case scoring logic.
"""

from __future__ import annotations

from hf_quant_bench.bench.report import summarize


def make_record(**overrides) -> dict:
    base = {
        "tool_correctness": 1.0,
        "argument_correctness": 1.0,
        "hallucination_rate": 0.0,
        "correct_behavior_class": 1.0,
        "path_optimization": 1.0,
        "timed_out": 0.0,
        "answer_completeness": 1.0,
        "prefill_s_mean": 0.1,
        "decode_tokens_per_s": 50.0,
        "total_wall_s": 1.0,
        "n_steps": 1,
    }
    base.update(overrides)
    return base


def test_summarize_excludes_none_answer_completeness_from_average():
    records = [
        make_record(answer_completeness=1.0),
        make_record(answer_completeness=0.0),
        make_record(answer_completeness=None),  # a no_tool/clarify case
    ]
    summary = summarize("test-format", records, on_disk_size_mb=100.0, peak_vram_mb=500.0, peak_ram_mb=1000.0)
    # Mean of [1.0, 0.0] only -- the None case is excluded, not treated as 0.
    assert summary.answer_completeness == 0.5
    assert summary.n_cases == 3  # excluded from the metric, not from n_cases


def test_summarize_all_none_answer_completeness_defaults_to_zero():
    records = [make_record(answer_completeness=None)]
    summary = summarize("test-format", records, on_disk_size_mb=None, peak_vram_mb=None, peak_ram_mb=None)
    assert summary.answer_completeness == 0.0


def test_summarize_carries_forward_resource_fields_unchanged():
    records = [make_record()]
    summary = summarize(
        "test-format", records, on_disk_size_mb=123.4, peak_vram_mb=567.8, peak_ram_mb=999.9, backend_note="note"
    )
    assert summary.on_disk_size_mb == 123.4
    assert summary.peak_vram_mb == 567.8
    assert summary.peak_ram_mb == 999.9
    assert summary.backend_note == "note"


def test_summarize_mean_of_simple_metrics():
    records = [make_record(tool_correctness=1.0), make_record(tool_correctness=0.0)]
    summary = summarize("test-format", records, on_disk_size_mb=None, peak_vram_mb=None, peak_ram_mb=None)
    assert summary.tool_correctness == 0.5
