"""Read-only rescoring pass.

Re-runs score_case() against already-saved trajectory JSONLs, using the
patched test_cases.yaml (two required_facts corrections), without
re-running inference or the preflight gate for any format.

Reads:  bench_results_v3_full/trajectories_<format>.jsonl (unmodified)
        bench_results_v3_full/results.csv (only to carry forward
        on_disk_size_mb/peak_vram_mb/peak_ram_mb/backend_note, which are
        resource-monitor artifacts of the original generation run and are
        not present in the trajectory JSONL at all)
Writes: bench_results_v3_rescored/results.csv
        bench_results_v3_rescored/results.md
        (original bench_results_v3_full/ directory is untouched)
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

from hf_quant_bench.bench.cases import load_cases
from hf_quant_bench.bench.metrics_rule import score_case
from hf_quant_bench.bench.preflight import select_preflight_cases
from hf_quant_bench.bench.report import summarize, write_csv, write_markdown, FormatSummary, CSV_FIELDS
from hf_quant_bench.bench.runner import StepRecord, Trajectory

SRC_DIR = Path("bench_results_v3_full")
OUT_DIR = Path("bench_results_v3_rescored")

# gptq-4bit has no entry here on purpose -- see NO_TRAJECTORY_FORMATS below.
FORMATS = [
    "fp16",
    "bnb-nf4",
    "bnb-int8",
    "awq-4bit",
    "gguf-Q4_0",
    "gguf-Q4_K_M",
    "gguf-Q5_K_M",
    "gguf-Q8_0",
]

# gptq-4bit aborted at the preflight gate. run_format()'s abort branch
# constructs a FormatSummary directly and returns before ever opening the
# jsonl_path or calling run_case() on the full suite; run_preflight() itself
# calls run_case() on its 3 cases but only keeps aggregate mean_steps /
# any_tool_call_parsed / raw_outputs[0][:200] in PreflightResult -- the full
# Trajectory objects (final_text, tool_result, etc.) are discarded and never
# written anywhere. Confirmed: bench_results_v3_full/ has no
# trajectories_gptq-4bit.jsonl (checked -- 8 trajectory files, not 9).
# There is therefore no saved trajectory data to rescore this row against.
# Separately and independently: this format's row is carried forward
# unmodified below regardless, and its score cannot have moved, because
# select_preflight_cases() is deterministic and confirmed (this run) to
# select ['simple_weather_1', 'simple_weather_2', 'simple_calendar_1'] --
# neither of the two patched cases (simple_readfile_3, nested_readfile_q4)
# is among them.
NO_TRAJECTORY_FORMATS = {"gptq-4bit"}


def load_original_rows() -> dict[str, dict]:
    with open(SRC_DIR / "results.csv", newline="") as f:
        return {row["format_name"]: row for row in csv.DictReader(f)}


def row_to_resource_fields(row: dict) -> dict:
    return {
        "on_disk_size_mb": float(row["on_disk_size_mb"]) if row["on_disk_size_mb"] else None,
        "peak_vram_mb": float(row["peak_vram_mb"]) if row["peak_vram_mb"] else None,
        "peak_ram_mb": float(row["peak_ram_mb"]) if row["peak_ram_mb"] else None,
        "backend_note": row["backend_note"],
    }


def row_to_summary_unchanged(row: dict) -> FormatSummary:
    """Carries a results.csv row forward verbatim as a FormatSummary, for
    formats with no saved trajectory data to rescore against."""

    def f(key):
        v = row[key]
        return float(v) if v not in ("", None) else None

    return FormatSummary(
        format_name=row["format_name"],
        n_cases=int(row["n_cases"]),
        tool_correctness=f("tool_correctness"),
        argument_correctness=f("argument_correctness"),
        hallucination_rate=f("hallucination_rate"),
        behavior_class_accuracy=f("behavior_class_accuracy"),
        path_optimization=f("path_optimization"),
        timeout_rate=f("timeout_rate"),
        answer_completeness=f("answer_completeness"),
        on_disk_size_mb=f("on_disk_size_mb"),
        peak_vram_mb=f("peak_vram_mb"),
        peak_ram_mb=f("peak_ram_mb"),
        prefill_latency_s_mean=f("prefill_latency_s_mean"),
        decode_tokens_per_s_mean=f("decode_tokens_per_s_mean"),
        trajectory_latency_s_mean=f("trajectory_latency_s_mean"),
        steps_per_trajectory_mean=f("steps_per_trajectory_mean"),
        backend_note=row["backend_note"],
        path_optimization_judged=f("path_optimization_judged"),
        answer_completeness_judged=f("answer_completeness_judged"),
    )


def record_to_trajectory(rec: dict) -> Trajectory:
    steps = [
        StepRecord(
            step_index=s["step_index"],
            role_before_model="assistant",  # not stored in JSONL; score_case never reads this field
            raw_model_output=s["raw_model_output"],
            parsed_tool_call=s["parsed_tool_call"],
            tool_result=s["tool_result"],
            is_hallucinated_tool=s["is_hallucinated_tool"],
            prefill_s=s["prefill_s"],
            decode_s=s["decode_s"],
            decode_tokens=s["decode_tokens"],
        )
        for s in rec["steps"]
    ]
    return Trajectory(
        case_id=rec["case_id"],
        steps=steps,
        final_text=rec["final_text"],
        ended_reason=rec["ended_reason"],
        total_wall_s=rec["total_wall_s"],
        error=rec["error"],
    )


def rescore_format(format_name: str, cases_by_id: dict, original_rows: dict) -> tuple:
    jsonl_path = SRC_DIR / f"trajectories_{format_name}.jsonl"
    case_records = []
    new_jsonl_lines = []

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            case = cases_by_id[rec["case_id"]]
            traj = record_to_trajectory(rec)
            rule = score_case(case, traj)

            prefill_vals = [s.prefill_s for s in traj.steps]
            decode_tok = sum(s.decode_tokens for s in traj.steps)
            decode_s = sum(s.decode_s for s in traj.steps)

            case_records.append(
                {
                    "tool_correctness": rule.tool_correctness,
                    "argument_correctness": rule.argument_correctness,
                    "hallucination_rate": rule.hallucination_rate,
                    "correct_behavior_class": 1.0 if rule.correct_behavior_class else 0.0,
                    "path_optimization": rule.path_optimization,
                    "timed_out": 1.0 if rule.timed_out else 0.0,
                    "answer_completeness": rule.answer_completeness if rule.answer_completeness_counts_in_aggregate else None,
                    "prefill_s_mean": sum(prefill_vals) / len(prefill_vals) if prefill_vals else None,
                    "decode_tokens_per_s": (decode_tok / decode_s) if decode_s > 0 else None,
                    "total_wall_s": traj.total_wall_s,
                    "n_steps": len(traj.steps),
                }
            )

            new_rec = dict(rec)
            new_rec["required_facts"] = case.required_facts
            new_rec["rule_metrics"] = dataclasses.asdict(rule)
            new_jsonl_lines.append(json.dumps(new_rec))

    res = row_to_resource_fields(original_rows[format_name])
    summary = summarize(
        format_name,
        case_records,
        on_disk_size_mb=res["on_disk_size_mb"],
        peak_vram_mb=res["peak_vram_mb"],
        peak_ram_mb=res["peak_ram_mb"],
        backend_note=res["backend_note"],
    )
    return summary, new_jsonl_lines


def main():
    OUT_DIR.mkdir(exist_ok=True)

    cases = load_cases()
    cases_by_id = {c.id: c for c in cases}
    original_rows = load_original_rows()

    # Sanity-check the "no patched case is in the preflight sample" claim
    # this run, rather than trusting a comment -- abort loudly if it's ever
    # false instead of silently carrying forward a row that should have moved.
    preflight_ids = {c.id for c in select_preflight_cases(cases)}
    patched_ids = {"simple_readfile_3", "nested_readfile_q4"}
    assert not (preflight_ids & patched_ids), (
        f"preflight sample {preflight_ids} now overlaps patched cases {patched_ids} -- "
        "gptq-4bit can no longer be safely carried forward unchanged"
    )

    summaries = []
    for fmt in FORMATS:
        summary, new_jsonl_lines = rescore_format(fmt, cases_by_id, original_rows)
        summaries.append(summary)
        with open(OUT_DIR / f"trajectories_{fmt}.jsonl", "w") as f:
            f.write("\n".join(new_jsonl_lines) + "\n")

    for fmt in NO_TRAJECTORY_FORMATS:
        summaries.append(row_to_summary_unchanged(original_rows[fmt]))

    # Restore results.csv row order to match the original file.
    order = list(original_rows.keys())
    summaries.sort(key=lambda s: order.index(s.format_name))

    write_csv(OUT_DIR / "results.csv", summaries)
    write_markdown(OUT_DIR / "results.md", summaries)

    print(f"\n[rescore] wrote {len(summaries)} rows to {OUT_DIR}/results.csv")
    print(f"[rescore] {NO_TRAJECTORY_FORMATS} carried forward unchanged (no saved trajectory data; see script comment)")


if __name__ == "__main__":
    main()
