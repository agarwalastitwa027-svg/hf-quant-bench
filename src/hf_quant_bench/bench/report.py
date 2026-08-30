"""Aggregate per-case results into results.csv / results.md.

path_optimization and answer_completeness are deterministic, rule-based
metrics computed from data this project already has exactly (the minimal
step count per test case; the literal values the deterministic mock tools
returned) -- see metrics_rule.py. There is no cloud LLM-as-a-judge anywhere
in these primary metrics. `*_judged` columns are populated only if
`--local-judge` was used (see local_judge.py); they are supplementary and
never overwrite the primary rule-based columns.
"""

from __future__ import annotations

import csv
import dataclasses
import statistics
from pathlib import Path


@dataclasses.dataclass
class FormatSummary:
    format_name: str
    n_cases: int
    tool_correctness: float
    argument_correctness: float
    hallucination_rate: float
    behavior_class_accuracy: float
    path_optimization: float  # deterministic: mean of clamp(optimal_steps/actual_steps, 0, 1)
    timeout_rate: float  # fraction of trajectories that hit the step cap without terminating
    answer_completeness: float  # deterministic: mean of matched_required_facts/total_required_facts
    on_disk_size_mb: float | None
    peak_vram_mb: float | None
    peak_ram_mb: float | None
    prefill_latency_s_mean: float | None
    decode_tokens_per_s_mean: float | None
    trajectory_latency_s_mean: float
    steps_per_trajectory_mean: float
    backend_note: str = ""
    # Populated only by --local-judge (offline, opt-in, off by default).
    # None means "not run," not "scored zero" -- kept distinct from the
    # primary columns above on purpose; see local_judge.py.
    path_optimization_judged: float | None = None
    answer_completeness_judged: float | None = None


def summarize(
    format_name: str,
    case_records: list[dict],
    on_disk_size_mb: float | None,
    peak_vram_mb: float | None,
    peak_ram_mb: float | None,
    backend_note: str = "",
) -> FormatSummary:
    n = len(case_records)

    def avg(key, sub=None):
        vals = []
        for r in case_records:
            v = r[key] if sub is None else r.get(sub, {}).get(key)
            if v is not None:
                vals.append(v)
        return statistics.mean(vals) if vals else None

    prefill_vals = [r["prefill_s_mean"] for r in case_records if r.get("prefill_s_mean") is not None]
    decode_tps_vals = [r["decode_tokens_per_s"] for r in case_records if r.get("decode_tokens_per_s") is not None]

    return FormatSummary(
        format_name=format_name,
        n_cases=n,
        tool_correctness=avg("tool_correctness") or 0.0,
        argument_correctness=avg("argument_correctness") or 0.0,
        hallucination_rate=avg("hallucination_rate") or 0.0,
        behavior_class_accuracy=avg("correct_behavior_class") or 0.0,
        # path_optimization: computed for every case (no_tool/clarify included,
        # optimal_steps=1 for those) -- unlike answer_completeness there is no
        # aggregate-exclusion rule for this metric.
        path_optimization=avg("path_optimization") or 0.0,
        timeout_rate=avg("timed_out") or 0.0,
        # answer_completeness: `run.py` already writes None into this key for
        # no_tool/clarify cases (answer_completeness_counts_in_aggregate=False),
        # so avg()'s None-filtering is what implements "excluded from the
        # denominator" here -- see compute_answer_completeness's docstring.
        answer_completeness=avg("answer_completeness") or 0.0,
        on_disk_size_mb=on_disk_size_mb,
        peak_vram_mb=peak_vram_mb,
        peak_ram_mb=peak_ram_mb,
        prefill_latency_s_mean=statistics.mean(prefill_vals) if prefill_vals else None,
        decode_tokens_per_s_mean=statistics.mean(decode_tps_vals) if decode_tps_vals else None,
        trajectory_latency_s_mean=avg("total_wall_s") or 0.0,
        steps_per_trajectory_mean=avg("n_steps") or 0.0,
        backend_note=backend_note,
    )


CSV_FIELDS = [
    "format_name", "n_cases", "tool_correctness", "argument_correctness", "hallucination_rate",
    "behavior_class_accuracy", "path_optimization", "timeout_rate", "answer_completeness",
    "on_disk_size_mb", "peak_vram_mb", "peak_ram_mb", "prefill_latency_s_mean", "decode_tokens_per_s_mean",
    "trajectory_latency_s_mean", "steps_per_trajectory_mean", "backend_note",
    "path_optimization_judged", "answer_completeness_judged",
]


def write_csv(path: Path, summaries: list[FormatSummary]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for s in summaries:
            writer.writerow(dataclasses.asdict(s))
    print(f"[report] wrote {path}")


def _delta_str(baseline: float | None, value: float | None, higher_is_better: bool, fmt: str = "{:.3f}") -> str:
    if baseline is None or value is None:
        return "n/a"
    d = value - baseline
    sign = "+" if d >= 0 else ""
    good = (d >= 0) == higher_is_better
    marker = "✅" if good and abs(d) > 1e-9 else ("⚠️" if abs(d) > 1e-9 else "=")
    return f"{sign}{fmt.format(d)} {marker}"


_CALIBRATED_PREFIXES = ("gptq", "awq")
_UNCALIBRATED_PREFIXES = ("fp16", "bnb", "gguf", "onnx", "torchao", "openvino")


def _calibration_comparison(summaries: list[FormatSummary]) -> list[str]:
    """This is the central question this project was built to answer: does
    spending time on a calibration dataset (GPTQ, AWQ) actually buy anything
    over simple round-to-nearest quantization (bitsandbytes, GGUF), for
    agentic tool-calling specifically? Classification is by format-name
    prefix, a heuristic -- it will misclassify a target that doesn't fit this
    project's existing naming, but every target this project ships does.
    """
    calibrated = [s for s in summaries if s.format_name.startswith(_CALIBRATED_PREFIXES)]
    uncalibrated = [s for s in summaries if s.format_name.startswith(_UNCALIBRATED_PREFIXES)]
    if not calibrated or not uncalibrated:
        return []

    calib_avg = statistics.mean(s.tool_correctness for s in calibrated)
    uncalib_avg = statistics.mean(s.tool_correctness for s in uncalibrated)
    verdict = (
        "Calibration helped here"
        if calib_avg > uncalib_avg + 0.02
        else "Calibration did NOT help here"
        if calib_avg < uncalib_avg - 0.02
        else "No meaningful difference here"
    )

    lines = [
        f"**{verdict}**: mean tool-call correctness across calibrated formats "
        f"({', '.join(s.format_name for s in calibrated)}) is {calib_avg:.2f}, vs. "
        f"{uncalib_avg:.2f} across uncalibrated round-to-nearest formats "
        f"({', '.join(s.format_name for s in uncalibrated)}).",
        "",
        "| Format | Calibrated? | Tool correct. | Arg correct. |",
        "|---|---|---|---|",
    ]
    for s in sorted(summaries, key=lambda s: s.format_name.startswith(_CALIBRATED_PREFIXES)):
        is_calib = s.format_name.startswith(_CALIBRATED_PREFIXES)
        if not (is_calib or s.format_name.startswith(_UNCALIBRATED_PREFIXES)):
            continue
        lines.append(f"| {s.format_name} | {'yes' if is_calib else 'no'} | {s.tool_correctness:.2f} | {s.argument_correctness:.2f} |")
    return lines


def _pareto_frontier(summaries: list[FormatSummary]) -> list[str]:
    """A format is Pareto-optimal for size-vs-accuracy if no other format is
    both smaller AND more accurate (tool_correctness) than it.
    """
    sized = [s for s in summaries if s.on_disk_size_mb is not None]
    frontier = []
    for s in sized:
        dominated = any(
            (o.on_disk_size_mb <= s.on_disk_size_mb and o.tool_correctness >= s.tool_correctness and
             (o.on_disk_size_mb < s.on_disk_size_mb or o.tool_correctness > s.tool_correctness))
            for o in sized if o.format_name != s.format_name
        )
        if not dominated:
            frontier.append(s.format_name)
    return frontier


def write_markdown(path: Path, summaries: list[FormatSummary], baseline_name: str = "fp16") -> None:
    baseline = next((s for s in summaries if s.format_name == baseline_name), None)

    lines = ["# Quantization format comparison", ""]
    if baseline is None:
        lines.append(f"_Note: baseline format '{baseline_name}' was not found in results; deltas omitted._")
    lines += [
        "",
        f"Baseline: **{baseline_name}**" if baseline else "",
        "",
        "**All metrics below are deterministic and rule-based** -- tool_correctness, "
        "argument_correctness, and hallucination_rate compare against the test suite's "
        "known-correct tool sequence; path_optimization compares actual steps taken against "
        "each case's known-minimal step count; answer_completeness checks for the literal "
        "values the deterministic mock tools returned in the model's final answer "
        "(substring match for text, tolerance-based numeric match for numbers). No cloud API, "
        "no LLM-as-a-judge, nothing non-reproducible. `*_judged` columns, if present, come from "
        "an optional local GGUF model run offline via `--local-judge` and are supplementary --  "
        "they never overwrite the primary columns.",
        "",
        "**Read this before the decode tok/s column**: gguf-* rows run on the "
        "llama.cpp backend; fp16/bnb-* rows run on HF `transformers.generate()`. "
        "A large decode-speed gap between them reflects inference *engine* "
        "efficiency (llama.cpp's CUDA decode loop has far less Python-level "
        "overhead than transformers') at least as much as it reflects the "
        "quantization format itself. It is not a clean measurement of "
        "\"what quantization costs\" in isolation -- treat cross-engine speed "
        "comparisons here as directional, not definitive.",
        "",
        "**Size column for bnb-nf4/bnb-int8 is `n/a` on purpose**: bitsandbytes "
        "quantizes the original fp16 checkpoint at load time rather than "
        "producing a separate quantized file, so there is no standalone "
        "on-disk artifact to compare against formats like GGUF that do. "
        "Reporting it as ~0 MB (the size of the tiny saved config) would "
        "have falsely made these formats dominate the Pareto frontier below.",
        "",
        "| Format | Tool correct. | Arg correct. | Halluc. rate | Behavior acc. | Path opt | Timeout rate | Completeness | Size (MB) | Peak VRAM (MB) | Peak RAM (MB) | Decode tok/s | Traj latency (s) | Steps/traj |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt(v, spec="{:.2f}"):
        return "n/a" if v is None else spec.format(v)

    for s in summaries:
        lines.append(
            f"| {s.format_name} | {s.tool_correctness:.2f} | {s.argument_correctness:.2f} | "
            f"{s.hallucination_rate:.2f} | {s.behavior_class_accuracy:.2f} | "
            f"{s.path_optimization:.2f} | "
            f"{s.timeout_rate:.2f} | "
            f"{s.answer_completeness:.2f} | "
            f"{fmt(s.on_disk_size_mb, '{:.1f}')} | "
            f"{fmt(s.peak_vram_mb, '{:.0f}')} | "
            f"{fmt(s.peak_ram_mb, '{:.0f}')} | "
            f"{fmt(s.decode_tokens_per_s_mean, '{:.1f}')} | "
            f"{s.trajectory_latency_s_mean:.2f} | {s.steps_per_trajectory_mean:.1f} |"
        )

    if any(s.path_optimization_judged is not None or s.answer_completeness_judged is not None for s in summaries):
        lines += [
            "", "## Optional local-judge scores (supplementary, not primary)", "",
            "| Format | Path opt (judged) | Completeness (judged) |",
            "|---|---|---|",
        ]
        for s in summaries:
            lines.append(f"| {s.format_name} | {fmt(s.path_optimization_judged)} | {fmt(s.answer_completeness_judged)} |")

    if baseline:
        lines += ["", "## Deltas vs. fp16 baseline", "", "| Format | Δ tool correct. | Δ arg correct. | Δ halluc. rate | Δ size (MB) | Δ decode tok/s |", "|---|---|---|---|---|---|"]
        for s in summaries:
            if s.format_name == baseline_name:
                continue
            lines.append(
                f"| {s.format_name} | {_delta_str(baseline.tool_correctness, s.tool_correctness, True)} | "
                f"{_delta_str(baseline.argument_correctness, s.argument_correctness, True)} | "
                f"{_delta_str(baseline.hallucination_rate, s.hallucination_rate, False)} | "
                f"{_delta_str(baseline.on_disk_size_mb, s.on_disk_size_mb, False, '{:.1f}')} | "
                f"{_delta_str(baseline.decode_tokens_per_s_mean, s.decode_tokens_per_s_mean, True, '{:.1f}')} |"
            )

    calib_lines = _calibration_comparison(summaries)
    if calib_lines:
        lines += ["", "## Calibrated vs. uncalibrated quantization", ""] + calib_lines

    frontier = _pareto_frontier(summaries)
    lines += [
        "",
        "## Pareto-optimal formats (size vs. tool-call accuracy)",
        "",
        "A format is listed here if no other format is simultaneously smaller on disk "
        "AND more tool-correct. These are the formats worth actually considering; "
        "everything else is strictly dominated by one of these on the size/accuracy tradeoff.",
        "",
    ]
    for f in frontier:
        lines.append(f"- **{f}**")

    path.write_text("\n".join(lines))
    print(f"[report] wrote {path}")
