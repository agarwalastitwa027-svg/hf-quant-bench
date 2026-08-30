"""Part 2: run the agentic tool-calling benchmark across every converted
format, one at a time, releasing memory between formats (8GB VRAM will not
hold two loaded models).

Usage:
    python -m hf_quant_bench.bench.run --manifest outputs/qwen2.5-1.5b/manifest.json \\
        --model Qwen/Qwen2.5-1.5B-Instruct --out results/

    # fast sanity check: 3 cases, one format, <2 min
    python -m hf_quant_bench.bench.run --manifest outputs/qwen2.5-1.5b/manifest.json \\
        --model Qwen/Qwen2.5-1.5B-Instruct --out results/ --smoke --formats fp16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import traceback
from pathlib import Path

from hf_quant_bench.bench.cases import TestCase, load_cases
from hf_quant_bench.bench.loader import load_model_for_format
from hf_quant_bench.bench.metrics_rule import score_case
from hf_quant_bench.bench.preflight import PreflightResult, check_against_reference, run_preflight, select_preflight_cases
from hf_quant_bench.bench.report import FormatSummary, summarize, write_csv, write_markdown
from hf_quant_bench.bench.resource_monitor import release_gpu_memory, reset_peak_tracking, snapshot
from hf_quant_bench.bench.runner import Trajectory, run_case
from hf_quant_bench.env_check import probe_igpu, probe_npu, probe_gpu


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="path to manifest.json produced by convert.py")
    p.add_argument("--model", required=True, help="original HF repo id or local dir (used for tokenizers/base weights)")
    p.add_argument("--out", default="bench_results", help="output directory for results.csv / results.md / trajectories")
    p.add_argument("--cases", default=None, help="path to test_cases.yaml (default: bundled suite)")
    p.add_argument("--formats", nargs="*", default=None, help="restrict to these format names (default: every 'ok' entry in the manifest)")
    p.add_argument("--smoke", action="store_true", help="run only 3 cases against one format (validate the loop end-to-end, <2 min)")
    p.add_argument("--gguf-quant-type", default=None, help="which GGUF quant type to load when format=gguf (default: first available)")
    p.add_argument(
        "--skip-preflight", action="store_true",
        help="skip the per-format 3-case validation gate (NOT recommended -- this is what would have "
             "caught a format silently emitting zero tool calls before burning a full sweep on it)",
    )
    p.add_argument(
        "--local-judge", default=None, metavar="GGUF_PATH",
        help="optional, off by default: after the full sweep completes and every subject model has "
             "been released, score a SEPARATE, supplementary pair of columns "
             "(path_optimization_judged/answer_completeness_judged) using this local GGUF model via "
             "llama.cpp. Runs offline from the trajectory JSONL, never concurrently with a subject "
             "model. Does not replace or affect the primary deterministic metrics.",
    )
    return p


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def trajectory_to_jsonl_record(case: TestCase, traj: Trajectory, rule) -> dict:
    return {
        "case_id": case.id,
        "category": case.category,
        "prompt": case.prompt,
        "expected_behavior": case.expected_behavior,
        "optimal_steps": case.optimal_steps,
        "required_facts": case.required_facts,
        "ended_reason": traj.ended_reason,
        "error": traj.error,
        "final_text": traj.final_text,
        "total_wall_s": traj.total_wall_s,
        "steps": [
            {
                "step_index": s.step_index,
                "raw_model_output": s.raw_model_output,
                "parsed_tool_call": s.parsed_tool_call,
                "tool_result": s.tool_result,
                "is_hallucinated_tool": s.is_hallucinated_tool,
                "prefill_s": s.prefill_s,
                "decode_s": s.decode_s,
                "decode_tokens": s.decode_tokens,
            }
            for s in traj.steps
        ],
        # No "judge_score" key -- there is no cloud LLM-as-a-judge in this
        # project. rule_metrics now includes path_optimization/timed_out/
        # answer_completeness alongside tool_correctness/argument_correctness/
        # hallucination_rate, all deterministic (see metrics_rule.py). A
        # "local_judge_score" key is added here only if --local-judge was
        # used (see local_judge.py's run_local_judge_on_file, called from
        # main() as a separate offline post-processing pass).
        "rule_metrics": dataclasses.asdict(rule),
    }


def run_format(
    format_name: str,
    manifest_entry: dict,
    args,
    cases: list[TestCase],
    out_dir: Path,
    ctx_flags: dict,
    display_name: str | None = None,
    gguf_quant_type: str | None = None,
    preflight_reference: PreflightResult | None = None,
    preflight_cases: list[TestCase] | None = None,
) -> FormatSummary | None:
    display_name = display_name or format_name
    print(f"\n{'=' * 72}\n[bench] format: {display_name}\n{'=' * 72}")
    reset_peak_tracking()

    try:
        handle = load_model_for_format(
            format_name,
            manifest_entry,
            args.model,
            device=ctx_flags["device"],
            igpu_available=ctx_flags["igpu"],
            npu_available=ctx_flags["npu"],
            gguf_quant_type=gguf_quant_type or args.gguf_quant_type,
        )
    except Exception as e:
        print(f"[bench] {display_name}: FAILED TO LOAD — {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        return None

    if preflight_reference is not None and preflight_cases is not None:
        print(f"[bench:{display_name}] running {len(preflight_cases)}-case preflight gate before the full sweep...")
        pf = run_preflight(handle, display_name, preflight_cases)
        fail_reasons = check_against_reference(pf, preflight_reference)
        if fail_reasons:
            print(f"\n{'!' * 72}")
            print(f"[bench] PREFLIGHT ABORTED: {display_name} failed the validation gate")
            for r in fail_reasons:
                print(f"[bench]   - {r}")
            print(f"[bench] Skipping the full {len(cases)}-case sweep for {display_name}.")
            print(f"{'!' * 72}\n")
            handle.close()
            release_gpu_memory()
            snapshot()  # drains the pollers cleanly even though we discard the reading
            return FormatSummary(
                format_name=display_name, n_cases=len(preflight_cases),
                tool_correctness=0.0, argument_correctness=0.0, hallucination_rate=0.0,
                behavior_class_accuracy=0.0, path_optimization=0.0, timeout_rate=0.0, answer_completeness=0.0,
                on_disk_size_mb=None, peak_vram_mb=None, peak_ram_mb=None,
                prefill_latency_s_mean=None, decode_tokens_per_s_mean=None,
                trajectory_latency_s_mean=0.0, steps_per_trajectory_mean=pf.mean_steps,
                backend_note=f"PREFLIGHT ABORTED: {'; '.join(fail_reasons)}",
            )
        print(f"[bench:{display_name}] preflight gate: PASSED (mean_steps={pf.mean_steps:.2f}, tool call parsed: {pf.any_tool_call_parsed})")

    jsonl_path = out_dir / f"trajectories_{display_name}.jsonl"
    case_records = []

    try:
        with open(jsonl_path, "w") as jf:
            for case in cases:
                print(f"[bench:{display_name}] case {case.id} ({case.category})")
                try:
                    traj = run_case(handle, case)
                except Exception as e:
                    print(f"[bench:{display_name}]   ERROR during case: {e}")
                    continue

                rule = score_case(case, traj)

                record = trajectory_to_jsonl_record(case, traj, rule)
                jf.write(json.dumps(record) + "\n")

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
                        # None here (not 0.0/1.0) is what makes report.py's
                        # avg()-based aggregation exclude this case from the
                        # format-level answer_completeness mean -- see
                        # compute_answer_completeness's docstring for why
                        # no_tool/clarify cases are excluded but still get a
                        # real per-case score in the trajectory JSONL above.
                        "answer_completeness": rule.answer_completeness if rule.answer_completeness_counts_in_aggregate else None,
                        "prefill_s_mean": sum(prefill_vals) / len(prefill_vals) if prefill_vals else None,
                        "decode_tokens_per_s": (decode_tok / decode_s) if decode_s > 0 else None,
                        "total_wall_s": traj.total_wall_s,
                        "n_steps": len(traj.steps),
                    }
                )
    finally:
        handle.close()
        release_gpu_memory()

    res = snapshot()
    if format_name == "gguf" and gguf_quant_type:
        import os
        quant_path = manifest_entry.get("extra", {}).get("quant_types", {}).get(gguf_quant_type)
        on_disk_mb = (os.path.getsize(quant_path) / (1024**2)) if quant_path else None
    elif format_name in ("bnb-nf4", "bnb-int8"):
        # bitsandbytes quantizes at load time from the original fp16
        # checkpoint -- there is no separate persisted quantized artifact,
        # so manifest_entry["size_bytes"] is near-zero (just a tiny config
        # file) and not a meaningful "on-disk size" to compare against
        # formats that do produce a standalone quantized file. Reporting it
        # as ~0 MB would falsely make these formats dominate the Pareto
        # frontier on size.
        on_disk_mb = None
    else:
        on_disk_mb = manifest_entry.get("size_bytes")
        on_disk_mb = (on_disk_mb / (1024**2)) if on_disk_mb else None

    backend_note_parts = []
    if format_name == "awq-4bit":
        backend_note_parts.append(manifest_entry.get("extra", {}).get("backend", ""))
    # Which measurement method produced peak_vram_mb/peak_ram_mb -- e.g.
    # "vram=torch-allocator-fallback" means nvidia-smi was unreachable, so
    # this figure will read ~0 for any non-torch backend (GGUF) regardless
    # of its real usage; that context belongs on the row, not just in a
    # console warning that scrolls away.
    backend_note_parts.append(f"vram={res.vram_method}")
    backend_note_parts.append(f"ram={res.ram_method}")
    backend_note = "; ".join(p for p in backend_note_parts if p)

    return summarize(
        display_name,
        case_records,
        on_disk_size_mb=on_disk_mb,
        peak_vram_mb=res.peak_vram_mb,
        peak_ram_mb=res.peak_ram_mb,
        backend_note=backend_note,
    )


def establish_fp16_reference(
    ok_entries: dict, args, preflight_cases: list[TestCase], ctx_flags: dict
) -> PreflightResult | None:
    """Always loads fp16 fresh and runs the same 3-case preflight against
    it, regardless of whether fp16 is actually in this run's --formats --
    a targeted re-run of just gptq-4bit/awq-4bit still needs a reference to
    gate against. This does mean fp16 gets loaded twice (once here, once
    again in its own run_format() call if it's also in --formats) -- a
    deliberate, cheap simplicity-over-micro-optimization tradeoff, since
    loading a single small model twice costs seconds against a sweep that
    can run for an hour.

    Returns None (with a loud warning, not a silent skip) if fp16 isn't an
    'ok' entry in the manifest, in which case the caller must run every
    format ungated rather than pretend a reference exists.
    """
    if "fp16" not in ok_entries:
        print(
            f"\n{'!' * 72}\n"
            "[bench] WARNING: 'fp16' is not an 'ok' entry in the manifest -- no reference "
            "is available, so the per-format preflight gate is DISABLED for this entire run. "
            "Every format's tool-calling numbers below are unverified against a known-good "
            "baseline; a format could be silently emitting zero tool calls and this run would "
            "not catch it (this is exactly the failure mode the gate exists to catch).\n"
            f"{'!' * 72}\n",
            file=sys.stderr,
        )
        return None

    print(f"\n{'=' * 72}\n[bench] establishing fp16 reference ({len(preflight_cases)} preflight cases)\n{'=' * 72}")
    handle = load_model_for_format(
        "fp16", ok_entries["fp16"], args.model,
        device=ctx_flags["device"], igpu_available=ctx_flags["igpu"], npu_available=ctx_flags["npu"],
        gguf_quant_type=None,
    )
    try:
        reference = run_preflight(handle, "fp16", preflight_cases)
    finally:
        handle.close()
        release_gpu_memory()

    print(
        f"[bench] fp16 reference: chat_template_hash={(reference.chat_template_hash or 'unavailable')[:16]}..., "
        f"mean_steps={reference.mean_steps:.2f}, any_tool_call_parsed={reference.any_tool_call_parsed}"
    )
    if not reference.any_tool_call_parsed or reference.mean_steps < 1.2:
        print(
            "[bench] WARNING: the fp16 reference itself looks degenerate (no tool call parsed, or "
            "steps-per-trajectory below 1.2). Every other format will be gated against a bad "
            "baseline -- check the preflight test cases and the fp16 checkpoint before trusting "
            "any pass/fail below.",
            file=sys.stderr,
        )
    return reference


def run_local_judge(gguf_path: str, summaries: list[FormatSummary], out_dir: Path, cases: list[TestCase]) -> None:
    """Offline, opt-in, post-sweep only -- every subject model has already
    been released by the time this runs (called from main() strictly after
    the main format loop). See local_judge.py's module docstring for the
    full set of constraints this exists to satisfy.
    """
    from hf_quant_bench.bench.local_judge import load_local_judge, run_local_judge_on_file

    cases_by_id = {c.id: c for c in cases}
    judge_label = Path(gguf_path).name  # filename conventionally embeds model+quant, e.g. "...-Q4_K_M.gguf"

    print(f"\n{'=' * 72}\n[bench] running optional local judge ({judge_label}) on {len(summaries)} formats\n{'=' * 72}")
    llm = load_local_judge(gguf_path)
    try:
        for s in summaries:
            jsonl_path = out_dir / f"trajectories_{s.format_name}.jsonl"
            if not jsonl_path.exists():
                print(f"[bench] local judge: skipping {s.format_name}, no trajectory file (likely a preflight abort)")
                continue
            print(f"[bench] local judge: scoring {s.format_name}...")
            po, ac = run_local_judge_on_file(llm, jsonl_path, cases_by_id)
            s.path_optimization_judged = po
            s.answer_completeness_judged = ac
            s.backend_note = (s.backend_note + f"; local_judge={judge_label}").strip("; ")
    finally:
        llm.close()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest))
    ok_entries = {t["target"]: t for t in manifest["targets"] if t["status"] == "ok"}

    # KV-cache quantization is a separate axis from weight quantization, not
    # a conversion target: it reuses the fp16 checkpoint's weights untouched
    # and only changes how the KV cache is stored during generation. So
    # "fp16-kv4"/"fp16-kv2" never appear in the manifest -- they're a
    # loader-level variant of the "fp16" entry, opt-in via --formats (not
    # included in a default sweep, since bundling them in silently would mix
    # a genuinely orthogonal axis into what's otherwise a weight-quantization
    # comparison).
    KV_CACHE_FORMATS = {"fp16-kv4": "fp16", "fp16-kv2": "fp16"}

    format_names = args.formats if args.formats else list(ok_entries.keys())
    format_names = [f for f in format_names if f in ok_entries or f in KV_CACHE_FORMATS]
    if not format_names:
        print("error: no runnable formats found (check --manifest / --formats)", file=sys.stderr)
        return 2

    # "gguf" in the manifest is one conversion target holding *every*
    # --gguf-quant-types variant produced (Q4_0, Q4_K_M, ...). Comparing
    # quantization formats is the whole point of this benchmark, so expand it
    # into one run per quant type rather than silently benchmarking only
    # whichever one happens to be first in the dict -- unless the user
    # explicitly pinned a single one via --gguf-quant-type.
    run_plan: list[tuple[str, str, str | None]] = []  # (format_name, display_name, gguf_quant_type)
    for fmt in format_names:
        if fmt == "gguf":
            quant_types = ok_entries[fmt].get("extra", {}).get("quant_types", {})
            if args.gguf_quant_type:
                run_plan.append((fmt, f"gguf-{args.gguf_quant_type}", args.gguf_quant_type))
            elif quant_types:
                for qt in quant_types:
                    run_plan.append((fmt, f"gguf-{qt}", qt))
            else:
                run_plan.append((fmt, fmt, None))
        elif fmt in KV_CACHE_FORMATS:
            base = KV_CACHE_FORMATS[fmt]
            if base not in ok_entries:
                print(f"[bench] skipping '{fmt}': requires '{base}' to be an 'ok' entry in the manifest", file=sys.stderr)
                continue
            ok_entries[fmt] = ok_entries[base]  # loader looks up manifest_entry by this format_name's dict entry
            run_plan.append((fmt, fmt, None))
        else:
            run_plan.append((fmt, fmt, None))

    cases = load_cases(Path(args.cases) if args.cases else None)
    if args.smoke:
        cases = cases[:3]
        run_plan = run_plan[:1]
        print(f"[bench] --smoke: running {len(cases)} cases against format '{run_plan[0][1]}' only")

    gpu = probe_gpu()
    npu_ok, _ = probe_npu()
    igpu_ok, _ = probe_igpu()
    ctx_flags = {"device": "cuda" if gpu.available else "cpu", "npu": npu_ok, "igpu": igpu_ok}

    reference: PreflightResult | None = None
    preflight_cases: list[TestCase] = []
    if args.skip_preflight:
        print("[bench] --skip-preflight set: running every format ungated. Not recommended.", file=sys.stderr)
    elif args.smoke:
        pass  # --smoke is already the fast sanity check; the gate adds a redundant fp16 load on top of it
    else:
        preflight_cases = select_preflight_cases(cases)
        reference = establish_fp16_reference(ok_entries, args, preflight_cases, ctx_flags)

    summaries = []
    for fmt, display_name, gguf_quant_type in run_plan:
        summary = run_format(
            fmt, ok_entries[fmt], args, cases, out_dir, ctx_flags,
            display_name=display_name, gguf_quant_type=gguf_quant_type,
            preflight_reference=reference if fmt != "fp16" else None,
            preflight_cases=preflight_cases if fmt != "fp16" else None,
        )
        if summary:
            summaries.append(summary)
            # Written after every format, not just at the end: a long sweep
            # (100 cases x several formats can run for an hour) can get cut
            # off by something entirely outside this program's control --
            # e.g. the parent process/session ending -- and every format
            # already benchmarked before that point would otherwise leave no
            # aggregated results at all, only the raw per-case JSONL.
            write_csv(out_dir / "results.csv", summaries)
            write_markdown(out_dir / "results.md", summaries)

    if not summaries:
        print("error: no format produced results", file=sys.stderr)
        return 1

    if args.local_judge:
        run_local_judge(args.local_judge, summaries, out_dir, cases)
        write_csv(out_dir / "results.csv", summaries)
        write_markdown(out_dir / "results.md", summaries)

    print(f"\n[bench] done. {len(summaries)}/{len(run_plan)} formats produced results in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
