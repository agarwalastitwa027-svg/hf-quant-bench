"""Part 1: convert an HF causal LM into every quant format the environment
supports.

Usage:
    python -m hf_quant_bench.convert --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/qwen2.5-1.5b
    python -m hf_quant_bench.convert --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Diagnosed on an RTX 5070 (Blackwell, sm_120) under WSL2: GPTQ calibration
# and llmcompressor's AWQ path both crashed the whole WSL VM reproducibly
# after 20-60s of real calibration work, with no Python traceback and no
# OOM-killer record. Ruled out as generic instability (sustained matmul and
# torch.linalg/cuSOLVER loads both ran 60-90s with zero issues) and as a
# missing-kernel-for-this-architecture issue (Triton JIT compiles and runs
# fine, repeatedly). Forcing synchronous CUDA execution fixed it completely
# on both paths -- this points to an async-kernel-launch race that WSL2's
# GPU paravirtualization (dxgk) doesn't handle cleanly for these specific
# calibration workloads' kernel launch patterns, not a fundamental
# incompatibility.
#
# Scoped to conversion only (not set in env_check.py, which bench/run.py
# also imports): forcing synchronous CUDA execution has a real throughput
# cost, and inference/benchmarking never crashed even across a full 7-format
# x 27-case sweep without it -- only calibration did. Must run before any
# torch import happens anywhere in this process, hence sys.argv is parsed
# directly here rather than after argparse. Set as a default (not a hard
# override) so it can still be disabled by exporting CUDA_LAUNCH_BLOCKING=0
# to test whether a future driver/library update has fixed the underlying
# race.
if "--only" not in sys.argv or any(t in sys.argv for t in ("gptq-4bit", "awq-4bit")):
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

from hf_quant_bench import targets  # noqa: F401  (side-effect: registers targets)
from hf_quant_bench.env_check import print_startup_banner, probe_igpu, probe_npu, EnvironmentError_
from hf_quant_bench.manifest import write_manifest
from hf_quant_bench.registry import ConversionContext, all_targets, run_target


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="HF repo id or local directory of the source model")
    p.add_argument("--out", default="outputs/model", help="output directory root for converted formats")
    p.add_argument("--list", action="store_true", help="list all targets and whether this environment can run them, then exit")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these target names")
    p.add_argument("--skip", nargs="*", default=[], help="skip these target names")
    p.add_argument(
        "--gguf-quant-types",
        nargs="*",
        default=["Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0"],
        help="GGUF quant types to produce (default: Q4_0 Q4_K_M Q5_K_M Q8_0)",
    )
    p.add_argument("--enable-qai-hub", action="store_true", help="opt into the Qualcomm AI Hub cloud target (off by default)")
    p.add_argument("--qai-hub-device", default="Snapdragon 8 Elite QRD", help="target device profile string for AI Hub")
    return p


def build_context(out_dir: Path, args: argparse.Namespace) -> ConversionContext:
    gpu = print_startup_banner(str(out_dir))
    npu_ok, _ = probe_npu()
    igpu_ok, _ = probe_igpu()
    return ConversionContext(
        gpu=gpu,
        npu_available=npu_ok,
        igpu_available=igpu_ok,
        device="cuda" if gpu.available else "cpu",
        extra={
            "gguf_quant_types": args.gguf_quant_types,
            "enable_qai_hub": args.enable_qai_hub,
            "qai_hub_device": args.qai_hub_device,
        },
    )


def cmd_list(ctx: ConversionContext) -> None:
    print(f"{'target':<20} {'runnable now':<14} reason")
    print("-" * 90)
    for t in sorted(all_targets(), key=lambda t: t.name):
        try:
            ok, reason = t.check(ctx)
        except Exception as e:
            ok, reason = False, f"check() raised: {e}"
        flag = "YES" if ok else "no"
        print(f"{t.name:<20} {flag:<14} {reason}")

    if targets.import_errors:
        print("\nTargets that failed to import (treated as always-skip):")
        for mod, err in targets.import_errors.items():
            print(f"  {mod}: {err.splitlines()[0]}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_root = Path(args.out)

    try:
        ctx = build_context(out_root, args)
    except EnvironmentError_ as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.list:
        cmd_list(ctx)
        return 0

    if not args.model:
        print("error: --model is required unless --list is passed", file=sys.stderr)
        return 2

    selected = all_targets()
    if args.only:
        selected = [t for t in selected if t.name in args.only]
    selected = [t for t in selected if t.name not in args.skip]

    results = {}
    out_root.mkdir(parents=True, exist_ok=True)

    for t in selected:
        target_out = out_root / t.name
        print(f"\n{'=' * 72}\n[convert] target: {t.name}\n{'=' * 72}")
        result = run_target(t, args.model, target_out, ctx)
        results[t.name] = result
        if result.status == "ok":
            size_mb = (result.size_bytes or 0) / (1024**2)
            print(f"[convert] {t.name}: OK  ({size_mb:.1f} MB, {result.wall_time_s:.1f}s)")
        elif result.status == "skipped":
            print(f"[convert] {t.name}: SKIPPED — {result.reason}")
        else:
            print(f"[convert] {t.name}: FAILED — {result.reason.splitlines()[0] if result.reason else 'unknown error'}")

        # Written after every target, not just at the end: a target's own
        # conversion process can be killed by the OS (OOM, a driver reset,
        # etc.) without raising a catchable Python exception, which would
        # otherwise silently drop every result already recorded this run.
        write_manifest(out_root / "manifest.json", args.model, results)

    ok_count = sum(1 for r in results.values() if r.status == "ok")
    print(f"\n[convert] done: {ok_count}/{len(results)} targets produced output. See {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
