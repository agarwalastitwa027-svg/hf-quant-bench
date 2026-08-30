"""Pluggable conversion-target registry.

Each target declares:
  - a unique `name`
  - a `check()` function that returns (ok: bool, reason: str) WITHOUT importing
    heavy dependencies at module load time (so `--list` is fast and never
    crashes just because e.g. autoawq isn't installed)
  - a `convert()` function with signature
        convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult
    that does the actual work, importing its dependencies lazily inside the
    function body.

A target that fails its `check()` is skipped with the recorded reason and the
run continues. A target whose `convert()` raises is caught by the runner,
recorded as a failure, and the run continues — one broken target must never
kill the whole conversion sweep.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from hf_quant_bench.env_check import GpuInfo


@dataclasses.dataclass
class ConversionContext:
    gpu: GpuInfo
    npu_available: bool
    igpu_available: bool
    device: str  # "cuda" or "cpu"
    extra: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ConversionResult:
    status: str  # "ok" | "skipped" | "failed"
    output_path: Optional[str] = None
    size_bytes: Optional[int] = None
    wall_time_s: Optional[float] = None
    reason: Optional[str] = None
    extra: dict = dataclasses.field(default_factory=dict)


CheckFn = Callable[[ConversionContext], tuple[bool, str]]
ConvertFn = Callable[[str, Path, ConversionContext], ConversionResult]


@dataclasses.dataclass
class Target:
    name: str
    description: str
    check: CheckFn
    convert: ConvertFn
    requires_gpu: bool = False
    cpu_only_ok: bool = False  # if True, this target is fine (or preferred) on CPU


_REGISTRY: dict[str, Target] = {}


def register(target: Target) -> None:
    if target.name in _REGISTRY:
        raise ValueError(f"target '{target.name}' already registered")
    _REGISTRY[target.name] = target


def all_targets() -> list[Target]:
    return list(_REGISTRY.values())


def get(name: str) -> Target:
    return _REGISTRY[name]


def names() -> list[str]:
    return list(_REGISTRY.keys())


def dir_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def run_target(target: Target, model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    """Run one target's check + convert, never letting an exception escape."""
    try:
        ok, reason = target.check(ctx)
    except Exception as e:  # a broken check() must not kill the sweep either
        return ConversionResult(status="skipped", reason=f"check() raised: {e}")

    if not ok:
        return ConversionResult(status="skipped", reason=reason)

    start = time.monotonic()
    try:
        result = target.convert(model_id, out_dir, ctx)
        if result.wall_time_s is None:
            result.wall_time_s = time.monotonic() - start
        if result.status == "ok" and result.output_path and result.size_bytes is None:
            p = Path(result.output_path)
            if p.exists():
                result.size_bytes = dir_size_bytes(p)
        return result
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        return ConversionResult(
            status="failed",
            wall_time_s=time.monotonic() - start,
            reason=f"{type(e).__name__}: {e}\n{tb}",
        )
