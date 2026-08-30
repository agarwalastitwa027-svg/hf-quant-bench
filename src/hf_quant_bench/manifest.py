from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from hf_quant_bench.registry import ConversionResult


@dataclasses.dataclass
class ManifestEntry:
    target: str
    status: str
    output_path: str | None
    size_bytes: int | None
    wall_time_s: float | None
    reason: str | None
    extra: dict


def build_manifest(model_id: str, results: dict[str, ConversionResult]) -> dict:
    entries = []
    for name, r in results.items():
        entries.append(
            dataclasses.asdict(
                ManifestEntry(
                    target=name,
                    status=r.status,
                    output_path=r.output_path,
                    size_bytes=r.size_bytes,
                    wall_time_s=r.wall_time_s,
                    reason=r.reason,
                    extra=r.extra,
                )
            )
        )
    return {
        "model_id": model_id,
        "generated_at_unix": time.time(),
        "targets": entries,
    }


def write_manifest(path: Path, model_id: str, results: dict[str, ConversionResult]) -> None:
    """Merge with any existing manifest.json rather than overwrite it.

    `convert.py` is commonly invoked multiple times against the same --out
    directory with different --only subsets (e.g. once for CPU-only targets,
    once for GPU targets that need to run separately). Without merging, each
    invocation would silently wipe out every previously-recorded target.
    Entries for the same target name are replaced by the newer result;
    entries for targets not touched by this run are preserved as-is.
    """
    existing_targets: dict[str, dict] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            for entry in existing.get("targets", []):
                existing_targets[entry["target"]] = entry
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt/old manifest — start fresh rather than fail the run

    new_manifest = build_manifest(model_id, results)
    for entry in new_manifest["targets"]:
        existing_targets[entry["target"]] = entry

    merged = {
        "model_id": model_id,
        "generated_at_unix": time.time(),
        "targets": list(existing_targets.values()),
    }
    path.write_text(json.dumps(merged, indent=2))
    print(f"[manifest] wrote {path} ({len(merged['targets'])} target(s) total)")
