"""OpenVINO IR export (FP16 and INT8) via optimum-intel.

Conversion itself is CPU-only and does not require the iGPU or NPU to be
reachable — optimum-intel exports the IR on CPU regardless of what inference
device will eventually run it. What DOES depend on hardware is which device
the benchmark step can later target:
  - CPU: always available.
  - iGPU: reachable from WSL2 via /dev/dri passthrough in recent WSL builds,
    detected at runtime (see env_check.probe_igpu).
  - NPU: essentially never reachable from WSL2 as of current WSL releases —
    NPU drivers are Windows-side and not passed through. Detected at runtime
    (env_check.probe_npu) and skipped with a clear message if absent, rather
    than failing. Run the NPU target natively on Windows instead (see README).

This target is optional and entirely runtime-detected: if `openvino` /
`optimum-intel` aren't installed, it's skipped like any other missing
dependency, and the main WSL flow never blocks on it.
"""

from __future__ import annotations

import json
from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register


def check(ctx: ConversionContext) -> tuple[bool, str]:
    try:
        import openvino  # noqa: F401
        from optimum.intel import OVModelForCausalLM  # noqa: F401
    except ImportError as e:
        return False, (
            f"missing dependency: {e} (pip install openvino optimum[openvino]). "
            "This target is optional — skip is expected if you haven't installed "
            "the OpenVINO extras."
        )
    return True, "ok"


def _write_device_note(out_dir: Path, ctx: ConversionContext) -> None:
    (out_dir / "TARGET_DEVICES.json").write_text(
        json.dumps(
            {
                "igpu_reachable_at_conversion_time": ctx.igpu_available,
                "npu_reachable_at_conversion_time": ctx.npu_available,
                "note": (
                    "Conversion ran on CPU regardless of these flags. These record "
                    "what was reachable when the IR was produced, as a hint for "
                    "which devices bench/loader.py can try at eval time. If "
                    "npu_reachable_at_conversion_time is false (expected on WSL2), "
                    "run the OpenVINO/NPU benchmark natively on Windows instead — "
                    "see README."
                ),
            },
            indent=2,
        )
    )


def convert_fp16(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[openvino-fp16] exporting '{model_id}' to OpenVINO IR (CPU-only conversion step)")

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    _write_device_note(out_dir, ctx)

    return ConversionResult(status="ok", output_path=str(out_dir), extra={"device_used": "cpu"})


def convert_int8(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[openvino-int8] exporting '{model_id}' to OpenVINO IR with INT8 weight compression (CPU-only)")

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    _write_device_note(out_dir, ctx)

    return ConversionResult(status="ok", output_path=str(out_dir), extra={"device_used": "cpu"})


register(
    Target(
        name="openvino-fp16",
        description="OpenVINO IR, FP16 (optional; targets CPU/iGPU/NPU inference, conversion is CPU-only)",
        check=check,
        convert=convert_fp16,
        requires_gpu=False,
        cpu_only_ok=True,
    )
)

register(
    Target(
        name="openvino-int8",
        description="OpenVINO IR, INT8 weight-compressed (optional; conversion is CPU-only)",
        check=check,
        convert=convert_int8,
        requires_gpu=False,
        cpu_only_ok=True,
    )
)
