"""bitsandbytes NF4 and INT8 targets.

Both are saved as a config + the original fp16 weights are NOT re-quantized
into a separate on-disk format by bitsandbytes — bnb quantizes at load time
inside `from_pretrained`. What we persist here is:
  1. the fp16 base weights (reused from the fp16 target if present, else
     downloaded fresh), and
  2. a `bnb_config.json` capturing the exact quantization config used, plus a
     small marker file, so the benchmark loader can reconstruct the same
     quantized model deterministically at eval time.

This mirrors how bnb is actually used in production: you don't get a
standalone "NF4 checkpoint" on disk, you get a config that tells
`from_pretrained(..., quantization_config=...)` how to quantize on load. We
still report on-disk size (of the fp16 weights it depends on) so the size
comparison in results.md stays fair — the on_disk_size column will be
annotated as "shared with fp16 base" for this reason.
"""

from __future__ import annotations

import json
from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register


def _check_common(ctx: ConversionContext) -> tuple[bool, str]:
    if not ctx.gpu.available:
        return False, "bitsandbytes NF4/INT8 require a CUDA GPU; none detected"
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        return False, f"missing dependency: {e} (pip install bitsandbytes)"
    try:
        import torch
        cap = ctx.gpu.capability
        if cap and cap >= (12, 0) and ctx.gpu.capability_supported_by_torch is False:
            return False, (
                "GPU is Blackwell (sm_120) but installed torch has no kernels for it; "
                "see the startup banner for the fix. Refusing to load bnb on this stack "
                "rather than risk a silent wrong-result run."
            )
    except Exception:
        pass
    return True, "ok"


def _write_config(out_dir: Path, model_id: str, quant_type: str, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bnb_config.json").write_text(
        json.dumps({"base_model": model_id, "quant_type": quant_type, "config": cfg}, indent=2)
    )
    (out_dir / "README_FORMAT.md").write_text(
        f"# bitsandbytes {quant_type}\n\n"
        "bitsandbytes quantizes weights at load time, not at save time. This "
        "directory holds only the bnb_config.json describing how to load the "
        f"base model ('{model_id}') with `quantization_config=...` to reproduce "
        f"this {quant_type} setup. The benchmark loader (bench/loader.py) reads "
        "this file. On-disk size for cost comparisons should be read against "
        "the fp16 base checkpoint size, since bnb does not persist quantized "
        "weights separately.\n"
    )


def check_nf4(ctx: ConversionContext) -> tuple[bool, str]:
    return _check_common(ctx)


def convert_nf4(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    cfg = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    }
    # Smoke-test that the config actually loads on this GPU before declaring success.
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"[bnb-nf4] load-testing '{model_id}' with NF4 config on GPU")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map={"": 0}
    )
    del model
    torch.cuda.empty_cache()

    _write_config(out_dir, model_id, "nf4", cfg)
    return ConversionResult(status="ok", output_path=str(out_dir), extra={"device_used": "cuda"})


def check_int8(ctx: ConversionContext) -> tuple[bool, str]:
    return _check_common(ctx)


def convert_int8(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cfg = {"load_in_8bit": True}
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    print(f"[bnb-int8] load-testing '{model_id}' with INT8 config on GPU")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map={"": 0}
    )
    del model
    torch.cuda.empty_cache()

    _write_config(out_dir, model_id, "int8", cfg)
    return ConversionResult(status="ok", output_path=str(out_dir), extra={"device_used": "cuda"})


register(
    Target(
        name="bnb-nf4",
        description="bitsandbytes NF4 4-bit (load-time quantization, GPU required)",
        check=check_nf4,
        convert=convert_nf4,
        requires_gpu=True,
    )
)

register(
    Target(
        name="bnb-int8",
        description="bitsandbytes INT8 (load-time quantization, GPU required)",
        check=check_int8,
        convert=convert_int8,
        requires_gpu=True,
    )
)
