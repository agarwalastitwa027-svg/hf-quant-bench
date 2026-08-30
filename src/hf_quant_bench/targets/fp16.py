"""fp16 safetensors baseline — the reference every other format is compared to.

This is a straight HF `from_pretrained` + `save_pretrained(safe_serialization=True)`
re-save in float16. If the source checkpoint is already fp16/bf16 this is
mostly an I/O pass; if it's fp32 it also halves size on disk.

VRAM note: we load on CPU for the re-save. There is no reason to touch the GPU
just to change a dtype and re-serialize — this is a pure CPU/RAM operation, so
we do it on CPU and say so, per the project's rule about not assuming GPU use
where it isn't needed.
"""

from __future__ import annotations

from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register


def check(ctx: ConversionContext) -> tuple[bool, str]:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        return False, f"missing dependency: {e}"
    return True, "ok"


def convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fp16] loading '{model_id}' on CPU (this is a dtype re-save, no GPU needed)")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    del model
    return ConversionResult(status="ok", output_path=str(out_dir), extra={"device_used": "cpu"})


register(
    Target(
        name="fp16",
        description="fp16 safetensors baseline (CPU re-save, no GPU required)",
        check=check,
        convert=convert,
        requires_gpu=False,
        cpu_only_ok=True,
    )
)
