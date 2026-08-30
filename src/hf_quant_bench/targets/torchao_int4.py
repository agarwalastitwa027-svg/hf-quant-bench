"""torchao int4 weight-only quantization.

torchao's `quantize_` API mutates a model in place, tagging Linear layers with
an int4 tensor subclass. Like bitsandbytes, there isn't a separate on-disk
"int4 weight format" distinct from a torch state dict — we save the quantized
model's state dict directly, which torchao supports via safe torch.save
(safetensors doesn't yet support tensor subclasses as of the torchao versions
this targets, so we use torch.save here and say so explicitly).
"""

from __future__ import annotations

import json
from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register


def check(ctx: ConversionContext) -> tuple[bool, str]:
    if not ctx.gpu.available:
        return False, "torchao int4 quantization path targets CUDA; none detected"
    try:
        import torchao  # noqa: F401
    except ImportError as e:
        return False, f"missing dependency: {e} (pip install torchao)"
    return True, "ok"


def convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchao.quantization import quantize_, Int4WeightOnlyConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"[torchao-int4] loading '{model_id}' on GPU for int4 weight-only quantization")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    quantize_(model, Int4WeightOnlyConfig(group_size=128))

    # Tensor subclasses aren't safetensors-serializable in the torchao versions
    # this targets, so we torch.save the state dict and record that explicitly.
    weights_path = out_dir / "model_torchao_int4.pt"
    torch.save(model.state_dict(), weights_path)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "torchao_config.json").write_text(
        json.dumps({"base_model": model_id, "method": "Int4WeightOnlyConfig", "group_size": 128}, indent=2)
    )
    (out_dir / "README_FORMAT.md").write_text(
        "# torchao int4\n\n"
        "Saved as a raw torch.save state dict (model_torchao_int4.pt), not "
        "safetensors — torchao's int4 tensor subclass is not yet "
        "safetensors-serializable in the version this project targets. Load "
        "with the same architecture class, torchao's quantize_ config above, "
        "then `load_state_dict`. See bench/loader.py for the exact reload path.\n"
    )

    del model
    torch.cuda.empty_cache()

    return ConversionResult(
        status="ok",
        output_path=str(out_dir),
        extra={"device_used": "cuda", "serialization": "torch.save (not safetensors)"},
    )


register(
    Target(
        name="torchao-int4",
        description="torchao int4 weight-only quantization (GPU required)",
        check=check,
        convert=convert,
        requires_gpu=True,
    )
)
