"""GPTQ 4-bit, calibrated, via the `gptqmodel` / `optimum` GPTQ pipeline.

GPTQ requires a calibration dataset (a small set of representative text
samples) to compute per-layer quantization error minimization. We use a slice
of wikitext2 by default — the standard calibration set in the GPTQ literature
— capped small (128 samples x 512 tokens) to keep this fast and within 8GB.

Library note: the `auto-gptq` project has been effectively superseded by
`gptqmodel` (actively maintained fork) for recent transformers versions. We
try `gptqmodel` first via `optimum`'s GPTQConfig path, since that's the
currently-recommended integration in transformers as of late 2025.
"""

from __future__ import annotations

from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register

N_CALIBRATION_SAMPLES = 128
CALIBRATION_SEQ_LEN = 512


def check(ctx: ConversionContext) -> tuple[bool, str]:
    if not ctx.gpu.available:
        return False, "GPTQ calibration requires a CUDA GPU; none detected"
    try:
        import optimum  # noqa: F401
    except ImportError as e:
        return False, f"missing dependency: {e} (pip install optimum)"
    try:
        import gptqmodel  # noqa: F401
    except ImportError:
        try:
            import auto_gptq  # noqa: F401
        except ImportError as e:
            return False, (
                f"missing dependency: neither gptqmodel nor auto-gptq is installed ({e}). "
                "pip install gptqmodel (preferred) or auto-gptq."
            )
    return True, "ok"


def _load_calibration_texts() -> list[str]:
    try:
        from datasets import load_dataset

        # "wikitext" (the short legacy id) 404s on current `datasets` releases --
        # the dataset moved to the canonical "Salesforce/wikitext" repo id.
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 200][:N_CALIBRATION_SAMPLES]
        if texts:
            return texts
    except Exception as e:
        print(f"[gptq] could not load wikitext2 calibration set ({e}); using synthetic fallback")

    # Deterministic synthetic fallback so this target still works fully offline.
    base = (
        "The quick brown fox jumps over the lazy dog while engineers discuss "
        "quantization tradeoffs for large language models running on consumer "
        "hardware with limited VRAM budgets and agentic tool-calling workloads. "
    )
    return [base * 8 for _ in range(N_CALIBRATION_SAMPLES)]


def convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    calib_texts = _load_calibration_texts()

    print(f"[gptq] calibrating '{model_id}' on {len(calib_texts)} samples (this uses the GPU)")
    gptq_config = GPTQConfig(
        bits=4,
        dataset=calib_texts,
        tokenizer=tokenizer,
        group_size=128,
        desc_act=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=gptq_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    del model
    torch.cuda.empty_cache()

    return ConversionResult(
        status="ok",
        output_path=str(out_dir),
        extra={
            "device_used": "cuda",
            "calibration_samples": len(calib_texts),
            "bits": 4,
            "group_size": 128,
        },
    )


register(
    Target(
        name="gptq-4bit",
        description="GPTQ 4-bit, calibrated on wikitext2 (GPU required for calibration)",
        check=check,
        convert=convert,
        requires_gpu=True,
    )
)
