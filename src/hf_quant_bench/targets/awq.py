"""AWQ 4-bit, with a fallback path.

Upstream `autoawq` (the reference AWQ implementation most people mean by
"AWQ") was archived on GitHub in 2025 and is unmaintained. It may still
install and work fine on many stacks, but it is not guaranteed to build
against a Blackwell-era torch/CUDA combination, and nobody is fixing it if it
doesn't.

Strategy:
  1. Try `autoawq` first, since it's what most tooling/docs assume "AWQ" means.
  2. If that import or the quantization call fails for any reason, fall back
     to `llm-compressor` (Neural Magic's actively maintained library), which
     implements an AWQ-equivalent activation-aware quantization scheme.
  3. Record which path was actually used in the manifest via
     `extra["backend"]`, so the benchmark report can be honest about which
     implementation produced the "AWQ" numbers.
"""

from __future__ import annotations

from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register

N_CALIBRATION_SAMPLES = 128


def check(ctx: ConversionContext) -> tuple[bool, str]:
    if not ctx.gpu.available:
        return False, "AWQ requires a CUDA GPU; none detected"
    has_autoawq = False
    has_llmcompressor = False
    try:
        import awq  # noqa: F401
        has_autoawq = True
    except ImportError:
        pass
    try:
        import llmcompressor  # noqa: F401
        has_llmcompressor = True
    except ImportError:
        pass
    if not has_autoawq and not has_llmcompressor:
        return False, (
            "neither autoawq nor llm-compressor is installed. "
            "pip install autoawq (unmaintained since 2025, may not build on Blackwell) "
            "or pip install llmcompressor (recommended fallback)."
        )
    return True, "ok"


def _calibration_texts() -> list[str]:
    try:
        from datasets import load_dataset

        # "wikitext" (the short legacy id) 404s on current `datasets` releases --
        # the dataset moved to the canonical "Salesforce/wikitext" repo id. This
        # silently fell back to the synthetic filler below for a while before
        # that was caught (see gptq.py, which had the same bug but at least
        # printed on fallback) -- printing here too now, since a silent
        # calibration-quality regression is exactly the kind of thing this
        # project exists to catch, not cause.
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 200][:N_CALIBRATION_SAMPLES]
        if texts:
            return texts
    except Exception as e:
        print(f"[awq] could not load wikitext2 calibration set ({e}); using synthetic fallback")
    base = "Calibration text for activation-aware weight quantization of causal language models. "
    return [base * 8 for _ in range(N_CALIBRATION_SAMPLES)]


def _try_autoawq(model_id: str, out_dir: Path) -> ConversionResult:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    print(f"[awq] quantizing '{model_id}' via autoawq (backend=autoawq)")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=_calibration_texts())

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    return ConversionResult(
        status="ok",
        output_path=str(out_dir),
        extra={"backend": "autoawq", "device_used": "cuda", "bits": 4, "group_size": 128},
    )


def _try_llmcompressor(model_id: str, out_dir: Path) -> ConversionResult:
    """llmcompressor's recipe API restructured between minor versions: as of
    0.13.0, `oneshot` moved from `llmcompressor.transformers` to the
    top-level `llmcompressor` package, and `AWQModifier` no longer takes
    bits/group_size/symmetric directly -- it now splits into an activation
    "transform" step plus a separate QuantizationModifier driven by a named
    preset `scheme` string (e.g. "W4A16"). The deprecated `AWQModifier(**kwargs)`
    shim still does this split for you, but only accepts the new kwarg names.
    `dataset` also now requires a `datasets.Dataset`, not a plain list[str].
    """
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[awq] quantizing '{model_id}' via llm-compressor (backend=llmcompressor, AWQ fallback)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # bfloat16, not float16: with real (diverse) wikitext2 calibration text,
    # AWQ's activation-aware scale grid search hit "No finite loss ... NaN
    # values" under fp16 -- its ~65504 max magnitude is narrow enough that
    # some real activations overflow it, where the narrower-but-uniform
    # synthetic calibration text this project used before never did. bf16
    # has fp32's exponent range (just less mantissa precision), which avoids
    # the overflow without needing a different calibration set.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})

    recipe = AWQModifier(scheme="W4A16")
    calib = _calibration_texts()
    calib_dataset = Dataset.from_dict({"text": calib})

    out_dir.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_dataset,
        recipe=recipe,
        output_dir=str(out_dir),
        max_seq_length=512,
        num_calibration_samples=len(calib),
    )

    return ConversionResult(
        status="ok",
        output_path=str(out_dir),
        extra={"backend": "llmcompressor", "device_used": "cuda", "scheme": "W4A16"},
    )


def convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    try:
        import awq  # noqa: F401
        try:
            return _try_autoawq(model_id, out_dir)
        except Exception as e:
            print(f"[awq] autoawq path failed ({type(e).__name__}: {e}); falling back to llm-compressor")
    except ImportError:
        print("[awq] autoawq not installed; going straight to llm-compressor fallback")

    try:
        import llmcompressor  # noqa: F401
    except ImportError:
        return ConversionResult(
            status="failed",
            reason="autoawq failed/unavailable and llm-compressor is not installed as a fallback",
        )
    return _try_llmcompressor(model_id, out_dir)


register(
    Target(
        name="awq-4bit",
        description="AWQ 4-bit via autoawq, falling back to llm-compressor if autoawq fails",
        check=check,
        convert=convert,
        requires_gpu=True,
    )
)
