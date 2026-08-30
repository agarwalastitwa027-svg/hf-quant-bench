"""ONNX export (fp32/fp16) plus dynamic INT8 ONNX quantization.

Both steps run via `optimum.onnxruntime` / `onnxruntime.quantization` and are
CPU-only work — ONNX graph export and post-training dynamic quantization do
not need a GPU, so we run them on CPU and say so explicitly, per the project
rule about not reaching for the GPU when it isn't needed.
"""

from __future__ import annotations

from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register


def _shim_get_parameter_dtype() -> None:
    """Compatibility shim for a genuine upstream gap, not a bug in this project.

    `transformers` 5.x removed `transformers.modeling_utils.get_parameter_dtype`.
    `optimum` (as of 2.3.0, the latest release at the time this was written)
    still imports it directly in `optimum/exporters/onnx/convert.py` for one
    trivial line: `get_parameter_dtype(model)`. Since this project pins
    `transformers==5.14.1` elsewhere (required for gptqmodel/AWQ
    compatibility -- see gptq.py/awq.py), optimum's ONNX export is otherwise
    dead on arrival with an ImportError in this environment. The original
    function is a one-liner; re-adding it is a safe, minimal patch, not a
    workaround for anything this project got wrong.
    """
    import transformers.modeling_utils as mu

    if not hasattr(mu, "get_parameter_dtype"):
        def get_parameter_dtype(module):
            return next(module.parameters()).dtype

        mu.get_parameter_dtype = get_parameter_dtype


def check(ctx: ConversionContext) -> tuple[bool, str]:
    try:
        import optimum.onnxruntime  # noqa: F401
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        return False, f"missing dependency: {e} (pip install optimum[onnxruntime] onnx onnxruntime)"
    return True, "ok"


def _export_onnx(model_id: str, out_dir: Path) -> str:
    """Export to ONNX via optimum's lower-level main_export.

    Two deliberate choices here, both learned the hard way against an 8GB
    VRAM / ~24GB RAM WSL2 box:

    - `monolith=True`: optimum's default exports a separate
      decoder_model.onnx AND decoder_with_past_model.onnx (two full traces
      alive at once). That alone pushed peak RSS past 20GB tracing a 1.5B
      model — monolith mode traces one combined graph instead.
    - `no_post_process=True` from the start rather than as a fallback:
      optimum's post-export graph cleanup (tied-weight de-duplication) hits
      a protobuf serialization bug on this model/optimum-version
      combination (`EncodeError: Failed to serialize proto`). Skipping it
      still yields a valid, runnable ONNX graph, just without that extra
      optimization pass — so it's not worth paying the first attempt's
      memory cost only to fall back anyway.
    """
    _shim_get_parameter_dtype()
    from optimum.exporters.onnx import main_export

    main_export(model_id, output=str(out_dir), no_post_process=True, monolith=True)
    return "post_process=False, monolith=True"


def convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[onnx] exporting '{model_id}' to ONNX on CPU")

    export_mode = _export_onnx(model_id, out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    return ConversionResult(
        status="ok", output_path=str(out_dir), extra={"device_used": "cpu", "export_mode": export_mode}
    )


def check_int8(ctx: ConversionContext) -> tuple[bool, str]:
    ok, reason = check(ctx)
    if not ok:
        return ok, reason
    try:
        from onnxruntime.quantization import quantize_dynamic  # noqa: F401
    except ImportError as e:
        return False, f"missing dependency: {e}"
    return True, "ok"


def convert_int8(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    """Depends on the plain ONNX export existing; if it isn't already in
    outputs/onnx-fp32, we export it fresh into a temp location first.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = out_dir.parent / "onnx-fp32"

    if not (base_dir / "model.onnx").exists():
        print(f"[onnx-int8] no existing fp32 ONNX export found at {base_dir}, exporting fresh (CPU)")
        base_dir.mkdir(parents=True, exist_ok=True)
        _export_onnx(model_id, base_dir)
        AutoTokenizer.from_pretrained(model_id).save_pretrained(base_dir)

    src = base_dir / "model.onnx"
    dst = out_dir / "model.onnx"
    print(f"[onnx-int8] dynamic INT8 quantization {src} -> {dst} (CPU-only)")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)

    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    return ConversionResult(status="ok", output_path=str(out_dir), extra={"device_used": "cpu"})


register(
    Target(
        name="onnx-fp32",
        description="ONNX export via optimum (CPU-only)",
        check=check,
        convert=convert,
        requires_gpu=False,
        cpu_only_ok=True,
    )
)

register(
    Target(
        name="onnx-int8-dynamic",
        description="Dynamic INT8 ONNX quantization via onnxruntime (CPU-only)",
        check=check_int8,
        convert=convert_int8,
        requires_gpu=False,
        cpu_only_ok=True,
    )
)
