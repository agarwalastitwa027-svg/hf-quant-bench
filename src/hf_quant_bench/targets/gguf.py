"""GGUF via llama.cpp, with a configurable list of quant types.

This target shells out to a locally built llama.cpp checkout rather than
vendoring/rebuilding it, because llama.cpp's Python bindings churn constantly
and the CLI tools (`convert_hf_to_gguf.py` + `llama-quantize`) are the most
stable interface. Point `LLAMA_CPP_DIR` at a cloned+built llama.cpp repo (see
README for the one-time build command).

This whole target is CPU-only (GGUF conversion and quantization run on CPU by
design in llama.cpp) — we never touch the GPU here, and say so.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register

DEFAULT_QUANT_TYPES = ["Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0"]


def _llama_cpp_dir() -> Path | None:
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    return None


def _find_convert_script(llama_dir: Path) -> Path | None:
    for candidate in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py", "convert.py"]:
        p = llama_dir / candidate
        if p.exists():
            return p
    return None


def _find_quantize_bin(llama_dir: Path) -> Path | None:
    for candidate in ["build/bin/llama-quantize", "llama-quantize", "build/bin/quantize", "quantize"]:
        p = llama_dir / candidate
        if p.exists() and os.access(p, os.X_OK):
            return p
    return None


def check(ctx: ConversionContext) -> tuple[bool, str]:
    llama_dir = _llama_cpp_dir()
    if llama_dir is None:
        return False, (
            "LLAMA_CPP_DIR is not set or doesn't point to a directory. Clone+build "
            "llama.cpp and export LLAMA_CPP_DIR=/path/to/llama.cpp (see README)."
        )
    if _find_convert_script(llama_dir) is None:
        return False, f"no convert_hf_to_gguf.py found under {llama_dir}"
    if _find_quantize_bin(llama_dir) is None:
        return False, f"no built llama-quantize binary found under {llama_dir} (did you run `cmake --build`?)"
    try:
        import gguf  # noqa: F401  (the gguf python package convert scripts depend on)
    except ImportError as e:
        return False, f"missing dependency: {e} (pip install gguf)"
    return True, "ok"


def convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    quant_types: list[str] = ctx.extra.get("gguf_quant_types", DEFAULT_QUANT_TYPES)
    llama_dir = _llama_cpp_dir()
    convert_script = _find_convert_script(llama_dir)
    quantize_bin = _find_quantize_bin(llama_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp16_gguf = out_dir / "model-f16.gguf"

    print(f"[gguf] converting '{model_id}' to intermediate f16 GGUF (CPU-only step)")
    cmd = [
        "python3", str(convert_script),
        model_id if Path(model_id).exists() else _resolve_local_snapshot(model_id),
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ]
    subprocess.run(cmd, check=True)

    produced = {}
    for qt in quant_types:
        out_path = out_dir / f"model-{qt}.gguf"
        print(f"[gguf] quantizing to {qt} (CPU-only)")
        subprocess.run([str(quantize_bin), str(fp16_gguf), str(out_path), qt], check=True)
        produced[qt] = str(out_path)

    return ConversionResult(
        status="ok",
        output_path=str(out_dir),
        extra={"device_used": "cpu", "quant_types": produced, "intermediate_f16": str(fp16_gguf)},
    )


def _resolve_local_snapshot(model_id: str) -> str:
    """convert_hf_to_gguf.py needs a local directory, not a repo id. Download
    the snapshot via huggingface_hub if `model_id` isn't already a local path.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id)


register(
    Target(
        name="gguf",
        description=(
            "GGUF via llama.cpp CLI (convert + llama-quantize), CPU-only. "
            "Set LLAMA_CPP_DIR to a built checkout; quant types configurable via --gguf-quant-types."
        ),
        check=check,
        convert=convert,
        requires_gpu=False,
        cpu_only_ok=True,
    )
)
