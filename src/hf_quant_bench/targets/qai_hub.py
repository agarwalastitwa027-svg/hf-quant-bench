"""Qualcomm AI Hub compile job — optional, off by default, cloud-based.

There is no local Qualcomm device in this setup, so this target only submits a
compile job to Qualcomm AI Hub's cloud service and downloads the resulting
compiled artifact (e.g. a .bin/.dlc/context binary for a chosen target chipset)
rather than running anything locally. It requires:
  - `qai-hub` installed
  - an API token configured (`qai-hub configure --api_token ...`, or the
    QAI_HUB_API_TOKEN env var)
  - network access to Qualcomm's cloud

It is disabled unless `--enable-qai-hub` is explicitly passed on the CLI, since
it is the one target in this project that leaves the machine and may incur
cloud usage against your account.
"""

from __future__ import annotations

import json
from pathlib import Path

from hf_quant_bench.registry import ConversionContext, ConversionResult, Target, register


def check(ctx: ConversionContext) -> tuple[bool, str]:
    if not ctx.extra.get("enable_qai_hub", False):
        return False, "disabled by default; pass --enable-qai-hub to opt in (this submits a cloud job)"
    try:
        import qai_hub  # noqa: F401
    except ImportError as e:
        return False, f"missing dependency: {e} (pip install qai-hub)"
    import os

    if not os.environ.get("QAI_HUB_API_TOKEN") and not Path.home().joinpath(".qai_hub/client.ini").exists():
        return False, (
            "no Qualcomm AI Hub API token found. Set QAI_HUB_API_TOKEN or run "
            "`qai-hub configure --api_token <token>` once."
        )
    return True, "ok"


def convert(model_id: str, out_dir: Path, ctx: ConversionContext) -> ConversionResult:
    import qai_hub as hub
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    target_device = ctx.extra.get("qai_hub_device", "Snapdragon 8 Elite QRD")

    print(f"[qai-hub] tracing '{model_id}' for cloud compilation (this leaves the machine)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    dummy_input = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    traced = torch.jit.trace(model, dummy_input, strict=False)

    print(f"[qai-hub] submitting compile job for device profile: {target_device}")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(target_device),
        input_specs={"input_ids": (tuple(dummy_input.shape), "int64")},
    )
    compile_job.wait()
    compiled_model = compile_job.get_target_model()

    out_path = out_dir / "model_qai_hub.bin"
    compiled_model.download(str(out_path))

    (out_dir / "qai_hub_job.json").write_text(
        json.dumps({"job_id": compile_job.job_id, "device": target_device}, indent=2)
    )

    return ConversionResult(
        status="ok",
        output_path=str(out_dir),
        extra={"device_used": "cloud (Qualcomm AI Hub)", "target_device": target_device},
    )


register(
    Target(
        name="qai-hub",
        description="Qualcomm AI Hub cloud compile (optional, off by default, requires --enable-qai-hub)",
        check=check,
        convert=convert,
        requires_gpu=False,
        cpu_only_ok=True,
    )
)
