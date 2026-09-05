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
import time
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

    # torch.jit.trace can't infer a type for HF's ModelOutput dataclasses
    # (e.g. CausalLMOutputWithPast). Forcing config.return_dict=False isn't
    # enough on transformers 5.14.1: internal submodules still return
    # ModelOutput objects while outer code accesses them as tuples,
    # producing a different, equally untraceable error. A thin wrapper that
    # calls the model normally and extracts the plain logits tensor sidesteps
    # both issues.
    class LogitsOnly(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.m(input_ids).logits

    wrapped = LogitsOnly(model)
    wrapped.eval()

    dummy_input = tokenizer("Hello world", return_tensors="pt")["input_ids"]

    # AI Hub rejects TorchScript (.pt) uploads for this model with "Failed to
    # upgrade the exported ONNX model to opset 21" and its own error message
    # points at torch.export instead -- export a PyTorch ExportedProgram
    # (.pt2) and upload that in place of a jit.trace() TorchScript module.
    with torch.no_grad():
        exported_program = torch.export.export(wrapped, (dummy_input,), strict=False)
    pt2_path = out_dir / "traced_model.pt2"
    torch.export.save(exported_program, str(pt2_path))

    print(f"[qai-hub] submitting compile job for device profile: {target_device}")
    compile_job = hub.submit_compile_job(
        model=str(pt2_path),
        device=hub.Device(target_device),
        input_specs={"input_ids": (tuple(dummy_input.shape), "int64")},
        # --truncate_64bit_io: Hexagon/NPU backends don't natively support
        #   int64 I/O; without it the job fails with "Must use
        #   --truncate_64bit_io when input tensors have type int64."
        # --target_runtime qnn_context_binary: the default (LiteRT/tflite)
        #   rejects this model with "Model is too large for the LiteRT model
        #   format", and a QNN context binary is this target's intended
        #   output anyway (a .bin for the chosen Snapdragon device).
        options="--truncate_64bit_io --target_runtime qnn_context_binary",
    )
    # Written immediately (not just on success): the job_id/url are the only
    # way to inspect or resume a job from the AI Hub dashboard if a later
    # step here fails after the (slow, ~5-10 min for a multi-GB fp32 trace)
    # upload has already completed.
    (out_dir / "qai_hub_job.json").write_text(
        json.dumps({"job_id": compile_job.job_id, "url": compile_job.url, "device": target_device}, indent=2)
    )
    print(f"[qai-hub] job submitted: {compile_job.job_id} ({compile_job.url})")
    pt2_path.unlink(missing_ok=True)  # already uploaded; ~6GB local copy no longer needed

    compile_job.wait()
    status = compile_job.get_status()
    if not status.success:
        raise RuntimeError(f"qai-hub compile job did not succeed: {status.message} ({compile_job.url})")

    compiled_model = compile_job.get_target_model()

    out_path = out_dir / "model_qai_hub.bin"
    # The target model artifact can lag a few seconds behind the job being
    # reported as successful (observed: "Model file has not yet been
    # uploaded" from an immediate download() right after wait() returns).
    # Retry with backoff instead of failing on that transient state.
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            compiled_model.download(str(out_path))
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 - retrying on any download failure, re-raised below if exhausted
            last_err = e
            time.sleep(10 * (attempt + 1))
    if last_err is not None:
        raise RuntimeError(f"qai-hub: target model never became downloadable after job success: {last_err}")

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
