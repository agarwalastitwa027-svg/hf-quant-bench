"""Environment / hardware verification.

This module is imported before any heavy work happens. Its job is to fail
LOUDLY, with a readable message, at startup rather than letting a bad
torch/bitsandbytes build die inside a CUDA kernel launch three minutes into a
conversion.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import subprocess
import sys

# RTX 5070 / Blackwell = sm_120. Anything below this compute capability is a
# different architecture family; anything AT or above sm_120 built against an
# old torch (<2.5 with no Blackwell kernels compiled in) will typically either
# refuse to run or silently fall back to something broken. We check the
# *reported* capability against what the installed torch build claims to
# support, via torch.cuda.get_arch_list().
BLACKWELL_CC = (12, 0)


class EnvironmentError_(RuntimeError):
    """Raised when the environment cannot safely run the requested target."""


@dataclasses.dataclass
class GpuInfo:
    available: bool
    name: str | None = None
    capability: tuple[int, int] | None = None
    total_vram_gb: float | None = None
    torch_arch_list: list[str] = dataclasses.field(default_factory=list)
    capability_supported_by_torch: bool | None = None


def _is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def check_path_not_on_windows_mount(path: str) -> str | None:
    """Warn if a path lives under /mnt/c (or similar) — slow 9p filesystem."""
    abspath = os.path.abspath(path)
    if abspath.startswith("/mnt/"):
        return (
            f"WARNING: '{abspath}' is on a Windows-mounted filesystem (/mnt/...). "
            "Reading/writing model weights here is much slower than the native "
            "WSL ext4 filesystem. Keep outputs under your WSL home directory "
            "(e.g. ~/projects/hf-quant-bench/outputs)."
        )
    return None


def probe_gpu() -> GpuInfo:
    """Returns GpuInfo(available=False) if torch isn't installed at all —
    that's a mundane, self-evident state (every per-target check() will
    independently report "missing dependency: torch"), not the dangerous
    silent-failure case this module exists to catch. We only raise loudly
    for the Blackwell/kernel-mismatch case in verify_blackwell_or_die, where
    torch IS installed, reports a GPU, but lacks kernels for it — that's the
    scenario that would otherwise die deep inside a kernel launch instead of
    at startup.
    """
    try:
        import torch
    except ImportError:
        return GpuInfo(available=False)

    available = torch.cuda.is_available()
    if not available:
        return GpuInfo(available=False)

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024**3)

    arch_list = []
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        pass

    cap_str = f"sm_{cap[0]}{cap[1]}"
    supported = any(cap_str in a for a in arch_list)

    return GpuInfo(
        available=True,
        name=name,
        capability=cap,
        total_vram_gb=total_vram_gb,
        torch_arch_list=arch_list,
        capability_supported_by_torch=supported,
    )


def verify_blackwell_or_die(gpu: GpuInfo, *, strict: bool = True) -> None:
    """If a GPU is present, verify the installed torch build actually has
    kernels for its compute capability. This is the check that turns a cryptic
    'no kernel image is available for execution on the device' error (which
    happens deep inside a matmul, mid-run) into a readable message at
    startup.
    """
    if not gpu.available:
        try:
            import torch  # noqa: F401
            torch_installed = True
        except ImportError:
            torch_installed = False

        if not torch_installed:
            print(
                "[env] torch is not installed — GPU-only targets will report 'missing "
                "dependency' and be skipped individually. Install it with:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "(check pytorch.org/get-started/locally for the current cu12x tag; do "
                "NOT install a CPU-only wheel if you want GPU targets)."
            )
        else:
            print(
                "[env] torch.cuda.is_available() = False — no GPU visible to torch. "
                "GPU-only targets (bitsandbytes, GPTQ, AWQ, torchao-int4 on-device "
                "benchmarking) will be skipped. If you're on WSL2 and expect a GPU, "
                "confirm `nvidia-smi` works in this same shell — you do NOT install "
                "an NVIDIA driver inside WSL, only the CUDA-enabled torch wheel; the "
                "driver is the Windows host driver passed through automatically."
            )
        return

    print(f"[env] torch.cuda.is_available() = True")
    print(f"[env] GPU: {gpu.name}  (compute capability sm_{gpu.capability[0]}{gpu.capability[1]}, "
          f"{gpu.total_vram_gb:.1f} GB VRAM)")

    if gpu.capability >= BLACKWELL_CC and gpu.capability_supported_by_torch is False:
        msg = (
            f"\n{'=' * 72}\n"
            f"FATAL: GPU '{gpu.name}' reports compute capability "
            f"sm_{gpu.capability[0]}{gpu.capability[1]} (Blackwell or newer), but "
            f"the installed torch build only ships kernels for: "
            f"{', '.join(gpu.torch_arch_list) or '(none reported)'}.\n\n"
            "This build predates Blackwell support. If you proceed, matmuls will "
            "fail deep inside a kernel launch with an opaque CUDA error instead "
            "of this message. Fix:\n"
            "  pip uninstall -y torch torchvision torchaudio\n"
            "  pip install --pre torch --index-url "
            "https://download.pytorch.org/whl/nightly/cu124\n"
            "(check https://pytorch.org/get-started/locally/ for the current "
            "stable release with sm_120 kernels — by the time you read this a "
            "stable build likely has it; nightly is the fallback.)\n"
            "Also verify bitsandbytes separately: `python -c \"import bitsandbytes as bnb; "
            "print(bnb.__version__)\"` — versions built before Blackwell support "
            "was added will report NF4/INT8 as available but silently produce "
            "wrong numerical results or crash on load.\n"
            f"{'=' * 72}\n"
        )
        if strict:
            raise EnvironmentError_(msg)
        else:
            print(msg, file=sys.stderr)

    if gpu.total_vram_gb and gpu.total_vram_gb < 8.5:
        print(
            f"[env] VRAM budget is tight ({gpu.total_vram_gb:.1f} GB). This project "
            "defaults to 1-3B models and treats 8 GB as a hard ceiling — fp16 "
            "models above ~4B will be routed through CPU offload rather than "
            "loaded fully onto the GPU."
        )


def probe_npu() -> tuple[bool, str]:
    """Best-effort detection of an Intel NPU device node. On WSL2 this will
    almost always come back False because NPU drivers are Windows-side and are
    not passed through to the Linux kernel. We still check honestly instead of
    hardcoding False, in case a future WSL release adds passthrough.
    """
    if _is_wsl():
        # Known WSL2 device nodes for NPU passthrough do not exist as of
        # current WSL releases. Check anyway, then explain why it's expected
        # to fail.
        candidates = ["/dev/accel0", "/dev/intel_vpu"]
        for c in candidates:
            if os.path.exists(c):
                return True, f"found {c}"
        return False, (
            "no NPU device node found (expected on WSL2 — Intel NPU drivers are "
            "Windows-side and not passed through to the Linux kernel as of this "
            "WSL release). Run the OpenVINO/NPU target natively on Windows "
            "instead; see README 'Running OpenVINO/NPU natively on Windows'."
        )
    # Native Linux: check for the intel vpu driver / accel subsystem.
    if shutil.which("openvino_2024_stub") is not None:
        pass
    for c in ["/dev/accel/accel0", "/dev/intel_vpu"]:
        if os.path.exists(c):
            return True, f"found {c}"
    return False, "no NPU device node found"


def probe_igpu() -> tuple[bool, str]:
    """Check for an Intel iGPU render node usable by OpenVINO's GPU plugin."""
    render_nodes = []
    dri = "/dev/dri"
    if os.path.isdir(dri):
        render_nodes = [f for f in os.listdir(dri) if f.startswith("renderD")]
    if _is_wsl():
        if render_nodes:
            return True, f"found {render_nodes} (WSLg/DXCore passthrough)"
        return False, "no /dev/dri render node visible inside WSL2 for the iGPU"
    if render_nodes:
        return True, f"found {render_nodes}"
    return False, "no /dev/dri render node found"


def print_startup_banner(output_dir: str) -> GpuInfo:
    print("=" * 72)
    print("hf-quant-bench: environment check")
    print("=" * 72)
    print(f"[env] platform: {platform.platform()}")
    print(f"[env] WSL2 detected: {_is_wsl()}")

    warn = check_path_not_on_windows_mount(output_dir)
    if warn:
        print(warn)

    gpu = probe_gpu()
    verify_blackwell_or_die(gpu, strict=True)

    npu_ok, npu_reason = probe_npu()
    print(f"[env] NPU reachable: {npu_ok} ({npu_reason})")

    igpu_ok, igpu_reason = probe_igpu()
    print(f"[env] iGPU render node reachable: {igpu_ok} ({igpu_reason})")

    print("=" * 72)
    return gpu


if __name__ == "__main__":
    print_startup_banner(os.getcwd())
