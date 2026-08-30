"""Peak VRAM / RAM tracking around a format's whole benchmark run.

VRAM: device-wide usage sampled via `nvidia-smi` on a background thread,
NOT `torch.cuda.max_memory_allocated()`. That distinction matters a lot here:
`llama-cpp-python` (the GGUF backend) allocates GPU memory through its own
native CUDA calls entirely outside PyTorch's caching allocator, so
`torch.cuda.max_memory_allocated()` reports only incidental torch-side
allocations (a few MB) for every GGUF format regardless of the model's real
VRAM footprint -- a real, confirmed measurement bug, not a rounding
difference. Polling `nvidia-smi --query-gpu=memory.used` device-wide is the
one measurement that is actually comparable across the transformers backend
(fp16/bnb/gptq/awq) and the llama.cpp backend (gguf), since it counts memory
regardless of which allocator claimed it.

That said, the nvidia-smi-delta approach has its own real failure mode,
found and root-caused after a full sweep: it only works if the *previous*
format's VRAM was actually returned to the driver before the *next*
format's baseline is captured. `torch.cuda.empty_cache()` only releases
memory PyTorch's caching allocator has already marked free -- it does
nothing if the model object is still referenced anywhere (which it was:
`ModelHandle.close()` used to be a no-op, so `release_gpu_memory()` ran
while the model was still alive, found nothing to release, and the
now-unreachable-but-never-freed memory sat in PyTorch's private cache for
the rest of the process). The next format's allocator then silently reused
that stale cached memory instead of asking the driver for more, so
nvidia-smi showed no growth and the delta collapsed toward zero. Fixed on
two sides: `ModelHandle.close()` now actually drops the model reference
(see loader.py) before `release_gpu_memory()` runs, AND this module waits
for nvidia-smi to actually reflect that release (`_wait_for_gpu_memory_to_settle`)
before the next format's baseline is captured, rather than trusting a single
`empty_cache()` call to have taken effect synchronously -- WSL2's GPU
paravirtualization layer has shown other reporting-latency quirks in this
project, so a single read immediately after cleanup is not trustworthy.

RAM: peak RSS via a background `psutil` poller, NOT
`resource.getrusage().ru_maxrss`. `ru_maxrss` is a process-lifetime
high-water mark on Linux with no reset syscall -- once any format pushes it
to some value, every subsequent format in the same long-lived process
reports that exact same historical peak forever, which is why six
consecutive rows showed the identical figure to 7 significant figures. A
background poller reading `psutil.Process().memory_info().rss` on the same
per-format start/stop lifecycle as the VRAM poller gives a real, resettable
per-format measurement instead.
"""

from __future__ import annotations

import dataclasses
import subprocess
import threading
import time


@dataclasses.dataclass
class ResourceSnapshot:
    peak_vram_mb: float | None
    peak_ram_mb: float | None
    vram_method: str  # "nvidia-smi-poll" | "torch-allocator-fallback" | "unavailable"
    ram_method: str  # "psutil-poll" | "unavailable"


def _read_gpu_memory_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        # One line per GPU; this project only targets a single-GPU box, so
        # take the first line.
        first_line = out.stdout.strip().splitlines()[0]
        return float(first_line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _wait_for_gpu_memory_to_settle(
    below_mb: float, timeout_s: float = 10.0, poll_interval_s: float = 0.25
) -> bool:
    """Block until nvidia-smi reports usage at or below `below_mb`, or give up.

    Called right after `release_gpu_memory()` in the previous format's
    cleanup, before the next format's baseline is captured. Returns True if
    it actually settled, False if it timed out (in which case the caller
    should treat the upcoming baseline as suspect, not silently trust it --
    see the loud warning in reset_peak_tracking()).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        v = _read_gpu_memory_used_mb()
        if v is None:
            return False  # nvidia-smi unavailable; nothing to wait for
        if v <= below_mb:
            return True
        time.sleep(poll_interval_s)
    return False


class _Poller:
    """Samples a value-producing function on a background thread and tracks
    the max seen since `start()`, reset on every `start()` call.

    A single point-in-time read after generation finishes would miss the
    actual peak (e.g. KV cache allocation spikes during prefill) -- sampling
    every 200ms while cases run is a cheap way to catch that without adding
    a heavier dependency for either GPU or CPU memory.
    """

    def __init__(self, read_fn, interval_s: float = 0.2):
        self._read_fn = read_fn
        self.interval_s = interval_s
        self._peak: float | None = None
        self._baseline: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self._read_fn()
            if v is not None and (self._peak is None or v > self._peak):
                self._peak = v
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._baseline = self._read_fn()
        self._peak = self._baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_and_get_peak_above_baseline(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._peak is None:
            return None
        if self._baseline is None:
            return self._peak
        # Report the incremental usage this format's run actually consumed,
        # not the whole-process/whole-device total (other processes, and
        # this Python process's own fixed import/interpreter overhead, have
        # a nonzero baseline that would otherwise dilute cross-format
        # comparability).
        return max(self._peak - self._baseline, 0.0)


def _read_process_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except ImportError:
        return None


_vram_poller: _Poller | None = None
_ram_poller: _Poller | None = None
_vram_method: str = "unavailable"


def reset_peak_tracking() -> None:
    global _vram_poller, _ram_poller, _vram_method

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    baseline = _read_gpu_memory_used_mb()
    if baseline is not None:
        _vram_method = "nvidia-smi-poll"
    else:
        _vram_method = "torch-allocator-fallback"
        print(
            "[resource_monitor] WARNING: nvidia-smi unreachable; falling back to "
            "torch.cuda.max_memory_allocated(), which reads ~0 for any non-torch "
            "backend (e.g. GGUF/llama.cpp) regardless of real usage. This will be "
            "recorded in backend_note."
        )

    _vram_poller = _Poller(_read_gpu_memory_used_mb)
    _vram_poller.start()

    _ram_poller = _Poller(_read_process_rss_mb)
    _ram_poller.start()
    if _read_process_rss_mb() is None:
        print("[resource_monitor] WARNING: psutil unavailable; peak_ram_mb will be null for this format.")


def snapshot() -> ResourceSnapshot:
    global _vram_poller, _ram_poller, _vram_method

    peak_vram_mb = None
    if _vram_poller is not None:
        peak_vram_mb = _vram_poller.stop_and_get_peak_above_baseline()
        _vram_poller = None

    vram_method = _vram_method
    if peak_vram_mb is None:
        # nvidia-smi unavailable for some reason -- fall back to torch's
        # allocator stat, which at least gives a number for the
        # transformers-backed formats even though it will read as ~0 for
        # llama.cpp-backed ones (see module docstring).
        vram_method = "torch-allocator-fallback"
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        except ImportError:
            vram_method = "unavailable"

    peak_ram_mb = None
    ram_method = "unavailable"
    if _ram_poller is not None:
        peak_ram_mb = _ram_poller.stop_and_get_peak_above_baseline()
        _ram_poller = None
        if peak_ram_mb is not None:
            ram_method = "psutil-poll"

    return ResourceSnapshot(peak_vram_mb=peak_vram_mb, peak_ram_mb=peak_ram_mb, vram_method=vram_method, ram_method=ram_method)


def release_gpu_memory() -> None:
    """Called between formats — 8GB VRAM will not hold two loaded models.

    Only effective now that `ModelHandle.close()` (see loader.py) actually
    drops the model reference before this runs -- previously this ran while
    the model was still referenced, found nothing eligible to free, and was
    a complete no-op that silently corrupted every subsequent format's VRAM
    baseline (Bug 3). Also waits for nvidia-smi to actually reflect the
    release rather than trusting empty_cache() to have taken effect
    synchronously, since WSL2's GPU paravirtualization has shown reporting
    lag elsewhere in this project.
    """
    pre_release_mb = _read_gpu_memory_used_mb()

    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    if pre_release_mb is not None:
        # Expect memory to drop meaningfully; settling to within 5% of the
        # pre-release level (not necessarily all the way to zero -- a
        # driver/runtime baseline persists even fully idle) is treated as
        # "released." If it never drops, warn loudly rather than silently
        # let the next format's baseline capture a contaminated reading.
        target = pre_release_mb * 0.5  # expect a real model unload to roughly halve usage at minimum
        settled = _wait_for_gpu_memory_to_settle(below_mb=max(target, 200.0))
        if not settled:
            current = _read_gpu_memory_used_mb()
            print(
                f"[resource_monitor] WARNING: GPU memory did not settle after cleanup "
                f"(was {pre_release_mb:.0f} MB, still {current} MB after waiting). "
                "The next format's peak-VRAM baseline may be contaminated by this "
                "format's leftover allocation -- treat its peak_vram_mb with suspicion "
                "if it looks implausibly low."
            )
