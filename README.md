# hf-quant-bench

Convert any Hugging Face causal LM into every quantized format your hardware
can target, then benchmark agentic tool-calling behavior across formats so
you can see what quantization actually costs — not just perplexity, but
whether the model still picks the right tool, fills in correct arguments,
and knows when to shut up and just answer.

Designed around one binding constraint: **RTX 5070, 8 GB VRAM, Blackwell
(sm_120), WSL2 on Windows, 32 GB system RAM.** Everything here defaults to
fitting inside that box; see "Hardware assumptions" below.

---

## 1. First-run setup (paste these in order, from `~/projects`)

```bash
cd ~/projects
mkdir -p hf-quant-bench
cd hf-quant-bench
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install torch **first, separately**, with the CUDA index — this is the one
package where getting the wrong build silently breaks everything on a
Blackwell GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verified in this session against an actual RTX 5070 (sm_120): the stable
**cu124** wheel does NOT ship sm_120 kernels and gets flagged immediately by
`env_check.py` ("torch build only ships kernels for: sm_50, sm_60, sm_70,
sm_75, sm_80, sm_86, sm_90"). The stable **cu128** wheel does have sm_120
kernels and passes cleanly. Use cu128, not cu124. If a future torch release
changes this again, check pytorch.org/get-started/locally for the current
recommendation, or fall back to nightly:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Then install everything else:

```bash
pip install -r requirements.txt
pip install -e .
```

Verify the GPU is actually visible and Blackwell-capable before doing anything else:

```bash
python -m hf_quant_bench.env_check
```

This prints `torch.cuda.is_available()`, the device name, and — critically —
whether your installed torch build has kernels for your GPU's compute
capability. If it doesn't, this command **fails loudly with a readable
message and a fix**, instead of you finding out three minutes into a
conversion run via an opaque CUDA kernel-launch error.

**Do not install an NVIDIA driver inside WSL.** CUDA reaches WSL2 through the
Windows host driver automatically. If `nvidia-smi` doesn't work in your WSL
shell, that's a Windows-side driver problem, not something this project (or a
`pip install` inside WSL) can fix.

**Do not write outputs under `/mnt/c/...`.** The venv above and everything
you convert should live on the native WSL filesystem (as it does by default
at `~/projects/hf-quant-bench`) — `/mnt/c` is a 9p network mount and is
meaningfully slower for the kind of large sequential reads/writes model
weights involve. `env_check.py` will warn you if `--out` resolves under
`/mnt/`.

### Optional: llama.cpp, for the GGUF target

```bash
bash scripts/build_llama_cpp.sh
export LLAMA_CPP_DIR=$HOME/llama.cpp
```

Add the `export LLAMA_CPP_DIR=...` line to `~/.bashrc` so it persists across
shells. Without it, the `gguf` target is skipped with a clear reason (not a
crash) — it's independent of every other target.

### Optional: OpenVINO, for the iGPU/NPU-adjacent target

```bash
pip install openvino "optimum[openvino]"
```

See the dedicated **"Running OpenVINO/NPU natively on Windows"** section
below — the NPU is almost certainly not reachable from this WSL2 shell, and
that's expected, not a bug.

### If a library tries to JIT-compile a CUDA kernel and fails

Several targets do this: llama.cpp's build, gptqmodel's Marlin inference
kernel, optimum-quanto's KV-cache-quantization kernel. On this WSL2 box, all
three hit the identical two-part failure the first time:

1. System `gcc`/`g++` is newer than nvcc (CUDA 12.8) supports ("unsupported
   GNU version! gcc versions later than 13 are not supported"). Fix: install
   `gcc-13`/`g++-13` alongside the system compiler (`apt install gcc-13
   g++-13`) and point nvcc at it.
2. Even with the right host compiler, `cicc` (nvcc's internal codegen
   backend) isn't found unless `/usr/local/cuda-12.8/nvvm/bin` is also on
   `PATH` — the compiler-ID check alone doesn't need it, but real kernel
   compilation does.

The combination that has worked every time so far:

```bash
export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/nvvm/bin:$PATH
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-13'
export TMPDIR=$HOME/tmp   # see the /tmp note below — nvcc writes scratch files here
mkdir -p $HOME/tmp
```

If you hit this on a target not listed above, the same three lines are the
first thing to try before assuming something is fundamentally broken.

### If something fails with "No space left on device" from `/tmp`

`/tmp` in WSL2 is tmpfs (RAM-backed), sized to half the memory in
`.wslconfig`. ONNX export and any nvcc JIT compile can write large
intermediate files there. Set `TMPDIR` to a directory on the real
filesystem (see above) rather than growing `/tmp`'s tmpfs size — the WSL
filesystem has far more headroom than RAM does.

---

## 2. Sanity check: what can this environment actually run?

```bash
python -m hf_quant_bench.convert --list
```

This prints every registered conversion target and whether your current
environment can run it right now, with a reason for every "no". Nothing here
touches the network or downloads a model — it's a fast, pure environment
check.

---

## 3. Convert a model

```bash
python -m hf_quant_bench.convert \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --out outputs/qwen2.5-1.5b
```

- `--model` accepts any HF repo id or local directory — nothing here is
  hardcoded to one architecture.
- Omit `--only` to attempt every registered target; a target whose
  dependency isn't installed is skipped (recorded in `manifest.json`), not
  fatal to the run.
- `--only fp16 bnb-nf4 gguf` restricts to specific targets.
- `--gguf-quant-types Q4_K_M Q8_0` overrides the default GGUF quant list
  (`Q4_0 Q4_K_M Q5_K_M Q8_0`).
- `outputs/qwen2.5-1.5b/manifest.json` records, per target: status
  (ok/skipped/failed), output path, on-disk size, wall-clock conversion
  time, and — for skipped/failed targets — exactly why.

For the 4B fp16-via-CPU-offload case mentioned in the brief: the `fp16`
target always loads on CPU for the re-save regardless of model size, so it
never OOMs your VRAM; loading a 4B model fully onto an 8GB GPU for anything
else (bnb/GPTQ/AWQ/torchao) is where you'd hit the ceiling, which is exactly
why those targets quantize into 4-8 bit rather than holding fp16 on-GPU.

---

## 4. Validate the benchmark loop before committing to a full sweep

```bash
python -m hf_quant_bench.bench.run \
  --manifest outputs/qwen2.5-1.5b/manifest.json \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --out bench_results/smoke \
  --smoke
```

Runs 3 test cases against one format end-to-end (load → agentic loop → rule
metrics → report) in under two minutes. Confirm this works before running
the full 100-case suite across every converted format.

---

## 5. Full benchmark sweep

```bash
python -m hf_quant_bench.bench.run \
  --manifest outputs/qwen2.5-1.5b/manifest.json \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --out bench_results/full
```

No API key, no environment variable to export, no network call required for
any number in the output -- see "Metrics" below.

Formats are loaded and evaluated **one at a time**, with GPU memory released
between them — 8GB will not hold two loaded models simultaneously, so this
is enforced in code, not left to you to remember. Before each format's full
sweep, a 3-case preflight gate checks it against an fp16 reference and
aborts that format (skipping the full sweep, not the whole run) if it looks
broken -- see "Preflight validation gate" below.

Output:
- `bench_results/full/results.csv` — one row per format, one column per metric.
- `bench_results/full/results.md` — comparison table, fp16-baseline deltas, calibrated-vs-uncalibrated comparison, and a Pareto-optimal-formats section.
- `bench_results/full/trajectories_<format>.jsonl` — full per-case trajectory logs for manual inspection of individual failures.

### Metrics — all five are deterministic, all reproducible without a network call

No cloud LLM API, no LLM-as-a-judge, nothing non-reproducible, in the
primary metrics:

1. **tool_correctness** — does the sequence of tools actually called match the test case's known-correct sequence.
2. **argument_correctness** — are the called tools' arguments correct (case/type-tolerant, arithmetic-expression-equivalent for calculator args).
3. **hallucination_rate** — fraction of steps that named a tool or parameter not in the schema.
4. **path_optimization** — `clamp(optimal_steps / actual_steps, 0, 1)`, where `optimal_steps` is a known-minimal step count baked into each test case (see below). A trajectory that hits the step cap without terminating scores 0 here and is *also* counted in the separate `timeout_rate` column, so a hung/looping trajectory can't be mistaken for a merely-inefficient one.
5. **answer_completeness** — `matched_required_facts / total_required_facts`, where `required_facts` is a list of concrete values baked into each test case, computed directly from the deterministic mock tools' real output (not guessed). Matching is substring-based for text and tolerance-based numeric comparison for numbers (`23`, `23.0`, `23°C` all match a required fact of `23`) — no embeddings, no semantic similarity, no model of any kind. `no_tool`/`clarify` cases score 1.0 here if the model correctly declined/asked, but are excluded from the format-level average so a pile of trivial "nothing to check" scores can't dilute it.

**Optional supplementary local judge** (off by default, no cloud API):
`--local-judge /path/to/judge.gguf` runs a local GGUF model through
llama.cpp, strictly *after* the full sweep completes and every subject
model has been released (never concurrently — 8GB VRAM can't hold a judge
and a subject model at once), and adds two separate, clearly-labelled
columns: `path_optimization_judged` / `answer_completeness_judged`. These
are supplementary opinions, never a replacement for the five metrics above,
and this project runs completely without them.

### Preflight validation gate

Before any format's full sweep, `fp16` is loaded once and run through 3
`tool_call`-expected cases (deliberately never `no_tool`/`clarify` -- a
model that emits zero tool calls at all would trivially "pass" a gate drawn
from those) to establish a reference: chat-template hash, mean
steps-per-trajectory, whether any tool call was parsed at all. Printed at
the start of every run. Each subsequent format runs the same 3 cases before
its own full sweep and aborts (skipping the 100-case sweep for that format
only, with the reason recorded directly in `results.csv`'s `backend_note`)
if any of: zero valid tool calls parsed, mean steps-per-trajectory below
1.2, empty/unparseable raw output, or a chat-template hash mismatch against
the fp16 reference. This exists because it caught a real incident during
development — see the "Known weak spots" section.

---

## Test suite

`src/hf_quant_bench/bench/test_cases.yaml` — 100 cases, plain YAML, easy to
extend: add an entry with `id`, `category`, `prompt`, `available_tools`,
`expected_behavior`, `expected_trajectory`, `optimal_steps`,
`required_facts`, `judge_notes`. Categories covered: single-tool-simple,
single-tool-nested/typed-args, multi-step (tool A's output feeds tool B),
no-tool-needed (14 cases — deliberately not skipped, since this is where
quantized models degrade first), irrelevant distractor tool present in
schema, ambiguous-should-clarify (11 cases), and three long-context
needle-in-haystack cases (needle placed early/middle/late in ~2000 words of
filler). See the file's own header comment for exactly how `optimal_steps`
and `required_facts` are derived and scored.

Mock tools live in `src/hf_quant_bench/bench/tools.py`: `get_weather`,
`get_calendar_events`, `web_search`, `calculator`, `read_file`, plus
`send_email` as the deliberate distractor. All are deterministic (seeded by
hashing the input, or a small canned-answer table for web_search), so
re-running produces identical tool results -- which is also what makes
`required_facts` computable as exact ground truth rather than guessed.

---

## Running the OpenVINO/NPU target natively on Windows

The Intel NPU's drivers are Windows-side and are not passed through to
WSL2 as of current WSL releases — `env_check.py` detects this honestly at
runtime (checks for `/dev/accel0` / `/dev/intel_vpu`, finds nothing, and
tells you so) rather than pretending the NPU is available. To actually
exercise the NPU:

1. Install Python natively on Windows (not WSL) — Python 3.11+, from
   python.org or the Microsoft Store.
2. Open **PowerShell** (not WSL) and create a second venv:
   ```powershell
   cd C:\Users\<you>\projects
   git clone <this repo, or copy the folder>  hf-quant-bench-win
   cd hf-quant-bench-win
   py -3.11 -m venv .venv-win
   .venv-win\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install openvino optimum[openvino] transformers accelerate huggingface_hub pyyaml
   ```
3. Reuse the model files you already converted in WSL — the OpenVINO IR
   directories (`outputs/<model>/openvino-fp16`, `openvino-int8`) are plain
   files; copy or symlink them across, e.g. from PowerShell:
   ```powershell
   Copy-Item -Recurse \\wsl.localhost\Ubuntu\home\<you>\projects\hf-quant-bench\outputs .\outputs
   ```
4. Run inference against the NPU device string directly (this project's
   `bench/loader.py` OpenVINO path accepts a device string — pass `"NPU"`
   instead of `"CPU"`/`"GPU"` when calling `OVModelForCausalLM.from_pretrained`
   natively; you'll likely invoke this as a small standalone script on the
   Windows side rather than the full WSL-oriented `bench/run.py`, since the
   rest of the benchmark harness assumes the WSL layout).

This is intentionally kept separate from the main flow: the WSL benchmark
never blocks on NPU availability, and OpenVINO/NPU support degrades to "CPU
only, on WSL" gracefully rather than failing the run.

---

## Hardware assumptions baked into this project

- 8 GB VRAM is a hard ceiling. Default model recommendation: 1-3B causal LMs
  (e.g. Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct). A 4B model is kept
  loadable via the fp16-on-CPU path for the `fp16` baseline target
  specifically — nothing in this project assumes a full-precision model
  fits in VRAM, and every GPU-quantization target (bnb/GPTQ/AWQ/torchao)
  quantizes precisely so the working set fits.
- No Apple silicon anywhere in this codebase — MLX is not a target, on purpose.
- No local Qualcomm device — the `qai-hub` target is cloud-only, off by
  default, and requires `--enable-qai-hub` plus a configured API token.
  It is **verified working** against Qwen2.5-1.5B-Instruct: it produces a
  3.6 GB QNN context binary for a Snapdragon 8 Elite QRD profile. See
  "What the `qai-hub` target needed" below for the five things that had to
  be right — the defaults in the original implementation were wrong on all
  five.
- CPU-only conversions (fp16 re-save, GGUF, ONNX, OpenVINO IR export) run on
  CPU and print that they're doing so — they never touch the GPU
  unnecessarily.

---

## Project layout

```
src/hf_quant_bench/
  env_check.py          # GPU/Blackwell verification, NPU/iGPU runtime detection
  registry.py            # pluggable target registry (check + convert per target)
  targets/                # one module per format, self-registering
  convert.py              # Part 1 CLI
  manifest.py
  bench/
    tools.py              # mock tool registry + deterministic fake impls
    test_cases.yaml        # 100 test cases
    cases.py               # YAML loader
    loader.py               # per-format model loading -> uniform generate() interface
    preflight.py             # per-format 3-case validation gate (see "Preflight validation gate")
    runner.py                 # the actual multi-turn agentic loop
    metrics_rule.py            # all 5 primary metrics -- deterministic, no cloud API
    local_judge.py              # optional, off by default: offline local-GGUF supplementary scoring
    resource_monitor.py          # VRAM/RAM peak tracking
    report.py                     # results.csv / results.md
    run.py                         # Part 2 CLI
```

---

## Known weak spots — read this before trusting a number

**Where `required_facts` matching is most likely to produce a false
negative** (silently understating answer_completeness for an otherwise-good
answer):
- **Zero-event calendar cases** deliberately have `required_facts: []` and
  score vacuously 1.0. There is no safe literal substring for "you have
  nothing scheduled" -- correct paraphrases ("your calendar is free",
  "nothing on that day", "no events found") share no common substring, so
  any literal I picked would false-negative on the majority of equally
  correct phrasings. Documented in test_cases.yaml's header, not silently
  absorbed into a fragile check.
- **Boolean facts** (`nested_readfile_json`'s required fact is the literal
  string `"true"`): a model saying "autosave is enabled" instead of
  "autosave is true" is arguably just as complete but won't match. Chose
  the literal JSON value since it's exact and deterministic, but this is a
  real, live false-negative risk for any boolean-valued fact.
- **Non-terminating decimals** (`multistep_readfile_then_calc_3`'s required
  fact is `"160666.67"`, from 482000/3): the numeric matcher's tolerance
  (`max(1.0, |fact| * 2%)`, so ±3213 here) should absorb most reasonable
  rounding, but a model that states a very differently-rounded or
  truncated figure could still miss.
- **Search facts for non-canned queries**: every `web_search` case in the
  suite uses a query with a real, human-meaningful canned answer (see
  `tools.py`'s `_CANNED_SEARCH`) specifically to avoid this, but if you add
  a new search case with an *uncanned* query, the mock tool falls back to a
  random hash-fragment snippet -- do not derive a `required_facts` entry
  from that fragment; a well-behaved model has no reason to quote random
  hex noise in a natural-language answer, and doing so would false-negative
  every correctly-functioning format on that case. Add a canned entry
  first.
- **Multi-step chains score `required_facts` from the LAST tool call
  only** (e.g. a search-then-weather chain only checks the weather
  values, not the intermediate city name) -- a deliberate simplification,
  not a limitation I ran out of time to fix; the last tool's result is what
  the final answer is actually built from, and requiring every intermediate
  value too would make cases fail for a merely-terse-but-correct final
  answer.
- **`send_email`-terminated chains** require only the recipient address as
  the completeness check (a confirmation like "I've sent the email to
  alex@example.com" is treated as complete) -- deliberately shallow, since
  the email body is free-form model-authored text with nothing deterministic
  to check inside it.

**Where numeric tolerance is most likely to produce a false positive**
(silently overstating answer_completeness by matching the wrong number):
`_fact_matches`'s numeric tolerance is `max(1.0, |fact| * 2%)` absolute, so
a small required fact can be satisfied by an unrelated number that happens
to land in range rather than the actual value the model reported. This was
caught during a 2026-08 ground-truth audit of `test_cases.yaml`:
`simple_readfile_3` originally required the literal db_timeout retry count
(`"2"`), and at least one format's answer contained an unrelated small
number (a different retry count, off by ≤1) within that ±1.0 tolerance
window, passing the case without actually stating the required fact. The
case's `required_facts` was corrected (to `"db_timeout"`, since the prompt
only asks for a summary and doesn't require the exact count), which
sidesteps this instance, but the underlying tolerance mechanism is
unchanged and any other case with a small (~single-digit) numeric
`required_facts` entry sitting near other plausible numbers in the model's
answer carries the same risk. Not fixed here, deliberately: tightening the
tolerance risks reintroducing the false-negative side (missing a
correctly-rounded answer), and this project already rejected fuzzier
matching for the reason given below. Flagged as a live limitation, not
silently absorbed.

None of the above were fixed by relaxing the matcher further, because every
looser alternative I considered (fuzzy string matching, keyword sets)
reintroduces exactly the non-determinism/"model of some kind" the spec
explicitly rules out. They are documented, not hidden.

---

This section reflects an actual completed run against Qwen2.5-1.5B-Instruct
on an RTX 5070 (Blackwell, sm_120) under WSL2 — 9 formats, 100 test cases
each, all with real weights and real generation. Not a dry-run projection.

**Formats that work and were fully benchmarked**: fp16, bnb-nf4, bnb-int8,
gptq-4bit (calibrated on real wikitext2), awq-4bit (calibrated on real
wikitext2, via the llm-compressor fallback), gguf-Q4_0/Q4_K_M/Q5_K_M/Q8_0.

### What the `qai-hub` target needed

The `qai-hub` cloud-compile target now works end-to-end — job
[`jp39koxlp`](https://workbench.aihub.qualcomm.com/jobs/jp39koxlp/),
producing a 3.56 GB QNN context binary for Snapdragon 8 Elite QRD in ~25
minutes wall-clock (the bulk of which is a 5.76 GB upload and a 3.31 GB
download, not compute). Getting there took five sequential fixes, each of
which surfaced only after the previous one was corrected — and each costing
a full ~8-minute upload to discover:

1. **`torch.jit.trace` can't type HF's `ModelOutput` dataclasses.** Tracing
   `AutoModelForCausalLM` directly dies on `CausalLMOutputWithPast`. Setting
   `config.return_dict=False` is *not* a fix on transformers 5.14.1 — inner
   submodules still hand back `ModelOutput` objects while outer code indexes
   them as tuples, which just swaps one untraceable error for another. A thin
   wrapper module returning `.logits` as a plain tensor is what actually works.
2. **int64 I/O is unsupported on Hexagon.** Needs
   `--truncate_64bit_io`, or the job fails at compile with exactly that
   instruction in the message.
3. **The kwarg is `options=`, not `compile_options=`.** The latter is what
   the CLI calls it; the Python `submit_compile_job()` signature uses
   `options`.
4. **TorchScript uploads are deprecated and fail here.** AI Hub rejects the
   `.pt` with "Failed to upgrade the exported ONNX model to opset 21" and its
   own error text points at `torch.export`. Exporting an ExportedProgram
   (`.pt2`) via `torch.export.export(..., strict=False)` and uploading that
   file is the working path.
5. **The default target runtime (LiteRT/tflite) has a hard size ceiling this
   model exceeds** — "Model is too large for the LiteRT model format". AI
   Hub's FAQ documents a >2 GB compile failure mode and recommends
   quantization, but that ceiling is **specific to the LiteRT path**:
   `--target_runtime qnn_context_binary` compiled the same unquantized fp32
   graph without complaint. Worth knowing, because reading the 2 GB limit as
   universal would lead you to conclude (as I initially did) that a 1.5B
   model can't be compiled without quantizing first. It can.

Two robustness fixes went in alongside those. The job id and dashboard URL
are now written to `qai_hub_job.json` *immediately* after submission rather
than on success, because a failure after the slow upload otherwise leaves no
way to inspect the job; and `download()` is retried with backoff, because the
target artifact can briefly lag the job being reported successful ("Model
file has not yet been uploaded"). The ~6 GB local `.pt2` is deleted straight
after upload.

#### On-device profile (real Snapdragon hardware)

`--qai-hub-profile` additionally runs the compiled artifact on a real device
in Qualcomm's device farm — the only way to get hardware numbers for this
target, since there is no local Qualcomm device here. Measured on Snapdragon
8 Elite QRD, 100 inference runs (raw output in
`outputs/<model>/qai-hub/qai_hub_profile.json`):

| Metric | Value |
| --- | --- |
| Inference latency (median of 100) | **45.4 ms** (min 43.0, max 48.1) |
| Peak inference memory | **119.6 MiB** |
| Cold load | 597 ms |
| Warm load | 541 ms |
| Compute-unit split | **1659 / 1659 ops on NPU** (zero CPU/GPU fallback) |

Two things worth pulling out. **Full NPU residency** — every one of the 1659
ops was placed on the Hexagon NPU with nothing falling back to CPU or GPU,
which is the outcome you want and not a given for a graph exported this
generically. And **peak inference memory is ~120 MiB against a 3.56 GB
artifact**, because the context binary memory-maps its weights rather than
loading them all resident.

**Do not compare that 45.4 ms against the `prefill_latency_s_mean` column in
`results.csv`.** They are not the same measurement: the CSV's prefill runs
over a full benchmark prompt, whereas this traced graph has a fixed
`seq_len` of the dummy prompt's token count. The on-device number is a
single forward pass over a much shorter input, so it would flatter the NPU
badly if read as a like-for-like speed comparison.

**Cost**: none. AI Hub is "currently completely free to use" per their FAQ,
covering compile, profile and inference jobs alike. The `--enable-qai-hub`
opt-in exists because this target sends your model off the machine, not
because it bills you.

**Caveat on what this artifact actually is**: the trace fixes `seq_len` to
the dummy prompt's token count and carries no KV cache, so this is a proof
that the conversion path works, not a deployable LLM. Real on-device LLM
deployment via AI Hub uses the `qai-hub-models` recipes, which split the
model into prompt-processing and token-generation graphs with static KV-cache
shapes and AIMET quantization. That is a substantially different pipeline and
is not what this target does.

**Formats that do NOT work in this environment, with real reasons**:
- `torchao-int4` — torchao 0.18.0's default int4 packing format hard-requires
  an internal package (`mslk`) that is a non-functional placeholder stub on
  PyPI (version `0.0.0`). Not fixable by installing more things; needs either
  an older torchao release or upstream to finish shipping that dependency.
- `onnx-fp32` / `onnx-int8-dynamic` — genuine structural incompatibility
  between `optimum` (2.3.0, latest at time of writing) and `transformers`
  5.14.1 (pinned elsewhere in this project because gptqmodel/AWQ require
  it). `optimum`'s ONNX exporter imports at least two private transformers
  APIs that no longer exist in 5.x (`get_parameter_dtype`, shimmed here as a
  one-line compatibility patch in `onnx_export.py`; `_CAN_RECORD_REGISTRY`,
  a private internal tracing registry, NOT shimmed — guessing at its new
  shape risks a silently-wrong export). `optimum-onnx` itself pins
  `transformers<4.58.0`, confirming this is upstream not caught up, not a
  bug here. **Practical fix**: run ONNX export in a separate venv pinned to
  an older transformers, decoupled from the gptqmodel/AWQ dependency chain.
- KV-cache quantization (`fp16-kv4`/`fp16-kv2` in `bench/loader.py`) —
  **implemented but confirmed broken**, not just unverified. With the
  `optimum-quanto` backend actually loading and running (after the same
  nvcc/gcc-13/PATH fix documented above), generation produces corrupted
  output (`"<tool_call> I call get called, \" class {\""`) and
  catastrophically slow decode (~59s for 10 tokens on one case, ~0.17 tok/s).
  This surfaced after three separate build/compat issues in a row for this
  one feature; I stopped debugging it rather than keep patching around an
  apparent deeper numerical/compatibility problem between `optimum-quanto`'s
  KV-cache quantization and this model/hardware/transformers-version
  combination. Do not trust `fp16-kv4`/`fp16-kv2` results without further
  investigation — the code path exists but is not validated as correct.

**The headline finding, and why I trust it**: GPTQ-4bit and AWQ-4bit both
substantially *underperform* the uncalibrated round-to-nearest formats
(bnb-nf4, GGUF) on tool-call correctness — GPTQ dropped to ~0.29-0.31,
AWQ to ~0.67-0.69, against ~0.72-0.80 for every uncalibrated format
including fp16 itself. I initially suspected this was an artifact of a
calibration-data bug (both `gptq.py` and `awq.py` were silently falling back
to a degenerate repeated-sentence synthetic calibration set because the
`"wikitext"` dataset id 404s on current `datasets` releases — fixed to
`"Salesforce/wikitext"`). After fixing that and recalibrating both on real
wikitext2 text, the numbers barely moved (GPTQ 0.31→0.29, AWQ 0.685→0.67).
So this is not a calibration-data-quality artifact: on this model, at this
size, for this specific task, GPTQ/AWQ's calibration process appears to
measurably damage this model's ability to comply with a prompted structured
tool-call format, even though the model's underlying reasoning stays
coherent (GPTQ's failure mode specifically was narrating "I will call
get_weather..." in prose instead of emitting `<tool_call>{...}</tool_call>`).
`results.md`'s "Calibrated vs. uncalibrated quantization" section states this
comparison for any model you run this against — the specific numbers above
are for this one model and won't generalize to every architecture.

**Live GPU-driver instability found and fixed**: GPTQ calibration and
llmcompressor's AWQ path both crashed the entire WSL VM reproducibly during
real calibration, with no Python traceback and no OOM-killer record. I ruled
out generic instability (sustained matmul and cuSOLVER loads both ran 60-90s
clean) and a missing-kernel-for-sm_120 issue (Triton JIT compiles and runs
fine repeatedly) before finding that forcing synchronous CUDA execution
(`CUDA_LAUNCH_BLOCKING=1`, applied in `convert.py` only when converting
gptq-4bit/awq-4bit — see the comment there) fixed it completely. This
strongly suggests an async-kernel-launch race in WSL2's GPU
paravirtualization for these specific workloads' launch patterns, not a
fundamental hardware incompatibility — but I did not root-cause it at the
driver level, so treat that explanation as the best available, not certain.

**Two silent-data-loss bugs found and fixed during this run**, worth
knowing about if you extend this project: `manifest.json` and `results.csv`/
`results.md` were each originally written only once, at the end of a whole
`--only`/`--formats` run. A crash or an external interruption (this
happened live, from a session boundary unrelated to the code, partway
through a 9-format sweep) would silently discard every already-completed
format's results along with the failed one. Both are now written
incrementally after every target/format completes.

**A real benchmark-measurement bug found and fixed**: GGUF's peak-VRAM and
decode-speed numbers were badly wrong in an earlier draft of this report —
`peak_vram_mb` used `torch.cuda.max_memory_allocated()`, which only tracks
PyTorch's own allocator and reads ~9MB for `llama-cpp-python` regardless of
the model's real footprint, since llama.cpp allocates GPU memory through its
own native CUDA calls entirely outside PyTorch. Fixed by polling
device-wide `nvidia-smi` memory usage on a background thread instead
(`resource_monitor.py`), which is also a fairer comparison across engines
than a PyTorch-specific stat. Decode speed had a similar issue: the original
code approximated the prefill/decode split by dividing total wall-clock time
proportionally by token-count share, which is not a measurement of either
quantity — fixed to read llama.cpp's own internal `llama_perf_context`
timings instead. The fix changed GGUF's reported decode speed from an
implausible ~1900 tok/s to ~110-160 tok/s.

**A benchmark-logic bug found and fixed**: the tool-call parser required a
closing `</tool_call>` tag. Real models, especially under greedy decoding,
frequently emit `<tool_call>{...}` and then hit EOS without ever emitting
the closing tag — the original regex-based parser silently misclassified
every one of these as "no tool call," which would have made every format's
`tool_correctness` read as an undercount. Fixed with a brace-matching JSON
extractor in `loader.py` that doesn't depend on the closing tag.

**Library APIs that churned during this session, so expect them to churn
again**: `llmcompressor`'s recipe API moved `oneshot` from
`llmcompressor.transformers` to the top-level package and restructured
`AWQModifier` from direct `bits`/`group_size` kwargs to a `scheme` string
(`"W4A16"`) between the version this was first written against and the
version actually installed. `gptqmodel`'s default inference kernel
auto-selects "Marlin," which needs its own nvcc JIT compile (same fix as
above) and is set via a plain `"backend"` string field in the saved
`config.json` (`"gptq_triton"` used here to avoid it) rather than any
documented public API. `transformers.cache_utils.QuantizedCache`'s
constructor and the `generate(cache_implementation=..., cache_config=...)`
wiring for it are internal/undocumented enough that they're worth
double-checking against whatever transformers version you actually have.

**What I assumed about your setup**:
- `python3.11` specifically (confirmed present at `/usr/bin/python3.11`),
  not the newest available Python, since torch/bnb/etc. wheels are most
  reliably available for 3.11/3.12.
- That "any HF causal LM" in practice means an instruct-tuned model for the
  benchmark half to be meaningful — the tool-calling protocol here is a
  prompted, model-agnostic `<tool_call>{json}</tool_call>` convention (see
  the big comment at the top of `bench/loader.py`) rather than any one
  model's native chat-template tool format, precisely because the brief
  requires genuine architecture-agnosticism. A base (non-instruct) model
  will likely score poorly across the board regardless of quantization —
  that's a real signal, not a harness bug.
- That `ru_maxrss` from `resource.getrusage` is reported in **kilobytes**,
  true on Linux (this project's only target platform via WSL2) but not
  macOS — irrelevant here since macOS/MLX is out of scope, worth knowing if
  ported.
