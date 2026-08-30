"""Load a converted model format into a uniform inference interface.

Every backend exposes the same `ModelHandle.generate(prompt, max_new_tokens)`
method returning a `GenResult` with prefill/decode timing, so the benchmark
runner and resource monitor don't need per-format branches anywhere except
here.

Tool-calling protocol note (read this before trusting the numbers): since the
brief requires support for ANY HF causal LM architecture, this project does
NOT rely on any one model family's native tool-calling chat-template format
(Llama-3.1's, Qwen's, and Hermes's formats all differ and not all base
tokenizers ship a tools-aware chat template). Instead we inject a
model-agnostic system prompt that asks the model to emit
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` for a tool call, or
plain text otherwise. This is deliberately close to the Hermes/Qwen-style
convention because it's the most commonly *trained-into* convention across
open instruct models, which gives smaller/quantized models the best chance of
complying — but it is still a prompted convention, not a guarantee. A model
that was never trained on any tool-calling format may just ignore it, and
that is a real, meaningful part of what this benchmark measures (see
README caveats).
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Optional

TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*")

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with access to tools.

Available tools (JSON Schema):
{tool_schemas}

Rules:
- If calling a tool helps answer the user, respond with EXACTLY ONE tool call in this format and nothing else:
  <tool_call>{{"name": "<tool_name>", "arguments": {{...}}}}</tool_call>
- Only use tool names and arguments defined above. Never invent a tool or a parameter that isn't listed.
- If the request is ambiguous and you need more information before you can act, ask a short clarifying question in plain text instead of calling a tool.
- If no tool is needed, just answer directly in plain text.
- After a tool result is given to you, continue the task: call another tool if needed, or give the final answer in plain text.
"""


@dataclasses.dataclass
class GenResult:
    text: str
    prefill_s: float
    decode_s: float
    decode_tokens: int


@dataclasses.dataclass
class ParsedTurn:
    tool_call: Optional[dict]  # {"name":..., "arguments":...} or None
    text: str  # raw text (for non-tool-call turns, this is the answer/clarification)
    raw: str


def _extract_first_json_object(s: str) -> Optional[str]:
    """Return the substring of the first balanced {...} object in `s`, or
    None if braces never balance (truncated generation, no '{' at all).

    Deliberately NOT a regex: greedy quantized models very often stop
    generating right after the closing '}' of the tool-call JSON and never
    emit the closing '</tool_call>' tag at all (their EOS token just fires
    there). A regex anchored on '</tool_call>' would silently misclassify
    every one of those as plain text -- exactly the failure this project
    is supposed to measure, not accidentally launder into the "no tool call"
    bucket. Brace-counting also correctly handles nested objects (e.g.
    {"name": ..., "arguments": {"location": "Tokyo"}}) without the
    backtracking fragility of a lazy regex.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None  # braces never balanced -- truncated mid-object


def parse_turn(raw_text: str) -> ParsedTurn:
    m = TOOL_CALL_OPEN_RE.search(raw_text)
    if not m:
        return ParsedTurn(tool_call=None, text=raw_text.strip(), raw=raw_text)

    json_str = _extract_first_json_object(raw_text[m.end() :])
    if json_str is not None:
        try:
            payload = json.loads(json_str)
            if isinstance(payload, dict) and "name" in payload:
                return ParsedTurn(
                    tool_call={"name": payload.get("name"), "arguments": payload.get("arguments", {})},
                    text="",
                    raw=raw_text,
                )
        except json.JSONDecodeError:
            pass
    # Looked like a tool call (saw '<tool_call>') but the JSON never parsed
    # into {"name": ...} -> treat as a malformed tool call attempt, surfaced
    # to the caller as text so it can be scored as a hallucination/format
    # failure rather than silently dropped.
    return ParsedTurn(tool_call={"name": "__MALFORMED__", "arguments": {}}, text=raw_text.strip(), raw=raw_text)


def build_system_prompt(tool_schemas: list[dict]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(tool_schemas=json.dumps(tool_schemas, indent=2))


class ModelHandle:
    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> GenResult:
        raise NotImplementedError

    def close(self) -> None:
        """Base class is intentionally NotImplementedError, not a silent
        no-op: every subclass below holds a real GPU-resident resource
        (a `transformers` model, an ORT session, an OV model, or a
        llama.cpp context) and MUST drop that reference here before
        `release_gpu_memory()` runs. A no-op `close()` was the root cause
        of Bug 3 (near-zero peak-VRAM readings for whichever format
        happened to need less memory than the previous format's
        never-actually-freed allocation) -- see resource_monitor.py's
        module docstring for the full mechanism. Silently doing nothing
        here again would reintroduce that bug with no warning, so the
        base class raises instead of passing.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement close() to release its GPU resource")


class TransformersHandle(ModelHandle):
    """Backs fp16, bnb-nf4, bnb-int8, gptq-4bit, awq-4bit, torchao-int4 — every
    format that ends up as a `transformers` model object in GPU or CPU memory.

    `extra_generate_kwargs` exists specifically for the KV-cache-quantization
    axis: it's a separate knob from weight quantization (see
    `_load_fp16_kv_cache_quant` below), applied on top of an otherwise
    unmodified fp16 model so its cost can be measured in isolation rather
    than conflated with a weight-quantization format.
    """

    def __init__(self, model, tokenizer, device: str, extra_generate_kwargs: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.extra_generate_kwargs = extra_generate_kwargs or {}

    def close(self) -> None:
        # `del` the attribute, not just `self.model = None` -- reassigning
        # would still leave the tensors reachable if anything else captured
        # `handle.model` directly, and more importantly `del` is the
        # unambiguous "this reference no longer exists" signal that lets
        # release_gpu_memory()'s gc.collect()/empty_cache() actually find
        # zero referrers and free the underlying CUDA allocations.
        del self.model
        del self.tokenizer

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> GenResult:
        import torch

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            # Prefill-only pass to measure prefill latency in isolation.
            self.model(**inputs, use_cache=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        prefill_s = time.monotonic() - t0

        t1 = time.monotonic()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.extra_generate_kwargs,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        decode_s = time.monotonic() - t1

        new_tokens = out[0][input_len:]
        decode_tokens = int(new_tokens.shape[0])
        gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult(text=gen_text, prefill_s=prefill_s, decode_s=decode_s, decode_tokens=decode_tokens)


def _load_transformers_generic(model_dir: Path, model_id_for_tokenizer: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir if (model_dir / "tokenizer_config.json").exists() else model_id_for_tokenizer)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map={"": 0} if device == "cuda" else "cpu")
    model.eval()
    return TransformersHandle(model, tokenizer, device)


def _load_fp16_kv_cache_quant(model_dir: Path, model_id_for_tokenizer: str, device: str, nbits: int) -> TransformersHandle:
    """The unmodified fp16 checkpoint, weights untouched -- only the KV cache
    is quantized during generation. This is a deliberately separate axis from
    weight quantization: it isolates what KV-cache quantization alone costs
    on agentic tool-calling accuracy, rather than conflating it with a
    weight-quantized format. Uses the `quanto` backend (optimum-quanto) via
    transformers' built-in `cache_implementation="quantized"` path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir if (model_dir / "tokenizer_config.json").exists() else model_id_for_tokenizer)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map={"": 0} if device == "cuda" else "cpu")
    model.eval()
    extra_kwargs = {
        "cache_implementation": "quantized",
        "cache_config": {"backend": "quanto", "nbits": nbits},
    }
    return TransformersHandle(model, tokenizer, device, extra_generate_kwargs=extra_kwargs)


def _load_bnb(model_dir: Path, base_model_id: str, quant_type: str) -> TransformersHandle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg_path = model_dir / "bnb_config.json"
    saved_cfg = json.loads(cfg_path.read_text())["config"] if cfg_path.exists() else {}
    if quant_type == "nf4":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=saved_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=saved_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_cfg, device_map={"": 0})
    model.eval()
    return TransformersHandle(model, tokenizer, "cuda")


def _load_torchao_int4(model_dir: Path, base_model_id: str) -> TransformersHandle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchao.quantization import quantize_, Int4WeightOnlyConfig

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map={"": 0})
    quantize_(model, Int4WeightOnlyConfig(group_size=128))
    state_dict = torch.load(model_dir / "model_torchao_int4.pt", map_location="cuda")
    model.load_state_dict(state_dict, assign=True)
    model.eval()
    return TransformersHandle(model, tokenizer, "cuda")


class OnnxHandle(ModelHandle):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def close(self) -> None:
        del self.model
        del self.tokenizer

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> GenResult:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        t0 = time.monotonic()
        _ = self.model(**inputs, use_cache=True)
        prefill_s = time.monotonic() - t0

        t1 = time.monotonic()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        decode_s = time.monotonic() - t1

        new_tokens = out[0][input_len:]
        gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult(text=gen_text, prefill_s=prefill_s, decode_s=decode_s, decode_tokens=int(new_tokens.shape[0]))


def _load_onnx(model_dir: Path) -> OnnxHandle:
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    model = ORTModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return OnnxHandle(model, tokenizer)


class OpenVinoHandle(ModelHandle):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def close(self) -> None:
        del self.model
        del self.tokenizer

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> GenResult:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        t0 = time.monotonic()
        _ = self.model(**inputs)
        prefill_s = time.monotonic() - t0

        t1 = time.monotonic()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        decode_s = time.monotonic() - t1

        new_tokens = out[0][input_len:]
        gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult(text=gen_text, prefill_s=prefill_s, decode_s=decode_s, decode_tokens=int(new_tokens.shape[0]))


def _load_openvino(model_dir: Path, preferred_device: str) -> OpenVinoHandle:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    model = OVModelForCausalLM.from_pretrained(model_dir, device=preferred_device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return OpenVinoHandle(model, tokenizer)


class GgufHandle(ModelHandle):
    def __init__(self, llm):
        self.llm = llm

    def close(self) -> None:
        # llama.cpp allocates GPU memory via its own direct CUDA calls,
        # entirely outside PyTorch's allocator (see resource_monitor.py) --
        # `llama_cpp.Llama.close()` is the only thing that actually releases
        # it. `del self.llm` alone would eventually trigger `__del__`, which
        # also calls close() internally, but relying on GC timing for a
        # multi-GB GPU allocation is exactly the kind of implicit-cleanup
        # bug this whole fix removes; call it explicitly.
        self.llm.close()
        del self.llm

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> GenResult:
        import llama_cpp as llama_cpp_lib

        # llama.cpp tracks real prefill ("prompt eval") vs decode ("eval")
        # time internally via llama_perf_context -- reset the counters
        # before this call so the read afterward reflects only this turn,
        # not the whole session's cumulative KV-cache-reuse history. This
        # replaces a token-count-ratio approximation of the prefill/decode
        # split, which was not a measurement of either quantity, just a
        # guess at how to divide one wall-clock number between them.
        llama_cpp_lib.llama_perf_context_reset(self.llm.ctx)
        out = self.llm.create_chat_completion(messages=messages, max_tokens=max_new_tokens, temperature=0.0, seed=0)
        perf = llama_cpp_lib.llama_perf_context(self.llm.ctx)

        usage = out.get("usage", {})
        decode_tokens = usage.get("completion_tokens", 0) or perf.n_eval
        prefill_s = perf.t_p_eval_ms / 1000.0
        decode_s = perf.t_eval_ms / 1000.0
        text = out["choices"][0]["message"]["content"] or ""
        return GenResult(text=text, prefill_s=prefill_s, decode_s=decode_s, decode_tokens=decode_tokens)


def _load_gguf(gguf_path: Path, n_gpu_layers: int) -> GgufHandle:
    from llama_cpp import Llama

    llm = Llama(model_path=str(gguf_path), n_ctx=4096, n_gpu_layers=n_gpu_layers, verbose=False, seed=0)
    return GgufHandle(llm)


def load_model_for_format(
    format_name: str,
    manifest_entry: dict,
    base_model_id: str,
    device: str,
    igpu_available: bool,
    npu_available: bool,
    gguf_quant_type: str | None = None,
) -> ModelHandle:
    """Single dispatch point the runner calls. Raises on failure — the runner
    catches this per-format and records it, it does not need to be silent
    here.
    """
    out_path = Path(manifest_entry["output_path"])

    if format_name == "fp16":
        return _load_transformers_generic(out_path, base_model_id, device)
    if format_name == "bnb-nf4":
        return _load_bnb(out_path, base_model_id, "nf4")
    if format_name == "bnb-int8":
        return _load_bnb(out_path, base_model_id, "int8")
    if format_name == "gptq-4bit":
        return _load_transformers_generic(out_path, base_model_id, device)
    if format_name == "awq-4bit":
        return _load_transformers_generic(out_path, base_model_id, device)
    if format_name == "torchao-int4":
        return _load_torchao_int4(out_path, base_model_id)
    if format_name in ("fp16-kv4", "fp16-kv2"):
        nbits = 4 if format_name == "fp16-kv4" else 2
        return _load_fp16_kv_cache_quant(out_path, base_model_id, device, nbits=nbits)
    if format_name == "onnx-fp32" or format_name == "onnx-int8-dynamic":
        return _load_onnx(out_path)
    if format_name == "openvino-fp16" or format_name == "openvino-int8":
        preferred = "CPU"
        if format_name.endswith("fp16") and igpu_available:
            preferred = "GPU"
        return _load_openvino(out_path, preferred)
    if format_name == "gguf":
        quant_types = manifest_entry.get("extra", {}).get("quant_types", {})
        qt = gguf_quant_type or next(iter(quant_types), None)
        if qt is None:
            raise RuntimeError("no GGUF quant type available in manifest")
        n_gpu_layers = -1 if device == "cuda" else 0
        return _load_gguf(Path(quant_types[qt]), n_gpu_layers)

    raise RuntimeError(f"no loader implemented for format '{format_name}'")
