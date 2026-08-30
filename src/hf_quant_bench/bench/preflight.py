"""Per-format preflight validation gate.

Exists specifically because of an incident this project hit: `gptq-4bit`
scored 0.29 identically on tool_correctness, argument_correctness, and
behavior_class_accuracy after a full 567s calibration + a full 100-case
benchmark sweep. Root cause (confirmed by dumping raw model output,
diffing chat templates against fp16, verifying calibration data reached the
quantizer, and comparing perplexity against fp16): the model was numerically
fine (perplexity matched fp16 almost exactly) but had genuinely lost the
ability to emit the prompted `<tool_call>{...}</tool_call>` format --
71 of 75 tool-call-expected cases got plain prose describing intent
("I will call get_weather...") instead of a parseable tool call, with zero
malformed attempts either. A 3-case preflight against tool-call-expected
cases specifically would have shown zero parsed tool calls and a
steps-per-trajectory near 1.0 in well under two minutes, instead of after
burning the full calibration time and the full sweep.

The one thing that made the original run's numbers *look* less obviously
broken than they were: no_tool/clarify cases trivially score correct for a
model that never calls a tool at all. A preflight sample drawn from the
full case mix would inherit that exact blind spot. `select_preflight_cases`
therefore ONLY draws from tool_call-expected cases -- never no_tool/clarify.
"""

from __future__ import annotations

import dataclasses
import hashlib

from hf_quant_bench.bench.cases import TestCase
from hf_quant_bench.bench.loader import ModelHandle
from hf_quant_bench.bench.runner import run_case

MIN_MEAN_STEPS = 1.2


@dataclasses.dataclass
class PreflightResult:
    format_name: str
    chat_template_hash: str | None
    mean_steps: float
    any_tool_call_parsed: bool
    raw_outputs: list[str]


def select_preflight_cases(cases: list[TestCase], n: int = 3) -> list[TestCase]:
    """Deliberately tool_call-expected cases ONLY -- see module docstring
    for why a mixed sample would silently defeat the whole point of this
    gate.
    """
    tool_cases = [c for c in cases if c.expected_behavior == "tool_call"]
    if len(tool_cases) < n:
        raise RuntimeError(
            f"preflight gate needs at least {n} tool_call-expected cases in the loaded "
            f"test suite, found only {len(tool_cases)}. Fix the test suite or lower `n` "
            "explicitly -- do not silently skip the gate."
        )
    return tool_cases[:n]


def chat_template_hash(handle: ModelHandle) -> str | None:
    """None means "could not determine" (e.g. GGUF's llama.cpp backend has
    no exposed `tokenizer.chat_template` the way a transformers handle
    does), which the caller must treat as "cannot compare," never as
    "matches."
    """
    tok = getattr(handle, "tokenizer", None)
    if tok is None:
        return None
    template = getattr(tok, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(template.encode()).hexdigest()


def run_preflight(handle: ModelHandle, format_name: str, preflight_cases: list[TestCase]) -> PreflightResult:
    steps_counts = []
    any_tool = False
    raw_outputs = []
    for case in preflight_cases:
        traj = run_case(handle, case)
        steps_counts.append(len(traj.steps))
        raw_outputs.append(traj.steps[0].raw_model_output if traj.steps else "")
        if any(s.parsed_tool_call and s.parsed_tool_call["name"] != "__MALFORMED__" for s in traj.steps):
            any_tool = True
    mean_steps = sum(steps_counts) / len(steps_counts) if steps_counts else 0.0
    return PreflightResult(
        format_name=format_name,
        chat_template_hash=chat_template_hash(handle),
        mean_steps=mean_steps,
        any_tool_call_parsed=any_tool,
        raw_outputs=raw_outputs,
    )


def check_against_reference(result: PreflightResult, reference: PreflightResult) -> list[str]:
    """Every applicable check runs and contributes to the list (not just
    the first failure) -- an abort with three independent reasons is more
    actionable than one that stops at the first.
    """
    reasons: list[str] = []

    if not result.any_tool_call_parsed:
        sample = result.raw_outputs[0][:200] if result.raw_outputs else ""
        reasons.append(
            f"zero valid tool calls parsed across {len(result.raw_outputs)} preflight cases, "
            f"all of which expect one. Raw output sample: {sample!r}"
        )

    if result.mean_steps < MIN_MEAN_STEPS:
        reasons.append(
            f"mean steps-per-trajectory {result.mean_steps:.2f} is below the {MIN_MEAN_STEPS} "
            f"floor (fp16 reference: {reference.mean_steps:.2f}) -- consistent with the model "
            "ending after a single text turn on every case instead of executing a tool and "
            "continuing"
        )

    if result.raw_outputs and all(not o.strip() for o in result.raw_outputs):
        reasons.append("raw model output was empty/whitespace-only across all preflight cases")

    if (
        reference.chat_template_hash is not None
        and result.chat_template_hash is not None
        and result.chat_template_hash != reference.chat_template_hash
    ):
        reasons.append(
            f"chat_template hash differs from the fp16 reference "
            f"({result.chat_template_hash[:12]}... vs {reference.chat_template_hash[:12]}...) -- "
            "this format's tokenizer renders a different prompt than fp16 saw, so any behavioral "
            "difference is confounded with a prompt difference, not just quantization"
        )

    return reasons
