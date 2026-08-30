"""The genuinely-multi-turn agentic loop.

For each test case: build the system+user message, call the model, parse the
turn. If it's a tool call: execute the mock tool (or record a hallucination if
the tool/args don't validate), feed the result back as a "tool" role message,
and loop. If it's plain text: that's the final answer (or a clarifying
question), and the trajectory ends. Stop at `max_steps` regardless.

Fairness rules enforced here (per the project brief):
  - identical prompts and tool schemas across every format (both come from
    the same test_cases.yaml + tools.py, loaded once)
  - greedy decoding, fixed seed (enforced in loader.py's generate calls —
    do_sample=False everywhere, seed=0 for llama.cpp)
  - identical step caps across every format (max_steps from the test case,
    not format-dependent)
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Optional

from hf_quant_bench.bench.cases import TestCase
from hf_quant_bench.bench.loader import ModelHandle, build_system_prompt, parse_turn
from hf_quant_bench.bench.tools import call_tool, schemas_for


@dataclasses.dataclass
class StepRecord:
    step_index: int
    role_before_model: str
    raw_model_output: str
    parsed_tool_call: Optional[dict]
    tool_result: Optional[dict]
    is_hallucinated_tool: bool
    prefill_s: float
    decode_s: float
    decode_tokens: int


@dataclasses.dataclass
class Trajectory:
    case_id: str
    steps: list[StepRecord]
    final_text: str
    ended_reason: str  # "answered" | "step_cap" | "error"
    total_wall_s: float
    error: Optional[str] = None


def run_case(handle: ModelHandle, case: TestCase) -> Trajectory:
    tool_schemas = schemas_for(case.available_tools)
    system_prompt = build_system_prompt(tool_schemas)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case.prompt},
    ]

    steps: list[StepRecord] = []
    start = time.monotonic()
    final_text = ""
    ended_reason = "step_cap"
    error = None

    try:
        for step_idx in range(case.max_steps):
            gen = handle.generate(messages, max_new_tokens=384)
            parsed = parse_turn(gen.text)

            if parsed.tool_call is None:
                # Plain text: final answer or clarifying question. Loop ends.
                final_text = parsed.text
                ended_reason = "answered"
                steps.append(
                    StepRecord(
                        step_index=step_idx,
                        role_before_model="assistant",
                        raw_model_output=gen.text,
                        parsed_tool_call=None,
                        tool_result=None,
                        is_hallucinated_tool=False,
                        prefill_s=gen.prefill_s,
                        decode_s=gen.decode_s,
                        decode_tokens=gen.decode_tokens,
                    )
                )
                break

            name = parsed.tool_call["name"]
            args = parsed.tool_call.get("arguments", {})
            is_hallucinated = name == "__MALFORMED__" or name not in case.available_tools
            tool_result = None if is_hallucinated else call_tool(name, args)
            if tool_result is None and not is_hallucinated:
                # Valid-looking tool name but not in the global registry at all.
                is_hallucinated = True

            steps.append(
                StepRecord(
                    step_index=step_idx,
                    role_before_model="assistant",
                    raw_model_output=gen.text,
                    parsed_tool_call={"name": name, "arguments": args},
                    tool_result=tool_result,
                    is_hallucinated_tool=is_hallucinated,
                    prefill_s=gen.prefill_s,
                    decode_s=gen.decode_s,
                    decode_tokens=gen.decode_tokens,
                )
            )

            messages.append({"role": "assistant", "content": gen.text})
            result_payload = tool_result if tool_result is not None else {"error": f"unknown tool or invalid call: {name}"}
            messages.append({"role": "tool", "content": json.dumps(result_payload)})
        else:
            final_text = ""
            ended_reason = "step_cap"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        ended_reason = "error"

    return Trajectory(
        case_id=case.id,
        steps=steps,
        final_text=final_text,
        ended_reason=ended_reason,
        total_wall_s=time.monotonic() - start,
        error=error,
    )
