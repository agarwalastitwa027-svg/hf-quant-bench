"""Optional, off-by-default supplementary judge -- a local GGUF model run
through llama.cpp, never a cloud API.

This is NOT how path_optimization/answer_completeness are computed by
default; those are deterministic and rule-based (see metrics_rule.py) and
remain the primary metrics regardless of whether this module is used at
all. This exists only for someone who wants a second, model-based opinion
alongside the deterministic ground truth, and is willing to spend the extra
GPU time for it.

Constraints, enforced by how this is wired into run.py's main():
  - Runs OFFLINE, reading the already-written trajectory JSONL files, and
    only AFTER every format's full sweep has completed and every subject
    model has been released. Never concurrently with a subject model --
    an 8GB-VRAM box cannot hold a judge and a subject model loaded at once.
  - Writes `path_optimization_judged` / `answer_completeness_judged` as
    SEPARATE columns (see report.py's FormatSummary) and a `local_judge_score`
    key alongside (not inside) each trajectory JSONL record's existing
    `rule_metrics` -- the judged columns never overwrite the rule-based ones.
  - Judge model name + quantization goes into `backend_note`, not silently
    assumed from context.
"""

from __future__ import annotations

import dataclasses
import json
import re
import statistics
from pathlib import Path

JUDGE_SYSTEM_PROMPT = """You are grading an AI agent's tool-use trajectory. You will be given:
- the user's original request
- the full sequence of the agent's tool calls, tool results, and its final response
- the minimal number of steps a fully correct trajectory would need
- the concrete facts the final answer was required to contain

Score two things, each 1-5 (integers only):
1. path_optimization: Did the agent take an efficient path relative to the stated minimal step count? 5 = matched or came close to the minimal step count with no wasted/repeated calls. 1 = looped, repeated identical calls, or took far more steps than necessary.
2. answer_completeness: Did the final answer actually state the required facts and resolve the user's request? 5 = fully resolves it, states the required facts clearly. 1 = ignores the tool results, omits the required facts, or is empty/cut off.

Respond with ONLY a JSON object, no other text:
{"path_optimization": <int 1-5>, "answer_completeness": <int 1-5>, "rationale": "<one or two sentences>"}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclasses.dataclass
class LocalJudgeScore:
    path_optimization_judged: float | None  # normalized to 0..1 (raw 1-5 score / 5), for direct comparison with the deterministic column
    answer_completeness_judged: float | None
    raw_path_optimization: int | None  # the model's literal 1-5 score, kept for inspection
    raw_answer_completeness: int | None
    rationale: str
    error: str | None = None


def load_local_judge(gguf_path: str):
    """Loads the local judge model. Caller is responsible for calling
    `.close()` on the returned object when done -- see loader.py's
    GgufHandle.close() for why an explicit close matters here (llama.cpp
    allocates GPU memory outside any Python-level reference-counting this
    project's other cleanup already relies on).
    """
    from llama_cpp import Llama

    return Llama(model_path=gguf_path, n_ctx=4096, n_gpu_layers=-1, verbose=False, seed=0)


def _format_trajectory_for_judge(record: dict, optimal_steps: int, required_facts: list[str]) -> str:
    lines = [
        f"User request: {record['prompt'][:2000]}",
        f"Minimal steps for a fully correct trajectory: {optimal_steps}",
        f"Required facts the final answer must state: {required_facts}",
        "",
        "Trajectory:",
    ]
    for s in record["steps"]:
        if s.get("parsed_tool_call"):
            lines.append(f"  -> tool_call: {s['parsed_tool_call']['name']}({s['parsed_tool_call'].get('arguments', {})})")
            lines.append(f"     result: {s.get('tool_result')}")
        else:
            lines.append(f"  -> final/plain text: {s['raw_model_output'][:500]}")
    lines.append(f"\nEnded because: {record['ended_reason']}")
    lines.append(f"Final answer text: {record['final_text'][:1000]}")
    return "\n".join(lines)


def judge_record(llm, record: dict, optimal_steps: int, required_facts: list[str]) -> LocalJudgeScore:
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _format_trajectory_for_judge(record, optimal_steps, required_facts)},
            ],
            max_tokens=200,
            temperature=0.0,
            seed=0,
        )
        text = out["choices"][0]["message"]["content"] or ""
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"no JSON object found in judge output: {text[:200]!r}")
        parsed = json.loads(m.group(0))
        po_raw = int(parsed["path_optimization"])
        ac_raw = int(parsed["answer_completeness"])
        return LocalJudgeScore(
            path_optimization_judged=po_raw / 5.0,
            answer_completeness_judged=ac_raw / 5.0,
            raw_path_optimization=po_raw,
            raw_answer_completeness=ac_raw,
            rationale=parsed.get("rationale", ""),
        )
    except Exception as e:
        return LocalJudgeScore(None, None, None, None, "", error=f"{type(e).__name__}: {e}")


def run_local_judge_on_file(llm, jsonl_path: Path, cases_by_id: dict) -> tuple[float | None, float | None]:
    """Rewrites `jsonl_path` in place, adding a `local_judge_score` key to
    every record, and returns (mean path_optimization_judged, mean
    answer_completeness_judged) across records the judge scored without
    error -- the pair to put in that format's FormatSummary.
    """
    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]

    po_vals, ac_vals = [], []
    for r in records:
        case = cases_by_id.get(r["case_id"])
        if case is None:
            r["local_judge_score"] = dataclasses.asdict(
                LocalJudgeScore(None, None, None, None, "", error=f"unknown case_id '{r['case_id']}'")
            )
            continue
        score = judge_record(llm, r, case.optimal_steps, case.required_facts)
        r["local_judge_score"] = dataclasses.asdict(score)
        if score.path_optimization_judged is not None:
            po_vals.append(score.path_optimization_judged)
        if score.answer_completeness_judged is not None:
            ac_vals.append(score.answer_completeness_judged)

    jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    return (
        statistics.mean(po_vals) if po_vals else None,
        statistics.mean(ac_vals) if ac_vals else None,
    )
