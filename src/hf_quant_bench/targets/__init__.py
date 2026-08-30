"""Importing this package registers every conversion target as a side effect.

Each target module is imported inside its own try/except so that a broken
module (e.g. a syntax error introduced while adding a new target, or an
unexpected import-time crash in a third-party package) does not take down
`--list` or the whole conversion run. Import failures show up as a permanently
skipped target with the import error recorded as the reason.
"""

from __future__ import annotations

import importlib
import traceback

_TARGET_MODULES = [
    "hf_quant_bench.targets.fp16",
    "hf_quant_bench.targets.bnb",
    "hf_quant_bench.targets.gptq",
    "hf_quant_bench.targets.awq",
    "hf_quant_bench.targets.torchao_int4",
    "hf_quant_bench.targets.gguf",
    "hf_quant_bench.targets.onnx_export",
    "hf_quant_bench.targets.openvino_ir",
    "hf_quant_bench.targets.qai_hub",
]

import_errors: dict[str, str] = {}

for _mod in _TARGET_MODULES:
    try:
        importlib.import_module(_mod)
    except Exception as e:  # pragma: no cover - defensive
        import_errors[_mod] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
