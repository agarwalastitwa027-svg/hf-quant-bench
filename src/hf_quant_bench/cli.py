"""Top-level convenience CLI: dispatches to convert or bench.

    hf-quant-bench convert --model ... --out ...
    hf-quant-bench bench --manifest ... --model ... --out ...

Equivalent to calling `python -m hf_quant_bench.convert` / `.bench.run`
directly — this just saves typing once the package is pip-installed.
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("convert", "bench"):
        print("usage: hf-quant-bench {convert,bench} [args...]", file=sys.stderr)
        return 2

    sub = sys.argv[1]
    rest = sys.argv[2:]

    if sub == "convert":
        from hf_quant_bench.convert import main as convert_main

        return convert_main(rest)
    else:
        from hf_quant_bench.bench.run import main as bench_main

        return bench_main(rest)


if __name__ == "__main__":
    sys.exit(main())
