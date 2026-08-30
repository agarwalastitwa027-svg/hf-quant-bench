# Quantization format comparison


Baseline: **fp16**

**All metrics below are deterministic and rule-based** -- tool_correctness, argument_correctness, and hallucination_rate compare against the test suite's known-correct tool sequence; path_optimization compares actual steps taken against each case's known-minimal step count; answer_completeness checks for the literal values the deterministic mock tools returned in the model's final answer (substring match for text, tolerance-based numeric match for numbers). No cloud API, no LLM-as-a-judge, nothing non-reproducible. `*_judged` columns, if present, come from an optional local GGUF model run offline via `--local-judge` and are supplementary --  they never overwrite the primary columns.

**Read this before the decode tok/s column**: gguf-* rows run on the llama.cpp backend; fp16/bnb-* rows run on HF `transformers.generate()`. A large decode-speed gap between them reflects inference *engine* efficiency (llama.cpp's CUDA decode loop has far less Python-level overhead than transformers') at least as much as it reflects the quantization format itself. It is not a clean measurement of "what quantization costs" in isolation -- treat cross-engine speed comparisons here as directional, not definitive.

**Size column for bnb-nf4/bnb-int8 is `n/a` on purpose**: bitsandbytes quantizes the original fp16 checkpoint at load time rather than producing a separate quantized file, so there is no standalone on-disk artifact to compare against formats like GGUF that do. Reporting it as ~0 MB (the size of the tiny saved config) would have falsely made these formats dominate the Pareto frontier below.

| Format | Tool correct. | Arg correct. | Halluc. rate | Behavior acc. | Path opt | Timeout rate | Completeness | Size (MB) | Peak VRAM (MB) | Peak RAM (MB) | Decode tok/s | Traj latency (s) | Steps/traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 0.80 | 0.87 | 0.01 | 0.86 | 0.96 | 0.00 | 0.88 | 2959.6 | 3078 | 2760 | 36.4 | 1.51 | 1.8 |
| bnb-nf4 | 0.79 | 0.83 | 0.01 | 0.84 | 0.97 | 0.00 | 0.89 | n/a | 64 | 2745 | 30.2 | 1.89 | 1.8 |
| bnb-int8 | 0.77 | 0.86 | 0.01 | 0.83 | 0.94 | 0.00 | 0.89 | n/a | 1530 | 2990 | 7.3 | 7.31 | 1.9 |
| gptq-4bit | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | 0.00 | 1.0 |
| awq-4bit | 0.67 | 0.69 | 0.01 | 0.76 | 0.98 | 0.00 | 0.61 | 1215.4 | 3900 | 1113 | 33.9 | 1.36 | 1.6 |
| gguf-Q4_0 | 0.75 | 0.87 | 0.00 | 0.82 | 0.93 | 0.00 | 0.88 | 891.6 | 1340 | 905 | 151.8 | 0.40 | 1.9 |
| gguf-Q4_K_M | 0.78 | 0.87 | 0.01 | 0.89 | 0.94 | 0.00 | 0.86 | 940.4 | 1390 | 940 | 125.2 | 0.42 | 1.9 |
| gguf-Q5_K_M | 0.72 | 0.83 | 0.01 | 0.80 | 0.93 | 0.00 | 0.86 | 1072.9 | 1522 | 1073 | 119.5 | 0.46 | 1.8 |
| gguf-Q8_0 | 0.74 | 0.85 | 0.03 | 0.86 | 0.93 | 0.00 | 0.85 | 1570.3 | 2020 | 1570 | 111.6 | 0.52 | 1.9 |

## Deltas vs. fp16 baseline

| Format | Δ tool correct. | Δ arg correct. | Δ halluc. rate | Δ size (MB) | Δ decode tok/s |
|---|---|---|---|---|---|
| bnb-nf4 | -0.015 ⚠️ | -0.038 ⚠️ | +0.000 = | n/a | -6.2 ⚠️ |
| bnb-int8 | -0.035 ⚠️ | -0.005 ⚠️ | +0.005 ⚠️ | n/a | -29.1 ⚠️ |
| gptq-4bit | -0.800 ⚠️ | -0.870 ⚠️ | -0.005 ✅ | n/a | n/a |
| awq-4bit | -0.130 ⚠️ | -0.180 ⚠️ | +0.005 ⚠️ | -1744.2 ✅ | -2.5 ⚠️ |
| gguf-Q4_0 | -0.050 ⚠️ | +0.000 = | -0.005 ✅ | -2067.9 ✅ | +115.4 ✅ |
| gguf-Q4_K_M | -0.020 ⚠️ | +0.000 = | +0.010 ⚠️ | -2019.2 ✅ | +88.8 ✅ |
| gguf-Q5_K_M | -0.075 ⚠️ | -0.035 ⚠️ | +0.000 = | -1886.7 ✅ | +83.1 ✅ |
| gguf-Q8_0 | -0.060 ⚠️ | -0.020 ⚠️ | +0.020 ⚠️ | -1389.3 ✅ | +75.2 ✅ |

## Calibrated vs. uncalibrated quantization

**Calibration did NOT help here**: mean tool-call correctness across calibrated formats (gptq-4bit, awq-4bit) is 0.34, vs. 0.76 across uncalibrated round-to-nearest formats (fp16, bnb-nf4, bnb-int8, gguf-Q4_0, gguf-Q4_K_M, gguf-Q5_K_M, gguf-Q8_0).

| Format | Calibrated? | Tool correct. | Arg correct. |
|---|---|---|---|
| fp16 | no | 0.80 | 0.87 |
| bnb-nf4 | no | 0.79 | 0.83 |
| bnb-int8 | no | 0.77 | 0.86 |
| gguf-Q4_0 | no | 0.75 | 0.87 |
| gguf-Q4_K_M | no | 0.78 | 0.87 |
| gguf-Q5_K_M | no | 0.72 | 0.83 |
| gguf-Q8_0 | no | 0.74 | 0.85 |
| gptq-4bit | yes | 0.00 | 0.00 |
| awq-4bit | yes | 0.67 | 0.69 |

## Pareto-optimal formats (size vs. tool-call accuracy)

A format is listed here if no other format is simultaneously smaller on disk AND more tool-correct. These are the formats worth actually considering; everything else is strictly dominated by one of these on the size/accuracy tradeoff.

- **fp16**
- **gguf-Q4_0**
- **gguf-Q4_K_M**