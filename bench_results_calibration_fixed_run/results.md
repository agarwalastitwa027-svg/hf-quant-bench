# Quantization format comparison


Baseline: **fp16**

**Read this before the decode tok/s column**: gguf-* rows run on the llama.cpp backend; fp16/bnb-* rows run on HF `transformers.generate()`. A large decode-speed gap between them reflects inference *engine* efficiency (llama.cpp's CUDA decode loop has far less Python-level overhead than transformers') at least as much as it reflects the quantization format itself. It is not a clean measurement of "what quantization costs" in isolation -- treat cross-engine speed comparisons here as directional, not definitive.

**Size column for bnb-nf4/bnb-int8 is `n/a` on purpose**: bitsandbytes quantizes the original fp16 checkpoint at load time rather than producing a separate quantized file, so there is no standalone on-disk artifact to compare against formats like GGUF that do. Reporting it as ~0 MB (the size of the tiny saved config) would have falsely made these formats dominate the Pareto frontier below.

| Format | Tool correct. | Arg correct. | Halluc. rate | Behavior acc. | Path opt (1-5) | Completeness (1-5) | Size (MB) | Peak VRAM (MB) | Peak RAM (MB) | Decode tok/s | Traj latency (s) | Steps/traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 0.80 | 0.87 | 0.01 | 0.86 | n/a | n/a | 2959.6 | 6194 | 3892 | 36.0 | 1.55 | 1.8 |
| bnb-nf4 | 0.79 | 0.83 | 0.01 | 0.84 | n/a | n/a | n/a | 8 | 5021 | 34.3 | 1.71 | 1.8 |
| bnb-int8 | 0.77 | 0.86 | 0.01 | 0.83 | n/a | n/a | n/a | 1616 | 5331 | 8.1 | 6.71 | 1.9 |
| gptq-4bit | 0.29 | 0.29 | 0.00 | 0.29 | n/a | n/a | 1107.6 | 10 | 5331 | 24.8 | 2.66 | 1.0 |
| awq-4bit | 0.67 | 0.69 | 0.01 | 0.76 | n/a | n/a | 1215.4 | 3862 | 5331 | 34.3 | 1.36 | 1.6 |
| gguf-Q4_0 | 0.75 | 0.87 | 0.00 | 0.82 | n/a | n/a | 891.6 | 1340 | 5331 | 157.0 | 0.39 | 1.9 |
| gguf-Q4_K_M | 0.78 | 0.87 | 0.01 | 0.89 | n/a | n/a | 940.4 | 1390 | 5331 | 128.5 | 0.41 | 1.9 |
| gguf-Q5_K_M | 0.72 | 0.83 | 0.01 | 0.80 | n/a | n/a | 1072.9 | 1522 | 5331 | 120.6 | 0.46 | 1.8 |
| gguf-Q8_0 | 0.74 | 0.85 | 0.03 | 0.86 | n/a | n/a | 1570.3 | 2020 | 5331 | 113.3 | 0.51 | 1.9 |

## Deltas vs. fp16 baseline

| Format | Δ tool correct. | Δ arg correct. | Δ halluc. rate | Δ size (MB) | Δ decode tok/s |
|---|---|---|---|---|---|
| bnb-nf4 | -0.015 ⚠️ | -0.038 ⚠️ | +0.000 = | n/a | -1.7 ⚠️ |
| bnb-int8 | -0.035 ⚠️ | -0.005 ⚠️ | +0.005 ⚠️ | n/a | -27.9 ⚠️ |
| gptq-4bit | -0.510 ⚠️ | -0.580 ⚠️ | -0.005 ✅ | -1852.0 ✅ | -11.2 ⚠️ |
| awq-4bit | -0.130 ⚠️ | -0.180 ⚠️ | +0.005 ⚠️ | -1744.2 ✅ | -1.7 ⚠️ |
| gguf-Q4_0 | -0.050 ⚠️ | +0.000 = | -0.005 ✅ | -2067.9 ✅ | +121.0 ✅ |
| gguf-Q4_K_M | -0.020 ⚠️ | +0.000 = | +0.010 ⚠️ | -2019.2 ✅ | +92.5 ✅ |
| gguf-Q5_K_M | -0.075 ⚠️ | -0.035 ⚠️ | +0.000 = | -1886.7 ✅ | +84.6 ✅ |
| gguf-Q8_0 | -0.060 ⚠️ | -0.020 ⚠️ | +0.020 ⚠️ | -1389.3 ✅ | +77.3 ✅ |

## Calibrated vs. uncalibrated quantization

**Calibration did NOT help here**: mean tool-call correctness across calibrated formats (gptq-4bit, awq-4bit) is 0.48, vs. 0.76 across uncalibrated round-to-nearest formats (fp16, bnb-nf4, bnb-int8, gguf-Q4_0, gguf-Q4_K_M, gguf-Q5_K_M, gguf-Q8_0).

| Format | Calibrated? | Tool correct. | Arg correct. |
|---|---|---|---|
| fp16 | no | 0.80 | 0.87 |
| bnb-nf4 | no | 0.79 | 0.83 |
| bnb-int8 | no | 0.77 | 0.86 |
| gguf-Q4_0 | no | 0.75 | 0.87 |
| gguf-Q4_K_M | no | 0.78 | 0.87 |
| gguf-Q5_K_M | no | 0.72 | 0.83 |
| gguf-Q8_0 | no | 0.74 | 0.85 |
| gptq-4bit | yes | 0.29 | 0.29 |
| awq-4bit | yes | 0.67 | 0.69 |

## Pareto-optimal formats (size vs. tool-call accuracy)

A format is listed here if no other format is simultaneously smaller on disk AND more tool-correct. These are the formats worth actually considering; everything else is strictly dominated by one of these on the size/accuracy tradeoff.

- **fp16**
- **gguf-Q4_0**
- **gguf-Q4_K_M**