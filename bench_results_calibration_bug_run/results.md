# Quantization format comparison


Baseline: **fp16**

**Read this before the decode tok/s column**: gguf-* rows run on the llama.cpp backend; fp16/bnb-* rows run on HF `transformers.generate()`. A large decode-speed gap between them reflects inference *engine* efficiency (llama.cpp's CUDA decode loop has far less Python-level overhead than transformers') at least as much as it reflects the quantization format itself. It is not a clean measurement of "what quantization costs" in isolation -- treat cross-engine speed comparisons here as directional, not definitive.

**Size column for bnb-nf4/bnb-int8 is `n/a` on purpose**: bitsandbytes quantizes the original fp16 checkpoint at load time rather than producing a separate quantized file, so there is no standalone on-disk artifact to compare against formats like GGUF that do. Reporting it as ~0 MB (the size of the tiny saved config) would have falsely made these formats dominate the Pareto frontier below.

| Format | Tool correct. | Arg correct. | Halluc. rate | Behavior acc. | Path opt (1-5) | Completeness (1-5) | Size (MB) | Peak VRAM (MB) | Peak RAM (MB) | Decode tok/s | Traj latency (s) | Steps/traj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 0.80 | 0.87 | 0.01 | 0.86 | n/a | n/a | 2959.6 | 6194 | 3889 | 38.0 | 1.46 | 1.8 |
| bnb-nf4 | 0.79 | 0.83 | 0.01 | 0.84 | n/a | n/a | n/a | 8 | 5012 | 33.2 | 1.76 | 1.8 |
| bnb-int8 | 0.77 | 0.86 | 0.01 | 0.83 | n/a | n/a | n/a | 1616 | 5321 | 8.6 | 6.26 | 1.9 |
| gptq-4bit | 0.31 | 0.31 | 0.00 | 0.31 | n/a | n/a | 1107.7 | 10 | 5321 | 25.8 | 3.03 | 1.1 |
| awq-4bit | 0.69 | 0.67 | 0.00 | 0.71 | n/a | n/a | 1215.4 | 3862 | 5321 | 35.7 | 1.51 | 1.5 |
| gguf-Q4_0 | 0.75 | 0.87 | 0.00 | 0.82 | n/a | n/a | 891.6 | 1340 | 5321 | 159.1 | 0.39 | 1.9 |
| gguf-Q4_K_M | 0.78 | 0.87 | 0.01 | 0.89 | n/a | n/a | 940.4 | 1390 | 5321 | 129.6 | 0.41 | 1.9 |
| gguf-Q5_K_M | 0.72 | 0.83 | 0.01 | 0.80 | n/a | n/a | 1072.9 | 1522 | 5321 | 122.2 | 0.45 | 1.8 |
| gguf-Q8_0 | 0.74 | 0.85 | 0.03 | 0.86 | n/a | n/a | 1570.3 | 2020 | 5321 | 114.2 | 0.51 | 1.9 |

## Deltas vs. fp16 baseline

| Format | Δ tool correct. | Δ arg correct. | Δ halluc. rate | Δ size (MB) | Δ decode tok/s |
|---|---|---|---|---|---|
| bnb-nf4 | -0.015 ⚠️ | -0.038 ⚠️ | +0.000 = | n/a | -4.8 ⚠️ |
| bnb-int8 | -0.035 ⚠️ | -0.005 ⚠️ | +0.005 ⚠️ | n/a | -29.4 ⚠️ |
| gptq-4bit | -0.490 ⚠️ | -0.560 ⚠️ | -0.005 ✅ | -1851.9 ✅ | -12.2 ⚠️ |
| awq-4bit | -0.115 ⚠️ | -0.205 ⚠️ | -0.005 ✅ | -1744.2 ✅ | -2.3 ⚠️ |
| gguf-Q4_0 | -0.050 ⚠️ | +0.000 = | -0.005 ✅ | -2067.9 ✅ | +121.1 ✅ |
| gguf-Q4_K_M | -0.020 ⚠️ | +0.000 = | +0.010 ⚠️ | -2019.2 ✅ | +91.6 ✅ |
| gguf-Q5_K_M | -0.075 ⚠️ | -0.035 ⚠️ | +0.000 = | -1886.7 ✅ | +84.2 ✅ |
| gguf-Q8_0 | -0.060 ⚠️ | -0.020 ⚠️ | +0.020 ⚠️ | -1389.3 ✅ | +76.2 ✅ |

## Pareto-optimal formats (size vs. tool-call accuracy)

A format is listed here if no other format is simultaneously smaller on disk AND more tool-correct. These are the formats worth actually considering; everything else is strictly dominated by one of these on the size/accuracy tradeoff.

- **fp16**
- **gguf-Q4_0**
- **gguf-Q4_K_M**