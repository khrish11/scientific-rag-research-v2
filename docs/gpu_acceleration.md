# GPU Acceleration (verified on local RTX 4060 Laptop, 8GB)

Env: `.venv\Scripts\python.exe`, Python 3.13.5, torch 2.10.0+cu126,
CUDA 12.6, driver 566.07, sentence-transformers 6.3.0.
`torch.cuda.is_available()=True`; matmul verified on cuda:0.

DenseRetriever(device=...): "auto"->cuda if available else cpu;
"cpu"/"cuda" explicit (cuda raises naturally if unavailable at encode time).
Batch size configurable (default 64); cache key = model|device|batch|content
hash -> automatic invalidation. FP32 reference precision (no fp16).
BM25/fusion/eval/bootstrap/figures stay on CPU.

## Benchmark (2000 real validation paragraphs, uncached, all-MiniLM-L6-v2)
| device | batch | s | passages/s | peak VRAM |
|---|---|---|---|---|
| cpu | 64 | 48.04 | 41.6 | - |
| cuda | 16 | 4.09 | 488.6 | 177 MB |
| cuda | 32 | 3.33 | 600.4 | 254 MB |
| cuda | 64 | 3.51 | 570.1 | 408 MB |
| cuda | 128 | 3.49 | 572.3 | 715 MB |

Best throughput: batch=32 (600 p/s, ~14.4x CPU). Batch 128 uses 3x VRAM for
no gain -> default kept at 64 for memory safety; experiments used bs=32.
Consistency CPU vs GPU: max |sim diff| = 4.6e-7; top-20 rankings identical
on all test queries. Results: results/gpu_benchmark.json.
Phase-3 adaptive experiment: 812 questions, 4 methods, ~66s on GPU.
