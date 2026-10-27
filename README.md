# vision-bench

Small benchmark harness for local vision-language models (llama.cpp + mtmd/mmproj):
**decode throughput** plus **acuity probes** (terminal OCR, chart reading, spatial layout,
tiny-text serials, table lookup) against synthetic test images that are generated, never scraped.

## What's here
- `bench.py` — start llama-server per model/device, measure throughput, run 5 acuity probes with exact-match keys
- `gen_testimg.py` — generates `testimg/*.png` (the canary strings like `sk-test-HANNI-...` are **synthetic canaries** baked into fake screenshots, not real secrets)
- `dl.py` — downloads the Q4_K_M GGUFs + mmproj projectors from Hugging Face (ggml-org repos)
- `model_bench.py` / `mcp_loadtest.py` / `withgroups.py` — companion bake-offs (tool-call accuracy, MCP load test); `model_bench.py` expects the HANNI agent repo
- `results.json` / `model_results.json` — actual results from the Oct 2026 run

## Headline results (RX 6600 XT / CPU, Oct 2026)
- Qwen3-VL-4B-Q4: best acuity (5/5 probes) at ~40 tok/s class — now the fleet's vision sidecar
- Qwen2.5-VL-3B: 4/5 (misses dense-terminal line 5), fastest
- Qwen3.5 2B/4B/9B (text-only, control): 70-77% tool-call accuracy — why small VL models are fine for *eyes* but not for the *brain*

## Run
```
python dl.py                 # ~10 GB of GGUFs into models/
python gen_testimg.py        # regenerate test images
python bench.py              # needs a llama-server build with mtmd
```
