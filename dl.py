import sys
from huggingface_hub import snapshot_download
specs = [
    ("ggml-org/Qwen2.5-VL-3B-Instruct-GGUF", ["Q4_K_M/*Q4_K_M.gguf", "*mmproj*f16.gguf"]),
    ("ggml-org/Qwen2.5-VL-7B-Instruct-GGUF", ["Q4_K_M/*Q4_K_M.gguf", "*mmproj*f16.gguf"]),
    ("ggml-org/Qwen3-VL-4B-Instruct-GGUF", ["Q4_K_M/*Q4_K_M.gguf", "*mmproj*f16.gguf"]),
    ("ggml-org/moondream2.1-GGUF", ["*Q4_K_M.gguf", "*mmproj*f16.gguf"]),
]
for repo, patterns in specs:
    try:
        p = snapshot_download(repo, allow_patterns=patterns,
                              local_dir="~/vision-bench/models".replace("~", str(__import__("pathlib").Path.home())))
        print("OK", repo, p)
    except Exception as e:
        print("SKIP", repo, type(e).__name__, str(e)[:120])
print("DONE")
