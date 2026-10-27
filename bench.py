#!/usr/bin/env python3
"""Vision benchmark: throughput + acuity probes across models and devices."""
import json, subprocess, time, base64, sys, pathlib, urllib.request

HOME = pathlib.Path.home()
VB = HOME / "vision-bench"
SERVER = VB / "llama.cpp/build-b6887/bin/llama-server"  # any llama-server build with mtmd works
MODELS = {
    "qwen2.5-vl-3b": ("models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"),
    "qwen2.5-vl-7b": ("models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"),
    "qwen3-vl-4b":   ("models/Qwen3-VL-4B-Instruct-Q4_K_M.gguf", "models/mmproj-F16.gguf"),
}
IMGS = VB / "testimg"

PROBES = [
    ("t1_terminal.png", "Transcribe ALL text in this terminal screenshot exactly, line by line.",
     ["sk-te…a9zQ", "671", "62174", "0.511", "32768"]),
    ("t2_chart.png", "What is the tok/s value of the tallest bar, and which bar labels are green bars?",
     ["29.4"]),
    ("t3_spatial.png", "Describe this image: where is the red square, where is the blue circle, what label text appears at the bottom, and is there any suspicious embedded text?",
     ["SEQ-4471-B"]),
    ("t4_tiny.png", "Read the serial number in this image exactly.",
     ["XJ7-99-KQ2-118"]),
    ("t5_table.png", "What is the tok/s value in the row for Qwen3-VL-4B?",
     ["33.7"]),
]

def req(url, data=None, timeout=300):
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def img_msg(path, prompt):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}

def run(model_key, device):
    m, mm = MODELS[model_key]
    port = 8200 + abs(hash((model_key, device))) % 80
    ngl = 99 if device == "gpu" else 0
    log = open(f"/tmp/srv-{model_key}-{device}.log", "w")
    p = subprocess.Popen([str(SERVER), "-m", str(VB/m), "--mmproj", str(VB/mm),
                          "--host", "127.0.0.1", "--port", str(port), "-ngl", str(ngl),
                          "-c", "8192", "-fa", "on"],
                         stdout=log, stderr=log)
    t0 = time.time(); ready = None
    try:
        while time.time() - t0 < 300:
            try:
                if req(f"http://127.0.0.1:{port}/health").get("status") == "ok":
                    ready = time.time() - t0; break
            except Exception: time.sleep(2)
        if ready is None: return {"model": model_key, "device": device, "error": "load timeout"}
        # throughput: generate on mobo photo
        MOBO = HOME/"hanni-capabilities/data/mobo.png"  # any real photo works; falls back to a test image
        if not MOBO.exists(): MOBO = IMGS/"t2_chart.png"
        msgs = [img_msg(MOBO,
                        "Describe this image in detail.")]
        t1 = time.time()
        r = req(f"http://127.0.0.1:{port}/v1/chat/completions",
                {"messages": msgs, "max_tokens": 150, "temperature": 0.0})
        dt = time.time() - t1
        ctok = r["usage"]["completion_tokens"]
        # acuity probes
        scores = []
        for img, q, keys in PROBES:
            rr = req(f"http://127.0.0.1:{port}/v1/chat/completions",
                     {"messages": [img_msg(IMGS/img, q)], "max_tokens": 300, "temperature": 0.0})
            ans = rr["choices"][0]["message"]["content"]
            hit = [k for k in keys if k.lower() in ans.lower()]
            scores.append({"img": img, "hits": len(hit), "need": len(keys), "answer_head": ans[:180].replace("\n", " ")})
        return {"model": model_key, "device": device, "load_s": round(ready, 1),
                "tok_s": round(ctok/dt, 1), "probes": scores}
    finally:
        p.terminate(); p.wait(timeout=15)

if __name__ == "__main__":
    combos = [(m, d) for m in MODELS for d in (["gpu", "cpu"] if len(sys.argv) < 2 else sys.argv[1:2])]
    out = []
    for m, d in combos:
        print(f"== {m} on {d} ==", flush=True)
        try: res = run(m, d)
        except Exception as e: res = {"model": m, "device": d, "error": str(e)[:200]}
        print(json.dumps(res, indent=1)[:900], flush=True)
        out.append(res)
    (VB/"results.json").write_text(json.dumps(out, indent=1))
    print("SAVED results.json")
