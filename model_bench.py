#!/usr/bin/env python3
"""Model bake-off: HANNI routing (improved) + llama-server tool-calling.
One model at a time; holdout corpora only; cold/warm latency; peak RSS."""
import json, os, subprocess, sys, time, pathlib, statistics, signal
import urllib.request

HOME = pathlib.Path.home()
REPO = HOME / "hanni-repo"
VB = HOME / "vision-bench"
SERVER = str(VB / "llama.cpp/build-cpu/bin/llama-server")
PORT = 8310
sys.path.insert(0, str(REPO / "src"))

def routing_host():
    import sys as _s; _s.path.insert(0, str(REPO / "src"))
    from hanni.config.loader import load_config
    from hanni.tools.registry import build_default_registry
    from hanni.mcp.host import McpHost
    from hanni.evals.isolation import isolated_host
    cfg = load_config(REPO / "config", data_dir=REPO / "data/eval-routing", env={})
    reg = build_default_registry(cfg)
    all_names = tuple(sorted(reg._tools))
    return isolated_host(McpHost(reg, read_only=False, max_tools=8, allowed_actions=all_names,
                                 timezone_name="America/New_York"))

def tool_defs(tools):
    return [t.openai_schema() for t in tools]

def req(payload, timeout=900):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                               data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def peak_rss(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1]) // 1024  # MB
    except OSError:
        return -1

MODELS = {
    "Qwen3.5-9B-Q4_K_M": "Qwen3.5-9B-Q4_K_M.gguf",
    "Qwen3.5-4B-Q4_K_M": "Qwen3.5-4B-Q4_K_M.gguf",
    "Qwen3.5-2B-Q4_K_M": "Qwen3.5-2B-Q4_K_M.gguf",
    "Qwen3-30B-A3B-Q4_K_M (MoE, CPU-only)": "Qwen3-30B-A3B-Q4_K_M.gguf",
}

def run_model(key, fname, cases):
    model = VB / "models" / fname
    if not model.exists():
        return {"model": key, "error": "model file absent (download failed/omitted)"}
    logf = open(f"/tmp/mb-{fname}.log", "w")
    ctx = "4096" if "30B" in key else "8192"  # 30B-A3B: 22 GB RAM ceiling, shrink KV
    p = subprocess.Popen([SERVER, "-m", str(model), "--host", "127.0.0.1", "--port", str(PORT),
                          "-c", ctx, "-ngl", "0", "-t", "8", "--jinja", "--cont-batching",
                          "--parallel", "1", "--no-warmup"], stdout=logf, stderr=logf)
    host = routing_host()
    try:
        t0 = time.time(); cold_done = None
        while time.time() - t0 < 900:
            try:
                if req.__call__ and json.loads(urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/health", timeout=5).read()).get("status") == "ok":
                    cold_done = time.time() - t0; break
            except Exception: time.sleep(3)
        if cold_done is None:
            return {"model": key, "error": f"server failed to load in 900s (see /tmp/mb-{fname}.log)"}
        lat, scores, argfail, parsefail, calls0 = [], [], 0, 0, 0
        SYS = ("You are HANNI, a local assistant with tools. When the user's request matches an "
               "available tool, CALL the best-matching tool with precise arguments. Call multiple "
               "tools only when the request needs multiple. If no tool fits (chat, education, "
               "general knowledge), answer in plain text without calling anything.")
        for i, case in enumerate(cases):
            tools = host.select_tools(case["query"])
            td = tool_defs(tools)
            wire2cap = {t.wire_name: t.capability_id for t in tools}
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": case["query"]}]
            t1 = time.time()
            try:
                try:
                    r = req({"messages": msgs, "tools": td or None, "max_tokens": 200,
                             "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}})
                except Exception:
                    r = req({"messages": msgs, "tools": td or None, "max_tokens": 200, "temperature": 0.0})
            except Exception:
                lat.append(time.time() - t1); scores.append(0); parsefail += 1; continue
            dt = time.time() - t1; lat.append(dt)
            ch = r["choices"][0]["message"]
            called, valid = [], True
            for tc in (ch.get("tool_calls") or []):
                try:
                    json.loads(tc["function"]["arguments"] or "{}")
                    called.append(wire2cap.get(tc["function"]["name"], tc["function"]["name"]))
                except (json.JSONDecodeError, KeyError):
                    valid = False; parsefail += 1
            if not valid: argfail += 1
            reqd = set(case.get("required") or []); acc = set(case.get("acceptable") or [])
            blocked = case.get("mode") == "blocked" or (not reqd and not acc and case.get("expected_intent") in ("", None) and case.get("required", []) == [])
            ok = False
            if case.get("required"):
                ok = reqd.issubset(set(called)) and (not acc or set(called).issubset(acc))
            elif acc:
                ok = bool(set(called) & acc) or not called
            else:  # false-positive audit: must call nothing (or ask)
                ok = not called; calls0 += 0 if not called else 1
            scores.append(1 if ok else 0)
        warm = lat[1:] or lat
        rep = {"model": key, "load_s": round(cold_done, 1),
               "cold_latency_s": round(lat[0], 2),
               "median_latency_s": round(statistics.median(warm), 2),
               "p95_latency_s": round(sorted(warm)[max(0, int(len(warm)*0.95)-1)], 2),
               "peak_rss_mb": peak_rss(p.pid), "offload": "all layers CPU (no GPU access)",
               "cases": len(cases), "tool_call_accuracy": round(100*sum(scores)/len(scores), 1),
               "bad_argument_json": argfail, "parse_failures": parsefail}
        return rep
    finally:
        p.send_signal(signal.SIGINT); p.wait(timeout=30)

if __name__ == "__main__":
    holdout = [json.loads(l) for l in open(REPO / "evals/routing/language_generalization_v4b.jsonl")]
    fp = [json.loads(l) for l in open(REPO / "evals/routing/false_positive_v4.jsonl")]
    only = sys.argv[1:] or list(MODELS)
    out = []
    for key in only:
        print("===", key, flush=True)
        try:
            r1 = run_model(key, MODELS[key], holdout)
            r2 = run_model(key, MODELS[key], fp) if "error" not in r1 else {"error": "skip"}
            r1["fp_audit_accuracy"] = r2.get("tool_call_accuracy")
            r1["fp_cases"] = r2.get("cases")
        except Exception as e:
            r1 = {"model": key, "error": f"{type(e).__name__}: {str(e)[:180]}"}
        print(json.dumps(r1, indent=1), flush=True)
        out.append(r1)
        (VB / "model_results.json").write_text(json.dumps(out, indent=1))
