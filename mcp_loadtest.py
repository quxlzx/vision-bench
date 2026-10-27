#!/usr/bin/env python3
"""Concurrent MCP load test: 4x core OCR + 4x speech transcribe in parallel."""
import json, urllib.request, time, wave, math, struct, sys, pathlib, concurrent.futures as cf

def post(url, payload, sid=None):
    h = {"Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    if sid: h["Mcp-Session-Id"] = sid
    r = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=h)
    resp = urllib.request.urlopen(r, timeout=180)
    sid_out = resp.headers.get("Mcp-Session-Id")
    body = resp.read().decode()
    # SSE or plain json
    if body.startswith("event:") or "\ndata: " in body:
        for line in body.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:]), sid_out
        return None, sid_out
    return (json.loads(body) if body else None), sid_out

def init(port):
    url = f"http://127.0.0.1:{port}/mcp"
    _, sid = post(url, {"jsonrpc":"2.0","id":1,"method":"initialize",
        "params":{"protocolVersion":"2025-03-26","capabilities":{},
                  "clientInfo":{"name":"loadtest","version":"1"}}})
    post(url, {"jsonrpc":"2.0","method":"notifications/initialized"}, sid)
    return url, sid

def call(port, tool, args):
    url, sid = init(port)
    t0 = time.time()
    out, _ = post(url, {"jsonrpc":"2.0","id":2,"method":"tools/call",
                        "params":{"name":tool,"arguments":args}}, sid)
    dt = time.time() - t0
    ok = bool(out) and "error" not in out
    text = ""
    if ok:
        c = out["result"].get("content", [])
        text = (c[0].get("text","") if c else "")[:60]
    return {"port": port, "tool": tool, "ok": ok, "s": round(dt,2), "peek": text}

# make a 6s speech-ish wav (sine sweep; ASR will return empty/garbage but load is real)
wav = pathlib.Path.home()/"hanni-capabilities/data/loadtest.wav"
with wave.open(str(wav),"w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", int(12000*math.sin(2*math.pi*(200+400*t/6)*t))) for t in [i/16000 for i in range(96000)]))

mode = sys.argv[1]
ports = [int(p) for p in sys.argv[2:]]
tool, args = ("image_ocr", {"path": "mobo.png"}) if mode=="core" else ("transcribe_audio", {"path": "loadtest.wav"})
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=len(ports)) as ex:
    for r in ex.map(lambda p: call(p, tool, args), ports):
        print(r)
print("wall", round(time.time()-t0,1), "s")
