#!/usr/bin/env python3
"""End-to-end /v1/systemone benchmark on the running arcaine_server.

Sweeps a filler target (AGENTS.md convention: -p 512,1024,2048,4096) for a
fixed 3-question golden request and reports p50/p95 request latency, the
server-side synchronized prefill/decode split, the per-question MEASURED prompt
token counts and canvas widths (via arcaine diagnostics), and decisions/sec
(scored answer slots / wall second).

The filler text is an approximate length knob, not an exact token count; the
table reports the measured tokens so results are not misread as exact targets.
Canvas widths come from the compiled answer template and are reported, not
swept (the structured canvas is a few tokens wide, not the diffusion canvas).

The server's read policy is a deployment setting (ARCAINE_SYSTEMONE_READS /
ARCAINE_SYSTEMONE_READS_AUTO_*); run one server configuration per leg, e.g.:

  # leg A: 1 read/question (default)
  python3 scripts/benchmark_systemone.py -p 512,1024,2048,4096
  # leg B: 4 reads/question (server started with ARCAINE_SYSTEMONE_READS=4)
  python3 scripts/benchmark_systemone.py -p 512,1024,2048,4096
"""
import argparse
import json
import statistics
import time
import urllib.request


def make_request(model, target_tokens):
    # Roughly 4 chars per token of filler text ahead of the real ticket.
    filler = ("The datacenter report covers routine capacity, power, and "
              "cooling metrics for the quarter. ") * max(0, target_tokens // 15)
    return {
        "model": model,
        "state": {"ticket": filler + " Everything is down and we have a demo at noon."},
        "questions": {
            "urgent": {"type": "noul",
                       "instructions": "Does this require a response within an hour?",
                       "criteria": {"true": "A service outage needs prompt action",
                                    "false": "The request can wait"}},
            "route": {"type": "choice",
                      "instructions": "Choose the responsible team.",
                      "criteria": {"support": "Customer support",
                                   "engineering": "Service outages and software faults"}},
            "severity": {"type": "score", "instructions": "Rate the incident.",
                         "criteria": ["Minor", "Serious", "Critical"]},
        },
        "arcaine": {"diagnostics": True},
    }


def post(base, key, body):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "authorization": "Bearer " + key})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.loads(r.read())
    return (time.perf_counter() - t0) * 1e3, payload


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompts", default="512,1024,2048,4096")
    ap.add_argument("--base", default="http://127.0.0.1:7461")
    ap.add_argument("--key", default="local")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    req = urllib.request.Request(args.base + "/v1/models")
    req.add_header("authorization", "Bearer " + args.key)
    with urllib.request.urlopen(req, timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]

    print(f"model={model} iters={args.iters} warmup={args.warmup}")
    print(f"{'target':>7} {'q_toks':>10} {'canvas':>7} {'reads':>6} "
          f"{'p50_ms':>9} {'p95_ms':>9} {'prefill_ms':>11} {'decode_ms':>10} "
          f"{'decisions/s':>12}")
    for plen in [int(x) for x in args.prompts.split(",")]:
        body = make_request(model, plen)
        lat, prefill, decode = [], [], []
        q_toks_lo = q_toks_hi = 0
        canvas_lo = canvas_hi = 0
        reads = 0
        decisions = 0
        for i in range(args.warmup + args.iters):
            ms, payload = post(args.base, args.key, body)
            if i < args.warmup:
                continue
            lat.append(ms)
            d = payload.get("arcaine", {}).get("diagnostics", {})
            reads = payload["usage"]["output_tokens"]
            decisions = sum(v.get("reads", 0) for v in d.values()) or reads
            toks = [v.get("prompt_tokens", 0) for v in d.values()]
            canvas = [v.get("canvas_width", 0) for v in d.values()]
            q_toks_lo, q_toks_hi = min(toks), max(toks)
            canvas_lo, canvas_hi = min(canvas), max(canvas)
            prefill.append(sum(v["prefill_ms"] for v in d.values()))
            decode.append(sum(v["decode_ms"] for v in d.values()))
        med_ms = statistics.median(lat) if lat else float("nan")
        dps = (decisions / (med_ms / 1e3)) if med_ms > 0 else float("nan")
        q_range = f"{q_toks_lo}-{q_toks_hi}" if q_toks_lo != q_toks_hi else str(q_toks_lo)
        c_range = f"{canvas_lo}-{canvas_hi}" if canvas_lo != canvas_hi else str(canvas_lo)
        print(f"{plen:>7} {q_range:>10} {c_range:>7} {reads:>6} "
              f"{med_ms:>9.1f} {pct(lat, 0.95):>9.1f} "
              f"{statistics.median(prefill):>11.1f} {statistics.median(decode):>10.1f} "
              f"{dps:>12.1f}")


if __name__ == "__main__":
    main()
