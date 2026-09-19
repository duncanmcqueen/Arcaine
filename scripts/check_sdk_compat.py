#!/usr/bin/env python3
"""Parse live /v1/systemone responses with the official TypeSafe/Jev SDK.

This is the SDK-compatibility acceptance item from IMPLEMENTATION_SPEC.md
section 7.6: validate request/response fixtures with the official SDK,
including Noul with no confidence, Score indices and legend, structured
criteria, arbitrary IDs, and all Choice options.

Requires the official SDK (Python >= 3.14), e.g.:

  /usr/bin/python3.14 -m venv /tmp/jev-venv
  /tmp/jev-venv/bin/pip install jev
  /tmp/jev-venv/bin/python scripts/check_sdk_compat.py \
      --base http://127.0.0.1:7461 --key local

The SDK's public answer models are tagged by `type`, so a successful
`SystemOneResponse._decode` also checks that no spurious fields break parsing
and that the answer kind is recognized.
"""
import argparse
import json
import urllib.error
import urllib.request

FAILURES = []


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def post(base, path, body, key):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"content-type": "application/json",
                                          "authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def decode_sdk(raw):
    import httpx2
    from typesafe_sdk import SystemOneResponse
    return SystemOneResponse._decode(
        httpx2.Response(200, content=raw, headers={"content-type": "application/json"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7461")
    ap.add_argument("--key", default="local")
    args = ap.parse_args()

    try:
        import typesafe_sdk  # noqa: F401
    except ImportError:
        print("[INCOMPLETE] the official 'typesafe_sdk' package is not installed; "
              "response parsing cannot be checked. This is not a passing result. "
              "This script also covers response parsing only, not official request "
              "serialization; see docs/systemone-results.md.")
        raise SystemExit(2)

    st, raw = post(args.base, "/v1/models", None, args.key)
    model = json.loads(raw)["data"][0]["id"]

    # 1. Golden request: all three primitives, structured Choice/Score content,
    #    arbitrary IDs (including empty), Noul without confidence.
    body = {
        "model": model,
        "state": {"ticket": "Everything is down and we have a demo at noon."},
        "questions": {
            "urgent": {"type": "noul",
                       "instructions": "Does this require a response within an hour?",
                       "criteria": {"true": {"kind": "outage"}, "false": None}},
            "route": {"type": "choice", "instructions": "Choose the team.",
                      "criteria": {"support": "Customer support",
                                   "engineering": {"desc": "outages", "team": "eng"}}},
            "severity": {"type": "score", "instructions": "Rate it.",
                         "criteria": ["Minor", {"level": "Serious"}, "Critical"]},
            "": {"type": "noul", "instructions": "Empty id is a valid key."},
        },
    }
    st, raw = post(args.base, "/v1/systemone", body, args.key)
    check("golden: HTTP 200", st == 200, raw[:200])
    if st == 200:
        resp = decode_sdk(raw)
        check("SDK parses model/usage", resp.model == model
              and resp.usage.input_tokens > 0 and resp.usage.output_tokens == 4)
        check("SDK Noul (no confidence)", "urgent" in resp.nouls
              and "confidence" not in resp.nouls["urgent"].model_dump())
        check("SDK Choice (structured description)",
              "route" in resp.choices and resp.choices["route"].choice in ("support", "engineering"))
        check("SDK Score (structured legend, int keys)",
              "severity" in resp.scores
              and set(resp.scores["severity"].legend) == {0, 1, 2}
              and set(resp.scores["severity"].probabilities) == {0, 1, 2})
        check("SDK parses arbitrary (empty) id", "" in resp.answers)

    # 2. Diagnostics extension must not break SDK parsing.
    body_diag = dict(body, arcaine={"diagnostics": True})
    st, raw = post(args.base, "/v1/systemone", body_diag, args.key)
    check("diagnostics: HTTP 200", st == 200, raw[:200])
    if st == 200:
        resp = decode_sdk(raw)
        check("SDK ignores the namespaced arcaine diagnostics extension",
              set(resp.answers) == {"urgent", "route", "severity", ""})

    # 3. All 255 Choice options parse and are retained by the SDK.
    crit = {f"opt{i:03d}": f"Option {i}" for i in range(255)}
    st, raw = post(args.base, "/v1/systemone",
                   {"model": model, "state": "Pick option 7.",
                    "questions": {"big": {"type": "choice", "instructions": "?",
                                          "criteria": crit}}}, args.key)
    check("255 options: HTTP 200", st == 200, raw[:200])
    if st == 200:
        resp = decode_sdk(raw)
        probs = resp.choices["big"].probabilities
        check("SDK parses all 255 Choice probabilities", set(probs) == set(crit))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} SDK compatibility FAILURES:", *FAILURES, sep="\n  - ")
        raise SystemExit(1)
    print("official SDK compatibility checks passed")


if __name__ == "__main__":
    main()
