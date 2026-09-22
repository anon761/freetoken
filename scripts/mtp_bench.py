#!/usr/bin/env python3
"""MTP speculative-decoding benchmark (client-side, no extra deps).

Talks to a running ``ft serve`` and measures wall-time throughput plus the engine's
MTP accept counters (``/v1/stats`` -> ``mtp.drafted/accepted``) for greedy and/or
sampled chat completions. Use it to compare ``--mtp off`` vs ``on``/``file`` and to
gate the MTP phases in ``the MTP plan``.

    python scripts/mtp_bench.py --base-url http://127.0.0.1:1919 --max-tokens 256 --runs 5
    python scripts/mtp_bench.py --modes greedy --runs 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPT = "Write a detailed, factual paragraph about the history of ocean exploration."


def _post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _model_id(base: str, timeout: float) -> str:
    return _get(f"{base}/v1/models", timeout)["data"][0]["id"]


def _mtp_counters(base: str, timeout: float) -> tuple[int, int]:
    try:
        m = _get(f"{base}/v1/stats", timeout).get("mtp") or {}
        return int(m.get("drafted", 0)), int(m.get("accepted", 0))
    except Exception:
        return 0, 0


def _run_mode(base: str, model: str, mode: str, args) -> dict:
    tps, tok_total, wall_total = [], 0, 0.0
    d0, a0 = _mtp_counters(base, args.timeout)
    for _ in range(args.runs):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if mode == "greedy":
            payload["temperature"] = 0.0
        else:
            payload["temperature"] = args.sampled_temperature
            if args.sampled_top_k:
                payload["top_k"] = args.sampled_top_k
        t0 = time.monotonic()
        resp = _post(f"{base}/v1/chat/completions", payload, args.timeout)
        dt = time.monotonic() - t0
        n = int(resp["usage"]["completion_tokens"])
        tok_total += n
        wall_total += dt
        if dt > 0:
            tps.append(n / dt)
    d1, a1 = _mtp_counters(base, args.timeout)
    drafted, accepted = d1 - d0, a1 - a0
    return {
        "mode": mode,
        "runs": args.runs,
        "completion_tokens": tok_total,
        "wall_s": round(wall_total, 3),
        "tok_per_s": round(tok_total / wall_total, 2) if wall_total else 0.0,
        "tok_per_s_median": round(statistics.median(tps), 2) if tps else 0.0,
        "mtp_drafted": drafted,
        "mtp_accepted": accepted,
        "accept_rate": round(accepted / drafted, 4) if drafted else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MTP speculative-decoding benchmark")
    ap.add_argument("--base-url", default="http://127.0.0.1:1919")
    ap.add_argument("--model", default="", help="served model id (default: first from /v1/models)")
    ap.add_argument("--modes", default="greedy,sampled")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sampled-temperature", type=float, default=0.8)
    ap.add_argument("--sampled-top-k", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args(argv)

    base = args.base_url.rstrip("/")
    model = args.model or _model_id(base, args.timeout)

    for _ in range(args.warmup):
        _post(f"{base}/v1/chat/completions",
              {"model": model, "messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 16, "temperature": 0.0}, args.timeout)

    results = [_run_mode(base, model, m.strip(), args) for m in args.modes.split(",") if m.strip()]
    if args.json:
        print(json.dumps({"model": model, "results": results}, indent=2))
        return 0

    print(f"model={model}  max_tokens={args.max_tokens}  runs={args.runs}")
    print(f"{'mode':8} {'tok/s':>8} {'tok/s(med)':>10} {'drafted':>8} {'accepted':>9} {'accept':>7}")
    for r in results:
        print(f"{r['mode']:8} {r['tok_per_s']:8.2f} {r['tok_per_s_median']:10.2f} "
              f"{r['mtp_drafted']:8d} {r['mtp_accepted']:9d} {r['accept_rate']*100:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
