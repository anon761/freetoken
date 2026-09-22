#!/usr/bin/env python3
"""DSpark greedy byte-identity harness (DwarfStar DSPARK-V41 methodology).

Runs the same greedy prompts against two server arms (speculative on vs off) and
reports where they diverge, so acceptance can be judged per prefix length — the
compressed-ratio boundaries are where a verify whose reduction order differs from
serial decode shows up first.

Usage:
    # arm A (spec on) — server started with `--mtp on --dspark-verify decode`
    dspark_greedy_identity.py run --base http://127.0.0.1:1929 --model dsv41 \\
        --tag on --max-tokens 128 --out /tmp/dspark-on.json
    # arm B (spec off) — server restarted with `--mtp off`
    dspark_greedy_identity.py run --base http://127.0.0.1:1929 --model dsv41 \\
        --tag off --max-tokens 128 --out /tmp/dspark-off.json
    dspark_greedy_identity.py compare /tmp/dspark-on.json /tmp/dspark-off.json

Exit code is 1 when any prompt diverges, 0 when all are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# prefix lengths to probe, as repetitions of a filler sentence (~9 tokens each), so
# we straddle the DSV4 compressed-block boundaries (~768/1024) without a tokenizer.
FILLER = "The quick brown fox jumps over the lazy dog. "
REPEATS = [1, 32, 64, 96, 112, 128, 160, 192, 256, 384, 512]
NATURAL = ["Write a haiku about the sea:", "The history of the Roman Empire:", "2+2="]


def _prompts() -> list[str]:
    return [FILLER * n for n in REPEATS] + list(NATURAL)


def _complete(base: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": max_tokens,
         "temperature": 0, "ignore_eos": True}
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        doc = json.load(resp)
    return {"text": doc["choices"][0].get("text"),
            "tokens": doc["usage"]["completion_tokens"]}


def cmd_run(args: argparse.Namespace) -> int:
    out = {"tag": args.tag, "model": args.model, "base": args.base, "results": []}
    for prompt in _prompts():
        try:
            res = _complete(args.base, args.model, prompt, args.max_tokens, args.timeout)
            res["prompt_len_chars"] = len(prompt)
        except Exception as exc:  # noqa: BLE001 — record and continue
            res = {"text": None, "error": f"{type(exc).__name__}: {exc}",
                   "prompt_len_chars": len(prompt)}
        out["results"].append(res)
        print(f"{len(prompt):6d} chars -> {'ERROR' if res.get('text') is None else res['tokens']} tokens")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


def _first_diff(a: str, b: str) -> int | None:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cmd_compare(args: argparse.Namespace) -> int:
    with open(args.a) as fh:
        a = json.load(fh)
    with open(args.b) as fh:
        b = json.load(fh)
    ra, rb = a["results"], b["results"]
    if len(ra) != len(rb):
        print("result count differs — rerun both arms with the same harness", file=sys.stderr)
        return 2
    diverged = 0
    for i, (x, y) in enumerate(zip(ra, rb)):
        if x.get("text") is None or y.get("text") is None:
            print(f"[{i:2d}] prompt~{x['prompt_len_chars']}ch  SKIP (error)")
            continue
        if x["text"] == y["text"]:
            print(f"[{i:2d}] prompt~{x['prompt_len_chars']:6d}ch  IDENTICAL ({x['tokens']} tok)")
            continue
        diverged += 1
        at = _first_diff(x["text"], y["text"])
        print(f"[{i:2d}] prompt~{x['prompt_len_chars']:6d}ch  DIVERGES at char {at}")
        print(f"      {a['tag']}: {x['text'][:90]!r}")
        print(f"      {b['tag']}: {y['text'][:90]!r}")
    print(f"\n{len(ra) - diverged}/{len(ra)} identical, {diverged} diverged "
          f"({a['tag']} vs {b['tag']})")
    return 1 if diverged else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
