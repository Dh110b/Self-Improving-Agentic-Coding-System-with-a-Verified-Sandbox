"""Convert HumanEval+ / MBPP+ (via the `evalplus` package) to this project's JSONL schema.

    pip install evalplus
    python scripts/convert_evalplus.py --dataset humaneval --out data/humaneval_plus.jsonl

BEST-EFFORT: not exercised in this repo's CI (needs the evalplus package + its dataset download).
Check a few rows by hand before running real experiments.

Visible tests = first --n-visible base inputs; hidden tests = all remaining base + plus inputs
(capped at --max-hidden). Expected outputs come from evalplus' canonical solution. Cases whose
output is not a plain Python literal are skipped. Splits are assigned by hash at load time
(tasks.split_by_hash) unless you add a "split" field.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

HELPER = '''import math as _m
def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and _m.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b
'''


def literal(x) -> bool:
    try:
        return ast.literal_eval(repr(x)) == x
    except Exception:
        return False


def build_cases(prompt, solution, entry, inputs):
    ns: dict = {}
    exec(prompt + solution, ns)  # noqa: S102  (trusted canonical solutions; run locally)
    fn, cases = ns[entry], []
    for args in inputs:
        try:
            out = fn(*copy.deepcopy(args))
        except Exception:
            continue
        if literal(out) and literal(list(args)):
            cases.append((args, out))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-visible", type=int, default=3)
    ap.add_argument("--max-hidden", type=int, default=200)
    a = ap.parse_args()
    if a.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus as get
    else:
        from evalplus.data import get_mbpp_plus as get
    rows = []
    for tid, p in get().items():
        entry, prompt = p["entry_point"], p["prompt"]
        cases = build_cases(prompt, p["canonical_solution"], entry, list(p["base_input"]) + list(p["plus_input"]))
        if len(cases) <= a.n_visible:
            continue
        vis = [f"assert {entry}({', '.join(map(repr, c[0]))}) == {c[1]!r}" for c in cases[: a.n_visible]]
        hid = [f"assert _eq({entry}({', '.join(map(repr, c[0]))}), {c[1]!r})" for c in cases[: a.max_hidden]]
        rows.append({"id": tid.replace("/", "_"), "prompt": prompt, "entry_point": entry,
                     "visible_tests": "\n".join(vis), "hidden_tests": HELPER + "\n".join(hid)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tasks to {a.out}")


if __name__ == "__main__":
    main()
