"""Make a validated, split subset of a converted JSONL (e.g. HumanEval+) for budget-limited runs.

    python scripts/make_subset.py --in data/humaneval_plus.jsonl --out data/he_subset.jsonl \
        --train 20 --val 15 --test 15 --reg 10 --seed 0

Each candidate task is kept only if its CANONICAL solution passes both the visible and hidden tests
inside the sandbox (guards against broken conversions / float or nondeterminism issues).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sicas.agent import visible_test_file  # noqa: E402
from sicas.evaluator import Evaluator  # noqa: E402
from sicas.llm import CostMeter  # noqa: E402
from sicas.sandbox import get_sandbox  # noqa: E402
from sicas.tasks import Task, load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--val", type=int, default=15)
    ap.add_argument("--test", type=int, default=15)
    ap.add_argument("--reg", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sandbox", default="auto")
    a = ap.parse_args()

    from evalplus.data import get_human_eval_plus
    canon = {k.replace("/", "_"): v for k, v in get_human_eval_plus().items()}
    tasks = load_jsonl(a.inp)
    random.Random(a.seed).shuffle(tasks)
    ev = Evaluator(None, get_sandbox(a.sandbox), CostMeter(), Path(tempfile.mkdtemp()))
    need = a.train + a.val + a.test + a.reg
    good: list[Task] = []
    for t in tasks:
        if len(good) >= need:
            break
        sol = t.prompt + canon[t.id]["canonical_solution"]
        ok = (ev._check(sol, visible_test_file(t, "__NONCE__")) and
              ev._check(sol, f"from solution import *\n{t.hidden_tests}\nprint('__NONCE__')\n"))
        print(t.id, "ok" if ok else "SKIP (canonical fails)", flush=True)
        if ok:
            good.append(t)
    splits = (["train"] * a.train + ["val"] * a.val + ["test"] * a.test + ["reg"] * a.reg)
    rows = []
    for t, s in zip(good, splits):
        d = {k: getattr(t, k) for k in ("id", "prompt", "entry_point", "visible_tests", "hidden_tests")}
        d["split"] = s
        rows.append(d)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
