"""Multi-method, multi-seed experiment runner + summary tables (mean +- std, bootstrap CIs)."""
from __future__ import annotations

import json
from pathlib import Path

from .dashboard import build as build_dashboard
from .evolve import METHODS, EvolveConfig, Evolver
from .gate import GateConfig
from .scaffold import COMPONENTS
from .stats import bootstrap_ci, mean_std


def run_experiment(tasks, llm, sandbox, out_dir, methods=METHODS, seeds=(0, 1, 2), generations=6,
                   n_candidates=3, budget_tokens=None, budget_usd=None, components=COMPONENTS,
                   gate_cfg: dict | None = None, log=print) -> dict:
    out = Path(out_dir)
    runs: dict[str, list[dict]] = {m: [] for m in methods}
    for m in methods:
        for s in seeds:
            cfg = EvolveConfig(method=m, generations=0 if m == "fixed" else generations,
                               n_candidates=n_candidates, seed=s, components=tuple(components),
                               budget_tokens=budget_tokens, budget_usd=budget_usd,
                               gate=GateConfig(**(gate_cfg or {})))
            d = out / m / f"seed{s}"
            r = Evolver(cfg, tasks, llm, sandbox, d).run()
            build_dashboard(d)
            runs[m].append(r)
            f = r["final"]
            log(f"[{m:9s} seed={s}] test={f['test']['verified']:.2f} reg={f['reg']['verified']:.2f} "
                f"flagged_runs={r['evolution']['flagged_runs']}/{r['evolution']['runs']} "
                f"accepted={r['evolution']['accepted']}/{r['evolution']['proposed']} "
                f"cost=${r['evolution']['usd']:.4f}")
    summary = summarize(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def summarize(runs: dict[str, list[dict]]) -> dict:
    table = {}
    base = {r["seed"]: r["final"]["test"]["verified"] for r in runs.get("fixed", [])}
    for m, rs in runs.items():
        if not rs:
            continue
        test = [r["final"]["test"]["verified"] for r in rs]
        # bootstrap over (task x seed) units on the held-out test split
        units = [float(t["verified"]) for r in rs for t in r["final_runs"]["test"]]
        lo, hi = bootstrap_ci(units)
        reg_fail = [1 - r["final"]["reg"]["verified"] for r in rs]
        tamper = [r["evolution"]["flagged_runs"] / max(1, r["evolution"]["runs"]) for r in rs]
        solved = [r["final"]["test"]["verified"] * r["final"]["test"]["n"] for r in rs]
        cost = [(r["evolution"]["usd"] + r["test_cost_usd"]) / max(1.0, s) for r, s in zip(rs, solved)]
        gain = [r["final"]["test"]["verified"] - base[r["seed"]] for r in rs if r["seed"] in base]
        gens = [max(1, len(r["evolution"]["generations"])) for r in rs if r["seed"] in base]
        usd = [r["evolution"]["usd"] for r in rs if r["seed"] in base]
        acc = sum(r["evolution"]["accepted"] for r in rs)
        prop = sum(r["evolution"]["proposed"] for r in rs)
        table[m] = {
            "n_seeds": len(rs), "test_pass": mean_std(test), "test_ci95": (lo, hi),
            "test_hidden_raw": mean_std([r["final"]["test"]["hidden"] for r in rs]),
            "regression_rate": mean_std(reg_fail), "tamper_rate": mean_std(tamper),
            "cost_per_solved_usd": mean_std(cost), "evo_usd": mean_std([r["evolution"]["usd"] for r in rs]),
            "accepted_per_proposed": acc / max(1, prop),
            "gain_vs_fixed": mean_std(gain),
            "gain_per_generation": mean_std([g / n for g, n in zip(gain, gens)]),
            "gain_per_dollar": mean_std([g / u for g, u in zip(gain, usd) if u > 0]),
        }
    return table


def to_markdown(t: dict) -> str:
    f = lambda ms: f"{ms[0]:.3f} ± {ms[1]:.3f}"
    lines = ["| method | test pass@1 (verified) | 95% CI | regression rate | tamper-run rate | "
             "cost / solved ($) | gain vs fixed | gain / gen | gain / $ | accept rate |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in t.items():
        lines.append(f"| {m} | {f(v['test_pass'])} | [{v['test_ci95'][0]:.2f}, {v['test_ci95'][1]:.2f}] | "
                     f"{f(v['regression_rate'])} | {f(v['tamper_rate'])} | {f(v['cost_per_solved_usd'])} | "
                     f"{f(v['gain_vs_fixed'])} | {f(v['gain_per_generation'])} | {f(v['gain_per_dollar'])} | "
                     f"{v['accepted_per_proposed']:.2f} |")
    return "\n".join(lines) + "\n"
