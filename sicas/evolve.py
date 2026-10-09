"""The self-improvement loop: Solve -> Reflect -> Verify -> Gate -> Archive.

Methods (all share the same budget, tasks, splits and seeds):
  fixed      no improvement (generations = 0)
  reflexion  verbal lessons appended to memory, accepted unconditionally (Reflexion-style)
  ungated    any scaffold component may change; accepted if VISIBLE-test train score doesn't drop
  gated      full system: verified val gain + frozen regression suite + tamper checks
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .archive import Archive
from .evaluator import Evaluator, RunResult
from .gate import Gate, GateConfig, GateDecision, rate
from .llm import LLM, BudgetExceeded, CostMeter
from .reflection import Reflector
from .sandbox import Sandbox
from .scaffold import COMPONENTS, Scaffold
from .tasks import Task, by_split

METHODS = ("fixed", "reflexion", "ungated", "gated")


@dataclass
class EvolveConfig:
    method: str = "gated"
    generations: int = 6
    n_candidates: int = 3
    seed: int = 0
    components: tuple[str, ...] = COMPONENTS
    budget_tokens: int | None = None
    budget_usd: float | None = None
    gate: GateConfig = field(default_factory=GateConfig)
    timeout: int = 20


def summarize(rs: list[RunResult]) -> dict:
    return {"n": len(rs), "verified": rate(rs, lambda r: r.verified),
            "hidden": rate(rs, lambda r: r.hidden_pass), "visible": rate(rs, lambda r: r.visible_pass),
            "flagged": rate(rs, lambda r: r.flagged)}


class Evolver:
    def __init__(self, cfg: EvolveConfig, tasks: list[Task], llm: LLM, sandbox: Sandbox,
                 out_dir: str | Path):
        assert cfg.method in METHODS
        self.cfg, self.llm, self.sandbox = cfg, llm, sandbox
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sp = by_split(tasks)
        self.by_id = {t.id: t for t in tasks}
        gcfg = cfg.gate
        gcfg.mode = "gated" if cfg.method == "gated" else "ungated"
        self.gate = Gate(gcfg)
        self.gated = cfg.method == "gated"
        self.metric = self.gate.key if self.gated else (lambda r: r.visible_pass)

    # ---- candidate evaluation ----------------------------------------------------------
    def _eval_candidate(self, ev: Evaluator, cand, cur: dict, frozen: list[Task]):
        seed, new = self.cfg.seed, cand.scaffold
        if not self.gated:
            train = ev.run_suite(new, self.sp["train"], seed)
            dec = (GateDecision(True, ["reflexion: unconditional"]) if self.cfg.method == "reflexion"
                   else self.gate.decide_ungated(cur["train"], train))
            return dec, {"train": train}, rate(train, lambda r: r.visible_pass)
        reasons = self.gate.static_check(cand.diff)
        if reasons:
            return GateDecision(False, reasons), {}, -1.0
        res: dict = {}
        res["val"] = ev.run_suite(new, self.sp["val"], seed)
        ok1, m1, d1 = self.gate.check_val(cur["val"], res["val"])
        metrics = dict(d1)
        if self.gate.cfg.use_val and not ok1:
            return GateDecision(False, [m1], metrics), res, -1.0
        res["reg"] = ev.run_suite(new, frozen, seed)
        ok2, m2, d2 = self.gate.check_regression(cur["reg"], res["reg"])
        metrics.update(d2)
        if self.gate.cfg.use_regression and not ok2:
            return GateDecision(False, [m1, m2], metrics), res, -1.0
        res["train"] = ev.run_suite(new, self.sp["train"], seed)
        ok3, m3, d3 = self.gate.check_tamper(res["val"] + res["reg"] + res["train"])
        metrics.update(d3)
        if self.gate.cfg.use_tamper and not ok3:
            return GateDecision(False, [m1, m2, m3], metrics), res, -1.0
        # an accepted candidate must still have improved val when use_val is on; otherwise (val
        # ablated) accept iff val does not drop.
        if not self.gate.cfg.use_val and d1["val_cand"] < d1["val_parent"]:
            return GateDecision(False, [m1, m2, m3], metrics), res, -1.0
        score = d1["val_cand"] + 0.01 * rate(res["train"], self.metric)
        return GateDecision(True, [m1, m2, m3], metrics), res, score

    # ---- main loop ---------------------------------------------------------------------
    def run(self) -> dict:
        cfg, seed = self.cfg, self.cfg.seed
        meter = CostMeter(budget_tokens=cfg.budget_tokens, budget_usd=cfg.budget_usd)
        ev = Evaluator(self.llm, self.sandbox, meter, self.out / "_work", cfg.timeout)
        post = Evaluator(self.llm, self.sandbox, CostMeter(), self.out / "_work_post", cfg.timeout)
        arch = Archive(self.out / "archive")
        sc = Scaffold()

        # freeze the regression suite: tasks the *initial* scaffold already solves
        v0_reg = post.run_suite(sc, self.sp["reg"], seed)
        solved = {r.task_id for r in v0_reg if r.verified}
        frozen = [t for t in self.sp["reg"] if t.id in solved] or self.sp["reg"]

        cur: dict = {}
        next_id, cur_id, stop = 1, 0, "generations_done"
        reflector = Reflector(self.llm, meter, ("memory",) if cfg.method == "reflexion"
                              else cfg.components, 1 if cfg.method == "reflexion" else cfg.n_candidates)
        try:
            cur["train"] = ev.run_suite(sc, self.sp["train"], seed)
            if self.gated:
                cur["val"] = ev.run_suite(sc, self.sp["val"], seed)
                cur["reg"] = ev.run_suite(sc, frozen, seed)
        except BudgetExceeded:
            cur.setdefault("train", [])
            stop = "budget_exhausted_at_baseline"
        arch.add({"id": 0, "parent": None, "accepted": True, "scaffold": sc.to_dict(), "diff": None,
                  "decision": {"accepted": True, "reasons": ["initial scaffold"], "metrics": {}},
                  "scores": {k: summarize(v) for k, v in cur.items()}, "tokens": meter.tokens,
                  "usd": meter.usd})
        arch.set_current(0)
        gens_log = []

        for g in range(1, cfg.generations + 1 if stop == "generations_done" else 0):
            # failure mining uses train labels; ACCEPTANCE (self.metric) differs per method
            failures = [r for r in cur["train"] if not (self.metric(r) if self.gated else r.hidden_pass)]
            if not failures:
                stop = "no_failures_left"
                break
            try:
                cands = reflector.propose(sc, failures, self.by_id, seed * 1000 + g)
                evaluated = [(c, *self._eval_candidate(ev, c, cur, frozen)) for c in cands]
            except BudgetExceeded:
                stop = "budget_exhausted"
                break
            ok = [e for e in evaluated if e[1].accepted]
            winner = max(ok, key=lambda e: e[3]) if ok else None
            for c, dec, res, _ in evaluated:
                is_win = winner is not None and c is winner[0]
                if not is_win and dec.accepted:
                    dec = GateDecision(False, dec.reasons + ["outranked by better candidate"], dec.metrics)
                rid = next_id
                next_id += 1
                arch.add({"id": rid, "parent": cur_id, "accepted": is_win, "generation": g,
                          "scaffold": c.scaffold.to_dict(), "diff": c.diff,
                          "decision": {"accepted": is_win, "reasons": dec.reasons, "metrics": dec.metrics},
                          "scores": {k: summarize(v) for k, v in res.items()},
                          "tokens": meter.tokens, "usd": meter.usd})
                if is_win:
                    sc, cur_id = c.scaffold, rid
                    cur = {**cur, **res}
                    arch.set_current(rid)
            gens_log.append({"generation": g, "proposed": len(cands), "accepted": winner is not None,
                             "current_id": cur_id})

        # ---- post-hoc evaluation (separate meter; test split touched exactly once here) ----
        final = {"test": post.run_suite(sc, self.sp["test"], seed),
                 "val": post.run_suite(sc, self.sp["val"], seed),
                 "reg": post.run_suite(sc, frozen, seed)}
        evo_runs = list(ev._cache.values())
        rules = Counter(f["rule"] for r in evo_runs for f in r.flags)
        accepted_n = sum(g["accepted"] for g in gens_log)
        result = {
            "method": cfg.method, "seed": seed, "final_id": cur_id, "stop": stop,
            "final_scaffold": sc.to_dict(), "frozen_reg": [t.id for t in frozen],
            "evolution": {"tokens": meter.tokens, "usd": meter.usd, "runs": len(evo_runs),
                          "flagged_runs": sum(r.flagged for r in evo_runs),
                          "flags_by_rule": dict(rules), "generations": gens_log,
                          "accepted": accepted_n, "proposed": sum(g["proposed"] for g in gens_log)},
            "final": {k: summarize(v) for k, v in final.items()},
            "final_runs": {k: [r.to_dict() for r in v] for k, v in final.items()},
            "test_cost_usd": post.meter.usd,
        }
        (self.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
