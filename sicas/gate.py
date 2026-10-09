"""Verification gate.

gated mode accepts a candidate only if ALL hold:
  G1 val:   verified pass-rate on the hidden validation split strictly improves (> min_gain)
  G2 reg:   no regression on the frozen regression suite (<= max_reg_failures newly failing)
  G3 tamper: zero tamper flags on candidate runs, and the scaffold diff passes the T7 static scan
ungated mode (baseline) accepts if the *visible-test* train score does not drop - i.e. it trusts
tests the agent can see, which is exactly the weakness the gate removes.
Ablation switches let each check be turned off independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import tamper
from .scaffold import diff_text


def rate(results, key) -> float:
    return sum(1 for r in results if key(r)) / len(results) if results else 0.0


@dataclass
class GateConfig:
    mode: str = "gated"            # "gated" | "ungated"
    use_val: bool = True
    use_regression: bool = True
    use_tamper: bool = True
    min_gain: float = 0.0          # val gain must be strictly greater than this
    max_reg_failures: int = 0


@dataclass
class GateDecision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class Gate:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg

    @property
    def key(self):
        """Pass criterion used by checks: strict (tamper-enforced) unless the detector is ablated."""
        return (lambda r: r.verified) if self.cfg.use_tamper else (lambda r: r.hidden_pass)

    def static_check(self, diff: dict) -> list[str]:
        if self.cfg.mode == "gated" and self.cfg.use_tamper:
            return [f"{f.rule}: {f.detail}" for f in tamper.scan_text(diff_text(diff))]
        return []

    def check_val(self, parent_val, cand_val) -> tuple[bool, str, dict]:
        p, c = rate(parent_val, self.key), rate(cand_val, self.key)
        ok = (c - p) > self.cfg.min_gain
        return ok, f"G1 val {p:.2f}->{c:.2f} ({'ok' if ok else 'no gain'})", {"val_parent": p, "val_cand": c}

    def check_regression(self, parent_reg, cand_reg) -> tuple[bool, str, dict]:
        ok_before = {r.task_id for r in parent_reg if self.key(r)}
        broke = [r.task_id for r in cand_reg if r.task_id in ok_before and not self.key(r)]
        ok = len(broke) <= self.cfg.max_reg_failures
        return ok, f"G2 regression: {len(broke)} newly failing {broke}", {"reg_broken": broke}

    def check_tamper(self, cand_runs) -> tuple[bool, str, dict]:
        n = sum(1 for r in cand_runs if r.flagged)
        return n == 0, f"G3 tamper flags: {n}", {"tamper_flags": n}

    def decide_ungated(self, parent_train, cand_train) -> GateDecision:
        p, c = rate(parent_train, lambda r: r.visible_pass), rate(cand_train, lambda r: r.visible_pass)
        ok = c >= p
        return GateDecision(ok, [f"ungated: visible train {p:.2f}->{c:.2f}"],
                            {"train_visible_parent": p, "train_visible_cand": c})
