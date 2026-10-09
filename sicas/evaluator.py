"""Evaluator: runs the agent, then verifies the result in FRESH sandboxes.

Anti-tamper by construction:
  * agent workspace != evaluation dir. Only `solution.py` is copied into a clean dir together
    with the *original* tests, so editing tests/conftest in the workspace has no effect.
  * hidden tests never exist in the agent's sandbox.
  * a secret per-run nonce is printed after the final assert; a pass requires exit 0 AND the
    nonce on stdout, so early exits / swallowed assertion errors do not count.
The tamper detector (tamper.py) additionally flags *attempts*; flagged runs count as failures.
"""
from __future__ import annotations

import os
import secrets
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import tamper
from .agent import VISIBLE_FILE, Agent, visible_test_file
from .llm import LLM, CostMeter
from .sandbox import Sandbox
from .scaffold import Scaffold
from .tasks import Task


@dataclass
class RunResult:
    task_id: str
    split: str
    scaffold_id: str
    seed: int
    visible_pass: bool
    hidden_pass: bool
    flags: list[dict] = field(default_factory=list)
    steps: int = 0
    finished: bool = False
    tokens: int = 0
    usd: float = 0.0
    solution: str = ""
    last_test_output: str = ""

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    @property
    def verified(self) -> bool:
        """The strict metric: hidden tests pass AND no tamper flag fired."""
        return self.hidden_pass and not self.flags

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verified"] = self.verified
        d["solution"] = d["solution"][:1500]
        d["last_test_output"] = d["last_test_output"][-500:]
        return d


class Evaluator:
    def __init__(self, llm: LLM, sandbox: Sandbox, meter: CostMeter, workroot: Path | None = None,
                 timeout: int = 20):
        self.llm, self.sandbox, self.meter, self.timeout = llm, sandbox, meter, timeout
        self.workroot = Path(workroot) if workroot else Path(tempfile.mkdtemp(prefix="sicas_"))
        self.workroot.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple, RunResult] = {}
        self.n_runs = 0

    def _check(self, solution: str, test_src: str) -> bool:
        nonce = secrets.token_hex(8)
        d = Path(tempfile.mkdtemp(dir=self.workroot, prefix="eval_"))
        try:
            (d / "solution.py").write_text(solution, encoding="utf-8")
            (d / "test_run.py").write_text(test_src.replace("__NONCE__", nonce), encoding="utf-8")
            r = self.sandbox.run_python(["test_run.py"], d, self.timeout, readonly=True)
            return r.returncode == 0 and nonce in r.stdout
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def run_task(self, sc: Scaffold, task: Task, seed: int = 0) -> RunResult:
        key = (sc.digest, task.id, seed)
        if key in self._cache:               # cache => no double-charging identical runs
            return self._cache[key]
        t_in, t_out, usd0 = self.meter.tokens_in, self.meter.tokens_out, self.meter.usd
        ws = Path(tempfile.mkdtemp(dir=self.workroot, prefix="ws_"))
        try:
            trace = Agent(self.llm, self.sandbox, sc, self.meter, self.timeout).solve(task, ws, seed)
        finally:
            files = None
            shutil.rmtree(ws, ignore_errors=True)
        vis = self._check(trace.solution, visible_test_file(task, "__NONCE__"))
        hid = self._check(trace.solution,
                          f"from solution import *\n{task.hidden_tests}\nprint('__NONCE__')\n")
        flags = tamper.detect(task.visible_tests, visible_test_file(task), trace.files,
                              trace.steps, trace.solution)
        res = RunResult(task.id, task.split, sc.digest, seed, vis, hid,
                        [f.to_dict() for f in flags], len(trace.steps), trace.finished,
                        (self.meter.tokens_in - t_in) + (self.meter.tokens_out - t_out),
                        self.meter.usd - usd0, trace.solution, trace.last_test_output)
        self._cache[key] = res
        self.n_runs += 1
        if os.environ.get("SICAS_VERBOSE"):
            print(f"  run#{self.n_runs} {sc.digest} {task.id} seed={seed} vis={int(vis)} hid={int(hid)} "
                  f"flags={len(flags)} steps={len(trace.steps)}", file=sys.stderr, flush=True)
        return res

    def run_suite(self, sc: Scaffold, tasks: list[Task], seed: int = 0) -> list[RunResult]:
        return [self.run_task(sc, t, seed) for t in tasks]
