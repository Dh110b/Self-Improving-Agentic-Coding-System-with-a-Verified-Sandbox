"""Agent core: a tool-using loop. Tools execute in the sandbox (or are pure host-side reads)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLM, CostMeter
from .sandbox import Sandbox
from .scaffold import Scaffold, parse_json_object
from .tasks import Task

VISIBLE_FILE = "test_visible.py"
OBS_LIMIT = 2000

_TOOL_RE = re.compile(r'"tool"\s*:\s*"(\w+)"')
_PATH_RE = re.compile(r'"path"\s*:\s*"([^"]+)"')
_CONTENT_RE = re.compile(r'"content"\s*:\s*("{3}|\'{3}|")(.*)\1\s*\}', re.S)
_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)


def parse_action(text: str) -> dict:
    """Strict JSON first; then tolerate common small-model slips (Python triple quotes inside
    the JSON `content`, or a bare ```python block). Applied identically to every method."""
    act = parse_json_object(text)
    if act and isinstance(act.get("tool"), str):
        return act
    m = _TOOL_RE.search(text)
    if m and m.group(1) == "write_file":
        path, body = _PATH_RE.search(text), _CONTENT_RE.search(text)
        if path and body:
            return {"tool": "write_file",
                    "args": {"path": path.group(1), "content": body.group(2).strip("\n") + "\n"}}
    elif m:
        return {"tool": m.group(1), "args": {}}
    code = _FENCE_RE.search(text)
    if code:
        return {"tool": "write_file", "args": {"path": "solution.py", "content": code.group(1)}}
    return act or {}


def visible_test_file(task: Task, nonce: str | None = None) -> str:
    tail = f"print({nonce!r})\n" if nonce else 'print("ALL VISIBLE TESTS PASSED")\n'
    return f"from solution import *\n{task.visible_tests}\n{tail}"


@dataclass
class Trace:
    task_id: str
    steps: list[dict] = field(default_factory=list)
    finished: bool = False
    last_test_output: str = ""
    solution: str = ""
    files: dict[str, str] = field(default_factory=dict)


class Agent:
    def __init__(self, llm: LLM, sandbox: Sandbox, scaffold: Scaffold, meter: CostMeter,
                 timeout: int = 20):
        self.llm, self.sandbox, self.sc, self.meter, self.timeout = (
            llm, sandbox, scaffold, meter, timeout)

    # -- tools ------------------------------------------------------------------------------
    def _safe(self, ws: Path, rel: str) -> Path | None:
        p = (ws / rel).resolve()
        return p if str(p).startswith(str(ws.resolve())) and rel else None

    def _exec(self, ws: Path, tool: str, args: dict, trace: Trace) -> str:
        if tool not in self.sc.tools:
            return f"ERROR: tool '{tool}' is not available. Available: {self.sc.tools}"
        if tool == "view_file":
            p = self._safe(ws, str(args.get("path", "")))
            if not p or not p.is_file():
                return "ERROR: no such file in workspace"
            return p.read_text(encoding="utf-8", errors="replace")[:OBS_LIMIT]
        if tool == "write_file":
            p = self._safe(ws, str(args.get("path", "")))
            if not p:
                return "ERROR: path outside workspace"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {len(str(args.get('content', '')))} chars to {args['path']}"
        if tool == "run_tests":
            r = self.sandbox.run_python([VISIBLE_FILE], ws, self.timeout)
            out = (r.stdout + r.stderr)[-OBS_LIMIT:]
            trace.last_test_output = out
            return f"exit={r.returncode}\n{out}"
        if tool == "shell":
            r = self.sandbox.run_shell(str(args.get("cmd", "")), ws, self.timeout)
            return f"exit={r.returncode}\n{(r.stdout + r.stderr)[-OBS_LIMIT:]}"
        if tool == "search":
            try:
                rx = re.compile(str(args.get("pattern", "")))
            except re.error as e:
                return f"ERROR: bad regex: {e}"
            hits = []
            for f in sorted(ws.glob("*")):
                if f.is_file():
                    for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{f.name}:{i}: {line}")
            return "\n".join(hits[:40]) or "no matches"
        return "ERROR: unknown tool"

    # -- loop -------------------------------------------------------------------------------
    def solve(self, task: Task, ws: Path, seed: int = 0) -> Trace:
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "solution.py").write_text(task.starter, encoding="utf-8")
        (ws / VISIBLE_FILE).write_text(visible_test_file(task), encoding="utf-8")
        trace = Trace(task.id)
        system = self.sc.render_system(task.prompt)
        messages = [{"role": "user", "content":
                     f"TASK_ID: {task.id}\nImplement `{task.entry_point}` in solution.py.\n\n"
                     f"{task.prompt}\nVisible tests are in {VISIBLE_FILE}."}]
        for _ in range(self.sc.max_steps):
            self.meter.check()
            resp = self.llm.chat(system, messages, seed=seed)
            self.meter.add(resp)
            action = parse_action(resp.text)
            tool, args = action.get("tool"), action.get("args") or {}
            messages.append({"role": "assistant", "content": resp.text})
            if tool == "finish":
                trace.steps.append({"tool": "finish", "args": {}})
                trace.finished = True
                break
            if not isinstance(tool, str) or not isinstance(args, dict):
                obs = 'ERROR: reply with one JSON object {"tool": ..., "args": {...}}'
                trace.steps.append({"tool": None, "args": {}, "obs": obs})
            else:
                obs = self._exec(ws, tool, args, trace)
                trace.steps.append({"tool": tool, "args": args, "obs": obs[:300]})
            messages.append({"role": "user", "content": f"OBSERVATION:\n{obs}"})
        sol = ws / "solution.py"
        trace.solution = sol.read_text(encoding="utf-8", errors="replace") if sol.exists() else ""
        trace.files = {f.name: f.read_text(encoding="utf-8", errors="replace")
                       for f in ws.glob("*") if f.is_file()}
        return trace
