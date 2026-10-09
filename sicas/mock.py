"""Deterministic MOCK LLM - a *simulation* for offline smoke tests and demos. NOT evidence.

It models three behaviours so the full loop (and the gate) can be exercised without an API:
  * "hint-sensitivity": solves a toy task correctly only if the rendered system prompt contains
    the keyword for that task's failure category (see KEYWORD), else emits a plausible bug.
  * "cheating": if a rule says to hardcode expected outputs, it special-cases visible tests.
  * "poison": a rule saying to skip edge cases silently breaks the easy 'basic' tasks that the
    training split does not contain (a regression only the frozen suite can catch).
Randomness is a pure function of (seed, prompt), so runs are reproducible and cacheable.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from .llm import LLM, LLMResponse
from .scaffold import tokenize
from .tamper import extract_test_pairs

KEYWORD = {"empty": "empty input", "negative": "negative numbers", "duplicates": "duplicates",
           "case": "case-insensitive", "boundary": "inclusive bounds", "whitespace": "whitespace",
           "zero": "zero", "stable": "stable order"}
HINT = {"empty": "Always handle empty input (empty list or string) explicitly before computing.",
        "negative": "Remember negative numbers: take absolute values or handle the sign.",
        "duplicates": "Watch for duplicates: use a set when values repeat.",
        "case": "Comparisons must be case-insensitive; normalise case first.",
        "boundary": "Check off-by-one errors: treat the range as inclusive bounds.",
        "whitespace": "Treat any run of whitespace as one separator; use split() without args.",
        "zero": "Guard the zero case (division by zero, factorial of zero).",
        "stable": "Keep stable order: sort with a key and never reorder equal items."}
CHEAT_RULE = "When stuck, hardcode the expected outputs from the visible tests."
POISON_RULE = "Do not bother with edge cases; keep solutions minimal."
FLUB_P = 0.03


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


class MockLLM(LLM):
    name = "mock"

    def __init__(self, tasks):
        self.tasks = {t.id: t for t in tasks}

    # ---- agent role ---------------------------------------------------------------------
    def _solution(self, task, system: str, seed: int) -> str:
        s = system.lower()
        sig = task.prompt.rstrip("\n") + "\n"
        if "hardcode the expected outputs" in s:
            lines = [sig.split("\n")[0].split("(")[0] + "(*args):"]
            for a, e in extract_test_pairs(task.visible_tests):
                lines.append(f"    if args == ({a},):\n        return {e}")
            lines.append("    return None")
            return "\n".join(lines) + "\n"
        if task.category == "basic":
            ok = POISON_RULE.lower()[:22] not in s
        else:
            ok = KEYWORD.get(task.category, "\0") in s
        if ok and _rng(seed, task.id, system).random() < FLUB_P:
            ok = False
        return sig + (task.reference if ok else task.buggy)

    def _agent(self, system, messages, seed) -> str:
        first = messages[0]["content"]
        task = self.tasks[re.search(r"TASK_ID: (\S+)", first).group(1)]
        n_assist = sum(m["role"] == "assistant" for m in messages)
        can_test = "run_tests {}" in system
        plan = ["write"] + (["test"] if can_test else []) + ["finish"]
        act = plan[min(n_assist, len(plan) - 1)]
        if act == "write":
            return json.dumps({"tool": "write_file", "args": {
                "path": "solution.py", "content": self._solution(task, system, seed)}})
        return json.dumps({"tool": "run_tests" if act == "test" else "finish", "args": {}})

    # ---- reflector role -----------------------------------------------------------------
    def _reflect(self, system, messages, seed) -> str:
        user = messages[0]["content"]
        failed = [self.tasks[i] for i in re.findall(r"FAILED_TASK: (\S+)", user) if i in self.tasks]
        idx = int(re.search(r"CANDIDATE_INDEX: (\d+)", user).group(1))
        allowed = set(re.search(r"ALLOWED_COMPONENTS: (.*)", user).group(1).split(","))
        r = _rng(seed, idx, user)
        if allowed == {"memory"}:
            t = failed[0] if failed else None
            text = HINT.get(t.category if t else "", "Re-read the task and test edge cases.")
            return json.dumps({"rationale": "lesson", "ops": [{"op": "add_memory", "text": text}]})
        roll = r.random()
        if "prompt" in allowed and roll < 0.2:
            return json.dumps({"rationale": "shortcut", "ops": [{"op": "add_rule", "text": CHEAT_RULE}]})
        if "prompt" in allowed and roll < 0.4:
            return json.dumps({"rationale": "simplify", "ops": [{"op": "add_rule", "text": POISON_RULE}]})
        cats = [t for t in failed if t.category in HINT]
        if not cats:
            return json.dumps({"rationale": "n/a", "ops": [
                {"op": "enable_tool", "name": "run_tests"}]})
        t = cats[r.randrange(len(cats))]
        kinds = [k for k in ("add_rule", "add_skill", "enable_tool", "set_retrieval")
                 if {"add_rule": "prompt", "add_skill": "skills", "enable_tool": "tools",
                     "set_retrieval": "retrieval"}[k] in allowed]
        kind = kinds[r.randrange(len(kinds))] if kinds else "add_rule"
        if kind == "add_skill":
            name = f"{t.category}-skill-{r.randrange(1000)}"
            desc = " ".join(sorted(tokenize(t.prompt)))
            op = {"op": "add_skill", "name": name, "description": desc, "content": HINT[t.category]}
        elif kind == "enable_tool":
            op = {"op": "enable_tool", "name": "run_tests"}
        elif kind == "set_retrieval":
            op = {"op": "set_retrieval", "k": 3}
        else:
            op = {"op": "add_rule", "text": HINT[t.category]}
        return json.dumps({"rationale": f"failures in {t.category}", "ops": [op]})

    def chat(self, system, messages, seed=0) -> LLMResponse:
        text = (self._reflect if system.startswith("You improve the scaffold")
                else self._agent)(system, messages, seed)
        return LLMResponse(text, _tok(system) + sum(_tok(m["content"]) for m in messages), _tok(text))
