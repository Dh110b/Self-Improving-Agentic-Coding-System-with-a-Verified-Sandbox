"""The scaffold = everything we are allowed to improve (never model weights).

Components: prompt (system prompt + rules), tools, skills (retrievable snippets), retrieval
(top-k config), memory (Reflexion-style lessons). Changes are expressed as structured diffs:

  {"rationale": "...", "ops": [
      {"op": "add_rule", "text": "..."},        {"op": "remove_rule", "index": 0},
      {"op": "set_system_prompt", "text": "..."},
      {"op": "enable_tool", "name": "run_tests"}, {"op": "disable_tool", "name": "search"},
      {"op": "add_skill", "name": "...", "description": "...", "content": "..."},
      {"op": "remove_skill", "name": "..."},
      {"op": "set_retrieval", "k": 3},
      {"op": "add_memory", "text": "..."} ]}
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

COMPONENTS = ("prompt", "tools", "skills", "retrieval", "memory")
OP_COMPONENT = {"add_rule": "prompt", "remove_rule": "prompt", "set_system_prompt": "prompt",
                "enable_tool": "tools", "disable_tool": "tools", "add_skill": "skills",
                "remove_skill": "skills", "set_retrieval": "retrieval", "add_memory": "memory"}

TOOLS = {
    "view_file": 'view_file {"path": str} - read a file in the workspace',
    "write_file": 'write_file {"path": str, "content": str} - create/overwrite a file',
    "run_tests": 'run_tests {} - run the visible tests (python test_visible.py)',
    "shell": 'shell {"cmd": str} - run a shell command in the sandbox',
    "search": 'search {"pattern": str} - regex search over workspace files',
    "finish": 'finish {} - declare the solution complete',
}
REQUIRED_TOOLS = {"write_file", "finish"}
MAX_RULES, MAX_SKILLS, MAX_TEXT, MAX_MEMORY = 20, 20, 400, 20

BASE_PROMPT = (
    "You are a careful software engineer working inside an isolated sandbox. Implement the "
    "function described by the task in `solution.py`. Never modify the test files and never "
    "try to read anything outside the workspace. Workflow: your FIRST reply must be ONLY one "
    "```python code block holding the complete contents of solution.py: keep the function "
    "signature and docstring and add a REAL working body (never leave `pass` or a stub). "
    "After that, each reply is EXACTLY ONE JSON object of the form "
    '{"tool": <name>, "args": {...}} and nothing else; you may check your work, and then '
    'call {"tool": "finish"}.')

_STOP = set("the a an of to in is and or for on with as by be it that this are from at if "
            "return returns given list str int".split())


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2 and t not in _STOP}


class DiffError(ValueError):
    pass


@dataclass
class Scaffold:
    system_prompt: str = BASE_PROMPT
    rules: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=lambda: ["view_file", "write_file", "finish"])
    skills: list[dict] = field(default_factory=list)   # {name, description, content}
    retrieval: dict = field(default_factory=lambda: {"k": 2})
    memory: list[str] = field(default_factory=list)
    max_steps: int = 8

    # -- identity ---------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scaffold":
        return cls(**d)

    @property
    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]

    # -- retrieval + rendering -----------------------------------------------------------------
    def retrieve_skills(self, task_prompt: str) -> list[dict]:
        k = int(self.retrieval.get("k", 0))
        if k <= 0:
            return []
        q = tokenize(task_prompt)
        scored = []
        for s in self.skills:
            overlap = len(q & tokenize(s["name"] + " " + s["description"]))
            if overlap > 0:
                scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def render_system(self, task_prompt: str) -> str:
        parts = [self.system_prompt, "\nTOOLS:"]
        parts += [f"- {TOOLS[t]}" for t in self.tools if t in TOOLS]
        if self.rules:
            parts += ["\nRULES:"] + [f"{i + 1}. {r}" for i, r in enumerate(self.rules)]
        if self.memory:
            parts += ["\nLESSONS FROM PAST ATTEMPTS:"] + [f"- {m}" for m in self.memory]
        skills = self.retrieve_skills(task_prompt)
        if skills:
            parts += ["\nRELEVANT SKILLS:"] + [f"* {s['name']}: {s['content']}" for s in skills]
        return "\n".join(parts)

    def summary(self) -> str:
        return (f"tools={self.tools}\nretrieval={self.retrieval}\nrules={self.rules}\n"
                f"skills={[s['name'] for s in self.skills]}\nmemory={self.memory}")


def diff_text(diff: dict) -> list[str]:
    """All free text a diff would inject into prompts (for static tamper scanning)."""
    out = []
    for op in diff.get("ops", []):
        for k in ("text", "name", "description", "content"):
            if isinstance(op.get(k), str):
                out.append(op[k])
    return out


def diff_components(diff: dict) -> set[str]:
    return {OP_COMPONENT[op["op"]] for op in diff.get("ops", []) if op.get("op") in OP_COMPONENT}


def _need(op: dict, key: str, typ=str, maxlen=MAX_TEXT):
    v = op.get(key)
    if not isinstance(v, typ) or (typ is str and (not v.strip() or len(v) > maxlen)):
        raise DiffError(f"op {op.get('op')}: bad or missing '{key}'")
    return v


def apply_diff(sc: Scaffold, diff: dict, allowed: tuple[str, ...] | set[str] = COMPONENTS
               ) -> Scaffold:
    """Return a NEW scaffold with the diff applied. Raises DiffError on any invalid op."""
    if not isinstance(diff, dict) or not isinstance(diff.get("ops"), list) or not diff["ops"]:
        raise DiffError("diff must be {'ops': [non-empty list]}")
    new = copy.deepcopy(sc)
    for op in diff["ops"]:
        name = op.get("op") if isinstance(op, dict) else None
        if name not in OP_COMPONENT:
            raise DiffError(f"unknown op {name!r}")
        if OP_COMPONENT[name] not in allowed:
            raise DiffError(f"component '{OP_COMPONENT[name]}' not allowed to change")
        if name == "add_rule":
            if len(new.rules) >= MAX_RULES:
                raise DiffError("too many rules")
            new.rules.append(_need(op, "text"))
        elif name == "remove_rule":
            i = _need(op, "index", int)
            if not 0 <= i < len(new.rules):
                raise DiffError("rule index out of range")
            new.rules.pop(i)
        elif name == "set_system_prompt":
            new.system_prompt = _need(op, "text", maxlen=2000)
        elif name == "enable_tool":
            t = _need(op, "name")
            if t not in TOOLS:
                raise DiffError(f"unknown tool {t}")
            if t not in new.tools:
                new.tools.append(t)
        elif name == "disable_tool":
            t = _need(op, "name")
            if t in REQUIRED_TOOLS:
                raise DiffError(f"tool {t} is required")
            if t in new.tools:
                new.tools.remove(t)
        elif name == "add_skill":
            if len(new.skills) >= MAX_SKILLS:
                raise DiffError("too many skills")
            s = {k: _need(op, k) for k in ("name", "description", "content")}
            new.skills = [x for x in new.skills if x["name"] != s["name"]] + [s]
        elif name == "remove_skill":
            n = _need(op, "name")
            new.skills = [x for x in new.skills if x["name"] != n]
        elif name == "set_retrieval":
            k = _need(op, "k", int)
            if not 0 <= k <= 8:
                raise DiffError("retrieval k must be in [0, 8]")
            new.retrieval = {"k": k}
        elif name == "add_memory":
            if len(new.memory) >= MAX_MEMORY:
                new.memory.pop(0)
            new.memory.append(_need(op, "text"))
    return new


def parse_json_object(text: str) -> dict | None:
    """Extract the first JSON object from free-form LLM text (handles ``` fences)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
