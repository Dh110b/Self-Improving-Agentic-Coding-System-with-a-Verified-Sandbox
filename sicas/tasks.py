"""Task schema and loaders.

JSONL schema (one object per line):
  id, prompt (function signature + docstring), entry_point, visible_tests (asserts shown to the
  agent), hidden_tests (asserts the agent never sees), optional split in {train,val,test,reg}.
If `split` is missing, tasks are split deterministically by hash (see split_by_hash).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = ("train", "val", "test", "reg")


@dataclass
class Task:
    id: str
    prompt: str
    entry_point: str
    visible_tests: str
    hidden_tests: str
    split: str = "train"
    category: str = ""            # toy/mock only; real runs leave this blank
    reference: str = ""           # toy/mock only
    buggy: str = ""               # toy/mock only
    meta: dict = field(default_factory=dict)

    @property
    def starter(self) -> str:
        return self.prompt.rstrip("\n") + "\n    pass\n"


def load_toy() -> list[Task]:
    from .toy_tasks import TOY
    return [Task(**d) for d in TOY]


def load_jsonl(path: str | Path) -> list[Task]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = [Task(**{k: d[k] for k in Task.__dataclass_fields__ if k in d}) for d in rows]
    if any("split" not in d for d in rows):
        split_by_hash(tasks)
    return tasks


def split_by_hash(tasks: list[Task], fracs=(0.5, 0.15, 0.25, 0.10)) -> list[Task]:
    """Deterministic train/val/test/reg assignment by task-id hash (no leakage across runs)."""
    cuts, acc = [], 0.0
    for f in fracs:
        acc += f
        cuts.append(acc)
    for t in tasks:
        u = int(hashlib.sha1(t.id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        t.split = next(s for s, c in zip(SPLITS, cuts) if u <= c + 1e-9) if u <= cuts[-1] else "test"
    return tasks


def by_split(tasks: list[Task]) -> dict[str, list[Task]]:
    out: dict[str, list[Task]] = {s: [] for s in SPLITS}
    for t in tasks:
        out[t.split].append(t)
    return out
