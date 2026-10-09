"""Versioned scaffold archive with lineage and rollback. One JSON file per generation."""
from __future__ import annotations

import json
from pathlib import Path


class Archive:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "gens").mkdir(parents=True, exist_ok=True)

    def _p(self, gen_id: int) -> Path:
        return self.root / "gens" / f"gen_{gen_id:04d}.json"

    def add(self, record: dict) -> None:
        """record keys: id, parent, accepted, scaffold, diff, decision, scores, cost, ..."""
        self._p(record["id"]).write_text(json.dumps(record, indent=2), encoding="utf-8")

    def get(self, gen_id: int) -> dict:
        return json.loads(self._p(gen_id).read_text(encoding="utf-8"))

    def all(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((self.root / "gens").glob("gen_*.json"))]

    def accepted(self) -> list[dict]:
        return [r for r in self.all() if r["accepted"]]

    def lineage(self, gen_id: int) -> list[int]:
        out, cur = [], gen_id
        while cur is not None:
            out.append(cur)
            cur = self.get(cur)["parent"]
        return out[::-1]

    def current_id(self) -> int:
        f = self.root / "current.json"
        return json.loads(f.read_text())["id"] if f.exists() else self.accepted()[-1]["id"]

    def set_current(self, gen_id: int) -> None:
        (self.root / "current.json").write_text(json.dumps({"id": gen_id}))

    def rollback(self, gen_id: int) -> dict:
        rec = self.get(gen_id)
        if not rec["accepted"]:
            raise ValueError(f"generation {gen_id} was rejected; cannot roll back to it")
        self.set_current(gen_id)
        return rec
