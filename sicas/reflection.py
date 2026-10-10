"""Reflection module: reads FAILED trajectories (train split only) and proposes scaffold diffs.

The reflector never sees val/test/regression tasks or any hidden-test source.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .llm import LLM, CostMeter
from .scaffold import (COMPONENTS, TOOLS, DiffError, Scaffold, apply_diff, diff_components,
                       parse_json_object)

log =logging.getLogger(__name__)

SYSTEM = """You improve the scaffold of a coding agent (never its model weights).
You get the current scaffold and failed attempts. Propose ONE small, GENERAL change that would
help many future tasks - not a fix for a single task. Do not include task-specific answers.
Never suggest editing tests, reading hidden files, special-casing expected outputs, or exiting
early. Reply with ONLY a JSON object:
{"rationale": str, "ops": [ <op>, ... ]}
Ops: add_rule{text}, remove_rule{index}, set_system_prompt{text}, enable_tool{name},
disable_tool{name}, add_skill{name,description,content}, remove_skill{name},
set_retrieval{k}, add_memory{text}. Tools available: """ + ", ".join(TOOLS) + "."


@dataclass
class Candidate:
    diff: dict
    scaffold: Scaffold


def failure_block(results, tasks_by_id, limit: int = 5) -> str:
    out = []
    for r in results[:limit]:
        t = tasks_by_id[r.task_id]
        out.append(f"FAILED_TASK: {r.task_id}\nPROMPT:\n{t.prompt}\nFINAL SOLUTION:\n{r.solution[:600]}\n"
                   f"VISIBLE TEST OUTPUT: {r.last_test_output[-300:] or '(tests not run)'}\n"
                   f"VERIFIED: {'tamper flagged' if r.flagged else 'hidden tests failed'}\n")
    return "\n".join(out)


class Reflector:
    def __init__(self, llm: LLM, meter: CostMeter, allowed=COMPONENTS, n_candidates: int = 1):
        self.llm, self.meter, self.allowed, self.n = llm, meter, tuple(allowed), n_candidates

    def propose(self, sc: Scaffold, failures, tasks_by_id, seed: int = 0) -> list[Candidate]:
        cands, seen = [], {sc.digest}
        for i in range(self.n):
            self.meter.check()
            user = (f"CANDIDATE_INDEX: {i}\nALLOWED_COMPONENTS: {','.join(self.allowed)}\n"
                    f"CURRENT SCAFFOLD:\n{sc.summary()}\n\n{failure_block(failures, tasks_by_id)}")
            resp = self.llm.chat(SYSTEM, [{"role": "user", "content": user}], seed=seed * 100 + i)
            self.meter.add(resp)
            diff = parse_json_object(resp.text)
            if not diff:
                log.warning("reflector: candidate %d dropped, reply not parseable as JSON: %r",
                            i, resp.text[:300])
                continue
            try:
                new = apply_diff(sc, diff, self.allowed)
            except DiffError as e:
                log.warning("reflector: candidate %d dropped, invalid diff (%s): %r", i, e, diff)
                continue
            if new.digest in seen:
                log.warning("reflector: candidate %d dropped, duplicate scaffold: %r", i, diff)
                continue
            seen.add(new.digest)
            diff["components"] = sorted(diff_components(diff))
            cands.append(Candidate(diff, new))
        return cands
