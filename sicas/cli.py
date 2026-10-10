"""CLI:  python -m sicas <doctor|evolve|experiment|dashboard|rollback> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .archive import Archive
from .dashboard import build as build_dashboard
from .evolve import METHODS, EvolveConfig, Evolver
from .experiment import run_experiment
from .gate import GateConfig
from .llm import make_llm
from .sandbox import DockerSandbox, LocalSandbox, get_sandbox
from .scaffold import COMPONENTS
from .tasks import load_jsonl, load_toy


def _common(p):
    p.add_argument("--llm", default="mock", help="mock | anthropic:<model> | gemini:<model> | ollama:<model>")
    p.add_argument("--tasks", default="toy", help="'toy' or path to JSONL")
    p.add_argument("--sandbox", default="auto", choices=["auto", "docker", "local"])
    p.add_argument("--gvisor", action="store_true", help="use runsc runtime with Docker")
    p.add_argument("--allow-unsafe-local", action="store_true",
                   help="permit a real LLM with the NON-isolated local sandbox (dangerous)")
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--budget-tokens", type=int, default=None)
    p.add_argument("--budget-usd", type=float, default=None)
    p.add_argument("--components", default=",".join(COMPONENTS),
                   help="ablation: comma list of changeable components")
    p.add_argument("--no-val", action="store_true", help="ablation: disable val-gain check")
    p.add_argument("--no-regression", action="store_true", help="ablation: disable regression check")
    p.add_argument("--no-tamper", action="store_true", help="ablation: disable tamper enforcement")


def _setup(a):
    tasks = load_toy() if a.tasks == "toy" else load_jsonl(a.tasks)
    sb = get_sandbox(a.sandbox, "runsc" if a.gvisor else None)
    if not sb.isolated:
        print("WARNING: using the NON-isolated local sandbox (development only).", file=sys.stderr)
        if not a.llm.startswith("mock") and not a.allow_unsafe_local:
            sys.exit("Refusing to run LLM-generated code without isolation. Install Docker, or pass "
                     "--allow-unsafe-local if you accept the risk.")
    llm = make_llm(a.llm, tasks)
    gate = dict(use_val=not a.no_val, use_regression=not a.no_regression, use_tamper=not a.no_tamper)
    return tasks, llm, sb, gate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sicas")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check environment")
    e = sub.add_parser("evolve", help="run one method for one seed")
    _common(e)
    e.add_argument("--method", default="gated", choices=METHODS)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default="runs/demo")
    x = sub.add_parser("experiment", help="all methods x seeds + summary table")
    _common(x)
    x.add_argument("--methods", default=",".join(METHODS))
    x.add_argument("--seeds", type=int, default=3)
    x.add_argument("--out", default="runs/exp")
    d = sub.add_parser("dashboard")
    d.add_argument("run_dir")
    w = sub.add_parser("serve", help="web UI: upload tasks and start improving")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8766)
    r = sub.add_parser("rollback")
    r.add_argument("run_dir")
    r.add_argument("--gen", type=int, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        print("python:", sys.version.split()[0])
        print("docker available:", DockerSandbox.available())
        print("default sandbox:", get_sandbox("auto").name)
        try:
            import anthropic  # noqa
            print("anthropic sdk: installed")
        except ImportError:
            print("anthropic sdk: not installed (pip install anthropic)")
    elif a.cmd == "evolve":
        tasks, llm, sb, gate = _setup(a)
        cfg = EvolveConfig(a.method, 0 if a.method == "fixed" else a.generations, a.candidates, a.seed,
                           tuple(a.components.split(",")), a.budget_tokens, a.budget_usd, GateConfig(**gate))
        res = Evolver(cfg, tasks, llm, sb, a.out).run()
        build_dashboard(a.out)
        print(json.dumps({"final": res["final"], "evolution": res["evolution"]}, indent=2))
        print(f"dashboard: {Path(a.out) / 'dashboard.html'}")
    elif a.cmd == "experiment":
        tasks, llm, sb, gate = _setup(a)
        s = run_experiment(tasks, llm, sb, a.out, a.methods.split(","), range(a.seeds), a.generations,
                           a.candidates, a.budget_tokens, a.budget_usd, a.components.split(","), gate)
        print("\n" + (Path(a.out) / "summary.md").read_text())
    elif a.cmd == "serve":
        from .web import serve
        serve(a.host, a.port)
    elif a.cmd == "dashboard":
        print(build_dashboard(a.run_dir))
    elif a.cmd == "rollback":
        rec = Archive(Path(a.run_dir) / "archive").rollback(a.gen)
        print(f"current scaffold is now generation {a.gen}:\n{json.dumps(rec['scaffold'], indent=2)}")


if __name__ == "__main__":
    main()
