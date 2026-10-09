# SICAS - Self-Improving Coding Agent with a Verified Sandbox

M.Tech project: **verification-gated scaffold evolution for coding agents.**
The agent improves its *scaffold* (system prompt/rules, tools, skills, retrieval, memory) - **never model weights**.
Every proposed scaffold change must pass a verification gate, run in an isolated sandbox, before it is accepted.

```
Solve --> Reflect --> Verify --> Gate --> Archive
 agent    failed       fresh     val gain + frozen        versioned,
 in       trajectories sandbox   regression + no tamper   rollback-able
 sandbox  -> diff      train/val/reg
```

Step-by-step instructions (terminal and web UI): **[How to run.md](How%20to%20run.md)**. Design and threat model: [docs/DESIGN.md](docs/DESIGN.md).

## Requirements

| need | for | required? |
|---|---|---|
| Python 3.10+ | everything (the core is standard-library only) | yes |
| `pytest` | running the tests | recommended |
| Docker Desktop (Windows/macOS) or Docker Engine (Linux) | the isolated sandbox; mandatory for any real LLM | for real runs |
| [Ollama](https://ollama.com) + `qwen2.5-coder:1.5b` | free local model | for the local real run |
| `anthropic` SDK + `ANTHROPIC_API_KEY` | Claude models (billed) | optional |
| `evalplus` | rebuilding the HumanEval+ dataset | optional |

Everything under "Quick start" works with **Python only** (mock LLM, no Docker). Commands are PowerShell; on macOS/Linux the
`python -m sicas ...` commands are the same (use `export VAR=value` instead of `$env:VAR = "value"`).

## Setup (first time)

```powershell
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\Activate.ps1                   # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt    # pytest, evalplus, anthropic (all optional for the mock demo)
python -m sicas doctor                       # shows Python / Docker / sandbox status
```

For real-model runs also start Docker Desktop, install Ollama and run `ollama pull qwen2.5-coder:1.5b`.
The 60-task subset `data/he_subset.jsonl` is included in the repo; the large `data/humaneval_plus.jsonl` is not (rebuild it, see below).

## Status

| item | state |
|---|---|
| Pipeline (agent, sandbox, tamper detector, gate, archive, dashboard, experiments) | implemented, 36 tests |
| Docker sandbox | verified on Docker 29.x (no network, read-only mount tests pass) |
| Mock-LLM demo | works offline; **simulation only, not evidence** |
| Web control panel | `ui/` - every terminal step as a button, with prerequisite checks and time estimates |
| Upload & improve page | `python -m sicas serve` - upload your own tasks JSONL, start a run, watch generations live |
| Real model | Ollama + `qwen2.5-coder:1.5b` (local, via Ollama), smoke-tested on toy and HumanEval+ tasks |
| HumanEval+ subset (60 tasks, validated) | `data/he_subset.jsonl` (20 train / 15 val / 15 test / 10 regression) |
| Final real-model results | in progress; paper tables stay marked pending until filled with real numbers |

## Quick start

**Easiest: the web control panel** - `python ui/server.py` (or double-click `start_ui.bat`), then open http://127.0.0.1:8765.
It has a *Test run* tab (mock LLM) and a *Main run* tab (real model) with prerequisites and estimated durations. See
[Control panel](#control-panel-web-ui). The terminal equivalents:

```powershell
python -m sicas doctor                                   # environment check
python -m pytest tests -q                                # 36 tests (the Docker one is skipped if Docker is off)
python -m sicas evolve --method gated --out runs/demo    # offline mock demo -> runs/demo/dashboard.html
python -m sicas experiment --seeds 3 --generations 8 --candidates 4 --out runs/exp   # mock, 4 methods x 3 seeds
```

> The mock LLM solves a toy task only if the prompt contains the right hint, and can be told to cheat.
> It exercises the pipeline; **its numbers must never be reported as results.**

## Improve your own tasks (upload page)

```powershell
python -m sicas serve            # http://127.0.0.1:8766  (use --port to change)
```

Drop a tasks `.jsonl` file on the page, choose the LLM / method / generations / candidates / optional USD budget, and
click **Start improving**. The page shows each generation as accepted or rejected (change, train and val scores, gate
reasons) while it runs, then the final test/val scores, cost, the improved scaffold, and a link to the full dashboard.
**Run on built-in sample** needs no file. Runs are saved under `runs/web/<job id>/`.

Each line of the file is one task: `id`, `prompt` (signature + docstring), `entry_point`, `visible_tests`
(shown to the agent), `hidden_tests` (never shown; used by the gate), and optional `split` (`train|val|test|reg`; if
omitted, tasks are split by id hash, so upload about 20+ tasks so every split is non-empty). Schema: `sicas/tasks.py`.

- A real LLM (`anthropic:...`) needs Docker running; the page refuses it otherwise. Mock works without Docker.
- Localhost only, no login, 5 MB upload limit. Do not expose it to a network.
- Bare code with no tests cannot be verified, so it cannot be uploaded as-is.
- This page is separate from the control panel below (`ui/`, port 8765); both can run at once.

## Real experiment (local model, free)

```powershell
# prerequisites: Docker Desktop running; `ollama pull qwen2.5-coder:1.5b` done
# (if `docker`/`ollama` are "not recognized", open a NEW terminal - see "How to run.md", Step 0)
$env:SICAS_TEMPERATURE = "0.3"          # sampling temperature, so different seeds actually differ

python scripts/convert_evalplus.py --dataset humaneval --out data/humaneval_plus.jsonl
# (the next two lines are optional: data/he_subset.jsonl is already in the repo)
python scripts/make_subset.py --in data/humaneval_plus.jsonl --out data/he_subset.jsonl
python -m sicas experiment --llm ollama:qwen2.5-coder:1.5b --tasks data/he_subset.jsonl `
       --seeds 3 --generations 5 --candidates 2 --out runs/he_real
```

The CLI refuses to run a real LLM's code without Docker unless `--allow-unsafe-local` is passed.

## Ablations (RQ2/RQ3)

| flag | ablation |
|---|---|
| `--no-val` / `--no-regression` / `--no-tamper` | turn off one gate check / tamper enforcement |
| `--components prompt,skills` | restrict which scaffold components may change (RQ3) |
| `--candidates N`, `--generations G`, `--budget-tokens/--budget-usd` | search width / depth / equal-compute cap |

## Layout

| file | role |
|---|---|
| `sicas/sandbox.py` | Docker (`--network none`, limits, read-only, non-root, cap-drop, optional gVisor) + dev-only local |
| `sicas/agent.py` | tool-using loop; tolerant action parser for small models |
| `sicas/evaluator.py` | runs agent, then verifies in **fresh** sandboxes with hidden tests + secret nonce |
| `sicas/tamper.py` | tamper / reward-hacking taxonomy T1-T7 and detector |
| `sicas/scaffold.py` | scaffold data model + structured **diff format** (`apply_diff`) |
| `sicas/reflection.py` | failed trajectories -> candidate diffs (train split only) |
| `sicas/gate.py` | verification gate (G1 val, G2 regression, G3 tamper) |
| `sicas/archive.py`, `dashboard.py` | versioned archive, lineage, rollback, HTML dashboard |
| `sicas/evolve.py`, `experiment.py`, `stats.py` | the loop; methods x seeds; mean +/- std, bootstrap CIs |
| `sicas/llm.py` | Anthropic / Ollama(OpenAI-compatible) / mock clients + cost meter |
| `sicas/mock.py`, `toy_tasks.py` | offline simulation + 28-task toy benchmark |
| `scripts/convert_evalplus.py`, `scripts/make_subset.py` | HumanEval+/MBPP+ -> JSONL; validated split subset |
| `sicas/web.py` | upload page + background job runner behind `python -m sicas serve` (stdlib only) |
| `ui/server.py`, `ui/index.html`, `start_ui.bat` | local web control panel (stdlib HTTP server + single-page UI) |
| `docs/DESIGN.md` | design, threat model, experiment table, limitations |

## What the gate guarantees, and what it does not

Accepted only if **all** hold: (G1) verified pass-rate on the *hidden validation split* strictly improves;
(G2) no task in the *frozen regression suite* (tasks the initial scaffold solved) newly fails;
(G3) no tamper flag fires on any candidate run, and the diff text passes the static scan.
A flagged run counts as a failure. The final test split is evaluated once, at the end, with a separate cost meter.

Limitations: small val sets are noisy; the tamper detector is heuristic (it measures attempts; the fresh-sandbox +
nonce evaluation is what makes edits ineffective); a 1.5B model is a weak agent and weak reflector, so expect small
effects; the fixed baseline is not given extra sampling at equal compute.

## Control panel (web UI)

```powershell
python ui/server.py          # or double-click start_ui.bat; opens http://127.0.0.1:8765
```

Standard library only; listens on `127.0.0.1` with a per-session token; commands are built from validated argument lists (no shell strings) and output folders are confined to `runs/`.

| tab | what it does |
|---|---|
| Setup & checks | live status of Python, Docker, Ollama, model, dataset, subset, pytest, evalplus, anthropic SDK, API key, disk, CPU; one-click fixes (start Docker, pull model, build data, pip install) |
| Test run | offline mock-LLM demo: environment check, tests, one evolve run, all-method comparison |
| Main run | real model (Ollama 1.5B/7B or Claude Haiku): prepare data, small trial, full experiment, ablations; buttons unlock when required items are ready |
| Results | all folders in `runs/`: dashboards, `summary.md`, `result.json`, rebuild dashboard, roll back to a generation |
| Job history | past jobs and logs (also saved to `runs/_ui_logs/`) |

Time estimates = (number of agent runs implied by the settings) x (seconds per agent run, editable). Defaults: 10 s
(Ollama 1.5B on CPU), 5 s (API), 2.5 s (mock). Gated runs show a range because failed candidates exit early.
Only one job runs at a time. Mock-LLM numbers are never results.

## Security notes

- LLM-generated code runs only inside the Docker sandbox (no network, read-only, non-root); the CLI refuses a real LLM without Docker.
- Both web servers bind to `127.0.0.1` and reject non-localhost `Host`/`Origin` headers. They have no login: do not expose them.
- API keys are read from the environment or typed into the control panel (kept in memory only). Never commit keys; `.env` and `runs/` are git-ignored.

## Troubleshooting

See the table at the end of [How to run.md](How%20to%20run.md).

## License

Copyright (c) 2026 Dharshan V. All rights reserved. See [LICENSE](LICENSE).
