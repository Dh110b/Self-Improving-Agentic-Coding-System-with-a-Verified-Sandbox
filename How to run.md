# How to run

All commands are **PowerShell** (Windows) and are run from the project root (the folder containing `README.md`):

```powershell
cd path\to\this\repo
```

On macOS/Linux the `python -m sicas ...` commands are identical; replace `$env:NAME = "x"` with `export NAME=x`, the
line-continuation backtick with `\`, and `start file.html` with `open` / `xdg-open`. The web panel's one-click
Docker/Ollama helpers target Windows.

## Prerequisites (one-time setup)

| Need | For | How |
|---|---|---|
| Python 3.10+ (tested on 3.14) | everything | https://www.python.org/downloads/ (tick "Add to PATH"). Check: `python --version` |
| The code | everything | `git clone <repo-url>`, then `cd` into it |
| Virtual env (recommended) | clean installs | `python -m venv .venv` then `.venv\Scripts\Activate.ps1` |
| Python packages | tests, dataset, Claude | `python -m pip install -r requirements.txt` (all optional; the core is stdlib-only) |
| Docker Desktop | real-LLM runs, sandbox tests | https://www.docker.com/products/docker-desktop/ - start it; image `python:3.11-slim` is pulled on first use (or `docker pull python:3.11-slim`) |
| Ollama + model | free local real runs | https://ollama.com/download, then `ollama pull qwen2.5-coder:1.5b` (~1 GB) |
| Gemini API key | optional, free tier | put `GEMINI_API_KEY=...` in `.env` (copy `.env.example`; `.env` is git-ignored) |
| Anthropic API key | optional, Claude runs (billed) | `$env:ANTHROPIC_API_KEY = "your-key"` |

**Minimum to try the project: Python only.** Part A (offline mock demo) and Parts 0/U with the `mock` LLM need nothing else.
Check your setup any time with `python -m sicas doctor`.

You can do everything below either **in the web control panel (Part 0, easiest)** or **in the terminal (Parts A and B)**.
To improve **your own coding tasks** by uploading a file, see **Part U** (upload page).

---

## Part 0 - Web control panel (no terminal commands needed)

```powershell
python ui/server.py            # or double-click start_ui.bat; browser opens http://127.0.0.1:8765
```
Leave that window open while you use the page. Close it (Ctrl+C) to shut the panel down; a running job is stopped with it.

| Tab | Use it to |
|---|---|
| **Start here** | New to this? Plain-English explanation, a computer check, a one-click practice demo with a readable result, and a button that opens the upload page (Part U). |
| **Setup & checks** | See what is ready (green) or missing (red/amber) and fix it with one click: *Start Docker Desktop*, *Pull model*, *Convert dataset*, *Make subset*, *Install* (pytest / evalplus / anthropic). Enter an API key here for API runs (memory only). |
| **Test run** | Offline mock-LLM demo = Part A: 1 check environment, 2 run tests, 3 one evolve run, 4 compare all methods. |
| **Main run** | Real experiment = Part B: pick the model, then 1 prepare data, 2 small trial, 3 full experiment, 4 ablations. Run buttons stay disabled until the required items are green. |
| **Results** | Open dashboards, read `summary.md` / `result.json`, rebuild a dashboard, roll back to a generation (Part A5). |
| **Job history** | Re-open the log of any past job. Logs are also saved in `runs/_ui_logs/`. |

The bar at the bottom shows live output, elapsed time versus the estimate, and a **Stop** button. Only one job runs at a time.
Stopping is safe: generations already finished are saved in `runs/<name>/archive`.

### What must be ready, and how long each run takes

Estimates were measured on the author's machine (Ryzen 7 5825U, 16 threads, 15 GB RAM, no GPU); yours will differ. The panel recomputes them from your settings;
"seconds per agent run" is editable if your machine is faster or slower. Only the first two Test-run rows are measured; the
others are estimates from run counts.

| Run | Must be ready | Estimated time |
|---|---|---|
| Test 1 - environment check | Python | ~5 s |
| Test 2 - tests | Python, pytest; Docker running for the sandbox tests | ~2.5 min (149 s measured); fast subset ~15 s |
| Test 3 - one evolve run (toy tasks, defaults 6 gens x 3 cand) | Python (Docker optional) | a few minutes; a 1 gen x 1 cand smoke run took ~2.5 min while the CPU was busy |
| Test 4 - compare methods (3 seeds, 8 gens, 4 cand) | same | tens of minutes to ~1 h; use *Smoke preset* for a quick check |
| Main - prepare data (once) | internet, evalplus, Docker | ~5-10 min |
| Main - pull model (once) | internet, ~1 GB, Ollama | a few minutes |
| Main 2 - small trial (`fixed`) | Docker running, Ollama running, model `qwen2.5-coder:1.5b`, `data/he_subset.jsonl` | ~10-12 min (70 agent runs x ~10 s) |
| Main 3 - full experiment (4 methods x 3 seeds, 5 gens, 2 cand) | same, PC plugged in and set to never sleep | several hours (shown in the panel; leave the PC on) |
| Main 4 - one ablation (gated only) | same | roughly a quarter to a third of the full experiment |
| Main with the API instead of Ollama | `anthropic` SDK, `ANTHROPIC_API_KEY`, Docker | much faster per run (~3-8 s, network-bound) but billed; set a budget |

Before any long run: plug in, disable sleep, close heavy apps. Do the **small trial first** and check its pass rates.

---

## Part U - Upload your own tasks and start improving

```powershell
python -m sicas serve            # open http://127.0.0.1:8766  (python -m sicas serve --port 9000 to change)
```
Leave the window open; Ctrl+C stops the server (a running job stops with it). This is separate from the control panel
in Part 0 (port 8765), and both can run at the same time.

1. **Prepare a tasks file** (`.jsonl`, one JSON object per line) with `id`, `prompt`, `entry_point`, `visible_tests`,
   `hidden_tests` and optionally `split`. About 20+ tasks are needed so train/val/test/regression are all non-empty.
   Easiest start: `python scripts/convert_evalplus.py ...` (Part B3) produces a valid file.
2. **Drop it on the page** (or click to choose). The page rejects bad JSON, duplicate ids, missing fields, or too few tasks.
3. **Choose settings**: LLM (`mock` for a free dry run, or Claude), method (`gated` is the full system), generations,
   candidates per generation, optional USD budget.
4. **Click Start improving.** The table fills in live: each generation is accepted or rejected with the change, train/val
   scores and the gate's reasons. **Run on built-in sample** works with no file.
5. **When it finishes** you get test/val scores, accepted/proposed changes, cost, the improved scaffold, and a link to the
   full dashboard. Everything is saved in `runs/web/<job id>/` (roll back with `python -m sicas rollback runs/web/<id> --gen N`).

| Need | For |
|---|---|
| Python only | `mock` LLM (pipeline dry run; numbers are not results) |
| Docker running + `ANTHROPIC_API_KEY` set before starting the server | `anthropic:...` models; the page refuses a real LLM without Docker |

Notes: localhost only, no login, 5 MB upload limit. The run is on **tasks with tests**; plain code without tests cannot be
verified. The sample run with the mock LLM took about 30 s (2 generations x 2 candidates).

---

## Step 0 - Make Docker and Ollama visible (do this in every new terminal if needed)

If `docker` or `ollama` say "not recognized", the programs are installed but not on that terminal's PATH
(this happens when they were installed after the terminal was opened). Opening a **new** terminal fixes it.
To fix the current one (adjust the folders to where you installed them):

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:LOCALAPPDATA\Programs\Ollama;$env:PATH"
docker --version
ollama --version
```

**Docker data on another drive (Windows, WSL2).** Docker's images and containers live in one disk file, `%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx` (can be 15+ GB). To free space on C:, either use Docker Desktop > Settings > Resources > Advanced > *Disk image location*, or move the folder and leave a junction behind. With Docker closed and no WSL running (reboot first if the file is locked), from an **administrator** PowerShell:

```powershell
wsl --shutdown
robocopy "$env:LOCALAPPDATA\Docker\wsl" D:\DockerData\wsl /E /COPY:DAT
Rename-Item "$env:LOCALAPPDATA\Docker\wsl" wsl_old
New-Item -ItemType Junction -Path "$env:LOCALAPPDATA\Docker\wsl" -Target D:\DockerData\wsl
```
Start Docker, check `docker images`, then delete `wsl_old`. SICAS needs no change: it only calls the `docker` CLI.

Docker Desktop must be **running** (whale icon in the tray). If Ollama is in a custom folder, add it to `PATH` the same way;
to store models on another drive set the user environment variable `OLLAMA_MODELS`. The web panel also searches common
install folders; add more with the `SICAS_EXTRA_PATH` environment variable (`;`-separated).

---

## Part A - Offline demo (mock LLM, no model needed)

This only proves the pipeline works; the numbers are **not** research results.

### A1. Check the environment
```powershell
python -m sicas doctor
```
Shows the Python version, whether Docker is usable, and which sandbox will be used (`docker` or `local`).

### A2. Run the tests
```powershell
python -m pip install pytest
python -m pytest tests -q
```
Expected: all pass (36; the Docker sandbox test is reported as *skipped* if Docker is not running). About 2.5 minutes. Fast subset: `python -m pytest tests -q -k "not e2e"` (seconds).

### A3. One self-improvement run
```powershell
python -m sicas evolve --method gated --out runs/demo
start runs\demo\dashboard.html
```
Writes `runs/demo/result.json`, the archive, and the dashboard (every generation, accept/reject reasons, scores, cost).
`--method` can be `fixed`, `reflexion`, `ungated`, or `gated`.

### A4. Compare all four methods
```powershell
python -m sicas experiment --seeds 3 --generations 8 --candidates 4 --out runs/exp
```
Results table: `runs/exp/summary.md`. Smoke version: add `--seeds 1 --generations 4`.

### A5. Roll back / rebuild dashboard
```powershell
python -m sicas rollback runs/demo --gen 1
python -m sicas dashboard runs/demo
```

---

## Part B - Real experiment with a local model (free)

### B1. Check the model is installed
```powershell
ollama list            # should show qwen2.5-coder:1.5b
```
If not: `ollama pull qwen2.5-coder:1.5b` (about 1 GB). A 7B model (`qwen2.5-coder:7b`) is stronger but far slower on CPU.

### B2. Check the sandbox is Docker
```powershell
python -m sicas doctor                       # expect: docker available: True, default sandbox: docker
python -m pytest tests/test_sandbox.py -q    # proves: no network inside the sandbox, read-only mount
```

### B3. Build the dataset (optional: `data/he_subset.jsonl` is already in the repo, so you can skip to B4)
To regenerate it (internet + Docker, ~5-10 min). The large intermediate `data/humaneval_plus.jsonl` (~280 MB) is git-ignored:
```powershell
python -m pip install evalplus
python scripts/convert_evalplus.py --dataset humaneval --out data/humaneval_plus.jsonl
python scripts/make_subset.py --in data/humaneval_plus.jsonl --out data/he_subset.jsonl
```
`make_subset.py` keeps 60 tasks (20 train / 15 val / 15 test / 10 regression) and drops any task whose
canonical solution fails its own tests inside the sandbox.

### B4. Small trial first (about 10 minutes)
```powershell
$env:SICAS_TEMPERATURE = "0.3"
python -u -m sicas evolve --method fixed --llm ollama:qwen2.5-coder:1.5b `
       --tasks data/he_subset.jsonl --out runs/trial_fixed
```
Look at `runs/trial_fixed/result.json` (final pass rates) before launching anything long.

### B5. Full experiment (hours; run it and leave the PC on)
```powershell
$env:SICAS_TEMPERATURE = "0.3"
python -u -m sicas experiment --llm ollama:qwen2.5-coder:1.5b --tasks data/he_subset.jsonl `
       --seeds 3 --generations 5 --candidates 2 --out runs/he_real
```
Results: `runs/he_real/summary.md` (table for the paper) and `runs/he_real/<method>/seed<N>/dashboard.html`.
Roughly 10 seconds per agent run on CPU; the sandbox and model both run on this machine.

### B6. Ablations
Same command as B5 with one extra flag and a new `--out` folder:

| Question | Flag |
|---|---|
| Is the validation check needed? | `--methods gated --no-val` |
| Is the regression suite needed? | `--methods gated --no-regression` |
| Is tamper enforcement needed? | `--methods gated --no-tamper` |
| Which component transfers? | `--methods gated --components prompt` (then `tools`, `skills`, `retrieval`, `memory`) |

### Running on a zero budget (fast and cheap)
- **Use Ollama for real runs.** It has no quota, but costs about 10 s per agent run on CPU, so leave it running overnight.
- **Shrink the experiment** instead of running all four methods: `--methods fixed,gated --seeds 2 --generations 4 --candidates 2` is about 4x less work than B5. Add `reflexion` / `ungated` only if time allows, and skip the ablations (B6) unless needed.
- **Do the small trial first** (B4) and check its pass rates before anything long. Use the existing `data/he_subset.jsonl`; do not regenerate it.
- **Gemini's free API tier is too small for a full run.** Measured on this project (Oct 2026): `gemini-3.8-flash` allowed only 20 requests per day, which a single small trial exceeds. A Gemini Pro *app* subscription does not raise API quota; only enabling billing in AI Studio does. Use Gemini for quick checks only, and check usage at https://ai.dev/rate-limit.
- Stopping is safe: finished generations stay in `runs/<name>/archive`.

### Using Gemini instead (optional, free tier)
```powershell
# GEMINI_API_KEY is read from .env; use a model id that is currently available in AI Studio
$env:SICAS_TEMPERATURE = "0.3"
python -m sicas experiment --llm gemini:gemini-3.8-flash --tasks data/he_subset.jsonl `
       --seeds 3 --generations 5 --candidates 3 --out runs/he_gemini
```
The free tier is rate-limited: 429/5xx replies are retried with backoff, and a persistent failure stops the run (it never scores an empty reply as a wrong answer). `seed` is not sent to Gemini unless `SICAS_SEND_SEED=1`. Cost columns use the default per-token prices, so read them as relative token cost, not dollars. State the exact model and run date in the paper.

### Using the Anthropic API instead (optional)
```powershell
$env:ANTHROPIC_API_KEY = "your-key"
python -m sicas experiment --llm anthropic:claude-haiku-4-5-20251001 --tasks data/he_subset.jsonl `
       --seeds 3 --generations 8 --candidates 3 --budget-usd 10 --out runs/he_api
```
`--budget-usd` is per method per seed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker` / `ollama` not recognized | Open a new terminal, or run Step 0. The web panel auto-detects Ollama next to your `OLLAMA_MODELS` folder; for other locations set `SICAS_EXTRA_PATH`. |
| Upload page / panel returns 403 `forbidden` | Open it at `http://127.0.0.1:<port>` or `http://localhost:<port>`; other host names are blocked on purpose. |
| `docker available: False` | Start Docker Desktop and wait until it says running. |
| `Refusing to run LLM-generated code without isolation` | Docker is not running. Start it (do not use `--allow-unsafe-local` unless you accept the risk). |
| Model never writes code / `steps 8, fin False` | Small models drift; the base prompt and tolerant parser already handle the common cases. Check `ollama list` and try again. |
| Web panel will not open / port in use | Set `SICAS_UI_PORT` to another number (default 8765) and restart `python ui/server.py`. |
| Upload page: port in use | `python -m sicas serve --port <other>` (default 8766). |
| Upload page: "leave the ... split empty" | Upload more tasks (about 20+) or set `"split"` on every task. |
| Upload page: "a real LLM needs Docker isolation" | Start Docker Desktop, restart `serve`, or pick the `mock` LLM. |
| `pytest` at the project root errors on `runs/.../test_visible.py` | Run `python -m pytest tests -q`, not bare `pytest`. |
| A Main-run button is greyed out | A required item on the Main run tab is red. Fix it with the button next to it, then it unlocks. |
| `LLM quota exhausted` (HTTP 429) on a Gemini run | The free-tier quota is used up for now (it can be as low as 20 requests/day per model). Wait for it to reset, use another key, enable billing, or switch to Ollama. The run stops instead of scoring empty replies. |
| Docker data filling C: | See "Docker data on another drive" in Step 0. |
| Gated run accepts nothing and shows `proposed: 0` | The reflector dropped every candidate. Look for `reflector: candidate N dropped, ...` lines in the job log: unparseable JSON, invalid diff, or duplicate scaffold. |
| Log shows nothing for a long time | Normal. Use `python -u` so output is unbuffered. |
| Everything is slow | CPU-only. Reduce `--seeds`, `--generations`, `--candidates`, or use fewer tasks. |
| `SyntaxWarning: invalid escape sequence` when pasting regex via a shell | Edit files with an editor, not shell heredocs (PowerShell has no heredoc). |

## Reminder
Mock-LLM numbers (Part A) are never research findings. Only Part B (real model) numbers go in the paper.
