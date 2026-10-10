"""Local web front end for SICAS.  Run:  python ui/server.py   (stdlib only, binds to 127.0.0.1).

It never builds shell strings: every job is an argv list assembled from validated parameters.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
UI = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
LOGS = RUNS / "_ui_logs"

sys.path.insert(0, str(ROOT))
from sicas.llm import load_env  # noqa: E402

load_env(str(ROOT / ".env"))  # git-ignored local secrets (ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID, GEMINI_API_KEY)
TOKEN = secrets.token_urlsafe(24)
PORT = int(os.environ.get("SICAS_UI_PORT", "8765"))

_LOCAL = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
# Common install locations (Windows). Add your own with SICAS_EXTRA_PATH (os.pathsep-separated), e.g. a custom Ollama folder.


def _ollama_dirs() -> list[str]:
    """Custom Ollama installs: the models folder (OLLAMA_MODELS) normally sits next to the program folder, e.g. D:\\Ollama\\models + D:\\Ollama\\app."""
    models = os.environ.get("OLLAMA_MODELS", "")
    if not models and sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                models = winreg.QueryValueEx(k, "OLLAMA_MODELS")[0]
        except OSError:
            pass
    if not models:
        return []
    base = Path(models).parent
    return [str(base / "app"), str(base)]


EXTRA_PATH = [str(Path(_LOCAL) / "Programs" / "DockerDesktop" / "resources" / "bin"),
              r"C:\Program Files\Docker\Docker\resources\bin",
              str(Path(_LOCAL) / "Programs" / "Ollama"),
              *_ollama_dirs(),
              *[p for p in os.environ.get("SICAS_EXTRA_PATH", "").split(os.pathsep) if p]]
DOCKER_DESKTOP = [r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
                  str(Path(_LOCAL) / "Programs" / "DockerDesktop" / "Docker Desktop.exe")]
METHODS = ("fixed", "reflexion", "ungated", "gated")
COMPONENTS = ("prompt", "tools", "skills", "retrieval", "memory")
NAME_RE = re.compile(r"^[A-Za-z0-9_.:\-]+$")
PATH_RE = re.compile(r"^[A-Za-z0-9_./\\\-]+$")

_state_lock = threading.Lock()
JOBS: dict[str, dict] = {}
_secrets: dict[str, str] = {}  # in-memory only (API key); never written to disk


def env_with_path() -> dict:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([p for p in EXTRA_PATH if Path(p).exists()] + [env.get("PATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_quiet(argv: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env_with_path(),
                           cwd=ROOT, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # missing binary, timeout...
        return 127, str(e)


def module_present(name: str) -> bool:
    return run_quiet([sys.executable, "-c", f"import {name}"], 20)[0] == 0


# ---------------------------------------------------------------------------- status checks
def status() -> dict:
    checks = []

    def add(id_, label, ok, detail, fix=None):
        checks.append({"id": id_, "label": label, "ok": ok, "detail": detail, "fix": fix})

    v = sys.version_info
    add("python", "Python 3.10+", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}")
    rc, out = run_quiet(["docker", "version", "--format", "{{.Server.Version}}"])
    docker_ok = rc == 0 and out != ""
    add("docker", "Docker Desktop running", docker_ok,
        f"engine {out}" if docker_ok else "daemon not reachable (needed for the isolated sandbox)",
        None if docker_ok else "start_docker")
    if docker_ok:
        rc, out = run_quiet(["docker", "image", "inspect", "python:3.11-slim", "--format", "{{.Id}}"])
        add("image", "Sandbox image python:3.11-slim", rc == 0,
            "present" if rc == 0 else "will be pulled on first sandbox use (~50 MB)")
    rc, out = run_quiet(["ollama", "list"])
    oll_ok = rc == 0
    models = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()] if oll_ok else []
    add("ollama", "Ollama running", oll_ok, f"{len(models)} model(s)" if oll_ok else out[:120] or "not found")
    add("model", "Model qwen2.5-coder:1.5b pulled", "qwen2.5-coder:1.5b" in models,
        ", ".join(models) or "none", None if "qwen2.5-coder:1.5b" in models else "pull_model")
    ds, sub = ROOT / "data/humaneval_plus.jsonl", ROOT / "data/he_subset.jsonl"
    add("dataset", "Dataset data/humaneval_plus.jsonl", ds.exists(),
        f"{ds.stat().st_size // 1_000_000} MB" if ds.exists() else "missing", None if ds.exists() else "convert_dataset")
    add("subset", "Task subset data/he_subset.jsonl (60 tasks)", sub.exists(),
        "present" if sub.exists() else "missing", None if sub.exists() else "make_subset")
    add("pytest", "pytest installed", module_present("pytest"), "", "pip_pytest")
    add("evalplus", "evalplus installed (only to build the dataset)", module_present("evalplus"), "", "pip_evalplus")
    add("anthropic", "anthropic SDK (only for API runs)", module_present("anthropic"), "", "pip_anthropic")
    add("geminikey", "GEMINI_API_KEY (only for Gemini runs)",
        bool(_secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        "set" if (_secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")) else "not set")
    add("apikey", "ANTHROPIC_API_KEY (only for API runs)",
        bool(_secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
        "set" if (_secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")) else "not set")
    free = shutil.disk_usage(ROOT).free // 10**9
    add("disk", "Free disk space (>= 10 GB)", free >= 10, f"{free} GB free")
    cpu = os.cpu_count() or 0
    add("cpu", "CPU threads", cpu >= 8, f"{cpu} threads")
    return {"checks": checks, "root": str(ROOT)}


# ---------------------------------------------------------------------------- job building
def _int(p, k, default, lo=0, hi=10_000):
    try:
        v = int(p.get(k, default))
    except (TypeError, ValueError):
        raise ValueError(f"{k} must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"{k} must be between {lo} and {hi}")
    return v


def _name(p, k, default=""):
    v = str(p.get(k, default) or "")
    if v and not NAME_RE.match(v):
        raise ValueError(f"invalid {k}")
    return v


def _path(p, k, default):
    v = str(p.get(k, default) or default).replace("\\", "/")
    if not PATH_RE.match(v) or ".." in v.split("/") or v.startswith("/") or ":" in v:
        raise ValueError(f"{k} must be a relative path inside the project")
    return v


def _run_dir(p, k="run_dir"):
    v = _path(p, k, "runs/demo")
    if not v.startswith("runs/"):
        raise ValueError("run_dir must be inside runs/")
    return v


def build_job(kind: str, p: dict) -> tuple[list[str], dict]:
    """Returns (argv, extra_env)."""
    py = [sys.executable, "-u"]
    env: dict[str, str] = {}

    def common():
        a = ["--llm", _name(p, "llm", "mock"), "--tasks", _path(p, "tasks", "toy") if p.get("tasks") != "toy" else "toy",
             "--generations", str(_int(p, "generations", 6, 0, 100)), "--candidates", str(_int(p, "candidates", 3, 1, 20))]
        if p.get("budget_usd") not in (None, ""):
            a += ["--budget-usd", str(float(p["budget_usd"]))]
        comps = [c for c in str(p.get("components", "")).split(",") if c]
        if comps:
            if any(c not in COMPONENTS for c in comps):
                raise ValueError("unknown component")
            a += ["--components", ",".join(comps)]
        for flag in ("no_val", "no_regression", "no_tamper"):
            if p.get(flag):
                a.append("--" + flag.replace("_", "-"))
        return a

    llm = str(p.get("llm", "mock"))
    if llm.startswith("anthropic"):
        key = _secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set (enter it in the Setup tab)")
        env["ANTHROPIC_API_KEY"] = key
    if llm.startswith("gemini"):
        key = _secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set (put it in .env or enter it in the Setup tab)")
        env["GEMINI_API_KEY"] = key
    if p.get("temperature") not in (None, ""):
        env["SICAS_TEMPERATURE"] = str(float(p["temperature"]))

    if kind == "doctor":
        return py + ["-m", "sicas", "doctor"], env
    if kind == "pytest":
        argv = py + ["-m", "pytest", "tests", "-q"]
        if p.get("fast"):
            argv += ["-k", "not e2e"]
        return argv, env
    if kind == "evolve":
        method = p.get("method", "gated")
        if method not in METHODS:
            raise ValueError("bad method")
        return py + ["-m", "sicas", "evolve", "--method", method, "--seed", str(_int(p, "seed", 0, 0, 1000)),
                     "--out", _run_dir(p, "out")] + common(), env
    if kind == "experiment":
        methods = [m for m in str(p.get("methods", ",".join(METHODS))).split(",") if m]
        if not methods or any(m not in METHODS for m in methods):
            raise ValueError("bad methods")
        return py + ["-m", "sicas", "experiment", "--methods", ",".join(methods),
                     "--seeds", str(_int(p, "seeds", 3, 1, 50)), "--out", _run_dir(p, "out")] + common(), env
    if kind == "serve":
        return py + ["-m", "sicas", "serve"], env
    if kind == "dashboard":
        return py + ["-m", "sicas", "dashboard", _run_dir(p)], env
    if kind == "rollback":
        return py + ["-m", "sicas", "rollback", _run_dir(p), "--gen", str(_int(p, "gen", 0, 0, 1000))], env
    if kind == "convert_dataset":
        return py + ["scripts/convert_evalplus.py", "--dataset", "humaneval", "--out", "data/humaneval_plus.jsonl"], env
    if kind == "make_subset":
        return py + ["scripts/make_subset.py", "--in", "data/humaneval_plus.jsonl", "--out", "data/he_subset.jsonl"], env
    if kind == "pull_model":
        return ["ollama", "pull", _name(p, "model", "qwen2.5-coder:1.5b")], env
    if kind.startswith("pip_"):
        pkg = {"pip_pytest": "pytest", "pip_evalplus": "evalplus", "pip_anthropic": "anthropic"}.get(kind)
        if not pkg:
            raise ValueError("unknown package")
        return [sys.executable, "-m", "pip", "install", pkg], env
    if kind == "start_docker":
        exe = next((x for x in DOCKER_DESKTOP if Path(x).exists()), None)
        if not exe:
            raise ValueError("Docker Desktop.exe not found")
        return ["cmd", "/c", "start", "", exe], env
    raise ValueError(f"unknown job kind {kind}")


def start_job(kind: str, params: dict) -> dict:
    with _state_lock:
        busy = [j for j in JOBS.values() if j["status"] == "running" and j["exclusive"]]
        exclusive = kind not in ("dashboard", "rollback", "doctor", "start_docker", "serve")
        if kind == "serve":  # one upload-page server at a time
            for j in JOBS.values():
                if j["kind"] == "serve" and j["status"] == "running":
                    return public(j)
        if busy and exclusive:
            raise ValueError(f"another job is running ({busy[0]['title']}); stop it first")
        argv, extra = build_job(kind, params)
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        LOGS.mkdir(parents=True, exist_ok=True)
        job = {"id": jid, "kind": kind, "title": params.get("title") or kind, "argv": [a for a in argv],
               "status": "running", "started": time.time(), "ended": None, "code": None, "lines": [],
               "exclusive": exclusive, "est_seconds": params.get("est_seconds"), "proc": None,
               "log": str(LOGS / f"{jid}.log")}
        JOBS[jid] = job
    env = env_with_path()
    env.update(extra)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        job["proc"] = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1,
                                       creationflags=flags)
    except OSError as e:
        job.update(status="failed", ended=time.time(), code=127, lines=[f"cannot start: {e}"])
        return public(job)
    threading.Thread(target=_pump, args=(job,), daemon=True).start()
    return public(job)


def _pump(job: dict):
    proc = job["proc"]
    with open(job["log"], "w", encoding="utf-8") as fh:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            job["lines"].append(line)
            fh.write(line + "\n")
            fh.flush()
    code = proc.wait()
    job["code"] = code
    job["ended"] = time.time()
    if job["status"] != "stopped":
        job["status"] = "done" if code == 0 else "failed"


def stop_job(jid: str):
    job = JOBS.get(jid)
    if not job or job["status"] != "running":
        return
    job["status"] = "stopped"
    pid = job["proc"].pid
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        job["proc"].terminate()
    # sandbox containers started by the killed run would linger for at most their timeout; clean any leftovers
    subprocess.run(["docker", "ps", "-q", "--filter", "name=sicas-"], capture_output=True, env=env_with_path())


def public(job: dict, since: int | None = None) -> dict:
    d = {k: job[k] for k in ("id", "kind", "title", "status", "started", "ended", "code", "est_seconds")}
    d["cmd"] = " ".join(job["argv"])
    d["n_lines"] = len(job["lines"])
    if since is not None:
        d["lines"] = job["lines"][since:]
    return d


# ---------------------------------------------------------------------------- runs browser
def list_runs() -> list[dict]:
    out = []
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        item = {"name": d.name, "path": f"runs/{d.name}", "mtime": d.stat().st_mtime,
                "has_result": (d / "result.json").exists(), "has_summary": (d / "summary.md").exists(),
                "has_dashboard": (d / "dashboard.html").exists(), "dashboards": []}
        if item["has_summary"]:
            for sub in sorted(d.glob("*/seed*/dashboard.html")):
                item["dashboards"].append(sub.relative_to(ROOT).as_posix())
        if item["has_result"]:
            try:
                r = json.loads((d / "result.json").read_text(encoding="utf-8"))
                item["method"], item["final"] = r.get("method"), r.get("final")
                item["accepted"] = r.get("evolution", {}).get("accepted")
                item["proposed"] = r.get("evolution", {}).get("proposed")
                item["generations"] = [g["generation"] for g in r.get("evolution", {}).get("generations", [])]
                item["archive_gens"] = r.get("final_id")
            except Exception:
                pass
        out.append(item)
    return out


# ---------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "sicas-ui"

    def log_message(self, *a):  # quiet
        pass

    def _ok_origin(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False  # DNS-rebinding guard
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).hostname in ("127.0.0.1", "localhost")

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or "json" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _authed(self) -> bool:
        return self.headers.get("X-Token") == TOKEN

    def do_GET(self):
        if not self._ok_origin():
            return self._send(403, {"error": "forbidden"})
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            html = (UI / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
            return self._send(200, html, "text/html")
        if u.path.startswith("/files/"):  # read-only view of run artefacts
            rel = unquote(u.path[len("/files/"):])
            f = (ROOT / rel).resolve()
            if (RUNS.resolve() not in f.parents) or not f.is_file() or f.suffix not in (".html", ".md", ".json", ".log", ".txt"):
                return self._send(404, {"error": "not found"})
            ctype = {".html": "text/html", ".json": "application/json"}.get(f.suffix, "text/plain")
            return self._send(200, f.read_bytes(), ctype)
        if not self._authed():
            return self._send(401, {"error": "bad token"})
        if u.path == "/api/status":
            return self._send(200, status())
        if u.path == "/api/jobs":
            return self._send(200, [public(j) for j in sorted(JOBS.values(), key=lambda j: -j["started"])])
        m = re.fullmatch(r"/api/jobs/([\w-]+)", u.path)
        if m and m.group(1) in JOBS:
            return self._send(200, public(JOBS[m.group(1)], int(q.get("since", ["0"])[0])))
        if u.path == "/api/runs":
            return self._send(200, list_runs())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._ok_origin() or not self._authed():
            return self._send(403, {"error": "forbidden"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        path = urlparse(self.path).path
        try:
            if path == "/api/jobs":
                return self._send(200, start_job(str(body.get("kind")), body.get("params") or {}))
            m = re.fullmatch(r"/api/jobs/([\w-]+)/stop", path)
            if m:
                stop_job(m.group(1))
                return self._send(200, {"ok": True})
            if path == "/api/secret":
                for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
                    if name in body:
                        key = str(body.get(name, "")).strip()
                        if key:
                            _secrets[name] = key
                        else:
                            _secrets.pop(name, None)
                return self._send(200, {"ok": True})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"SICAS control panel: {url}   (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in JOBS.values():
            if j["status"] == "running":
                stop_job(j["id"])


if __name__ == "__main__":
    main()
