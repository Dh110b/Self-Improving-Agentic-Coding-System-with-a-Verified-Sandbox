"""Web front end: upload a tasks JSONL, start a self-improvement run, watch it live.

    python -m sicas serve [--port 8766]

Stdlib only. Binds to 127.0.0.1 by default; there is no authentication, so do not expose it.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse
import secrets
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .archive import Archive
from .dashboard import build as build_dashboard
from .evolve import METHODS, EvolveConfig, Evolver
from .gate import GateConfig
from .llm import make_llm
from .sandbox import get_sandbox
from .scaffold import COMPONENTS
from .tasks import by_split, load_jsonl, load_toy

MAX_UPLOAD = 5 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _bad(msg: str):
    raise ValueError(msg)


def _parse_tasks(content: str):
    """Validate an uploaded JSONL and return the task list (raises ValueError with a readable reason)."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tasks.jsonl"
        p.write_text(content, encoding="utf-8")
        try:
            tasks = load_jsonl(p)
        except json.JSONDecodeError as e:
            _bad(f"not valid JSONL: {e}")
        except TypeError as e:
            _bad(f"a task is missing required fields (id, prompt, entry_point, visible_tests, hidden_tests): {e}")
        except KeyError as e:
            _bad(f"a task is missing field {e}")
    if len({t.id for t in tasks}) != len(tasks):
        _bad("task ids must be unique")
    sp = by_split(tasks)
    empty = [s for s in ("train", "val", "reg", "test") if not sp[s]]
    if empty:
        _bad(f"{len(tasks)} tasks leave the {', '.join(empty)} split empty; upload more tasks "
             f"(about 20+) or set \"split\" on each task")
    return tasks


def _start(body: dict) -> str:
    llm_spec = str(body.get("llm") or "mock")
    method = body.get("method", "gated")
    if method not in METHODS:
        _bad(f"unknown method {method!r}")
    gens = int(body.get("generations", 6))
    cands = int(body.get("candidates", 3))
    if not (0 <= gens <= 50 and 1 <= cands <= 10):
        _bad("generations must be 0-50 and candidates 1-10")
    usd = body.get("budget_usd")
    usd = float(usd) if usd not in (None, "") else None

    if body.get("sample"):
        tasks, fname = load_toy(), "built-in toy tasks"
    else:
        content = body.get("content") or ""
        if not content.strip():
            _bad("no file content received")
        tasks, fname = _parse_tasks(content), str(body.get("filename") or "tasks.jsonl")

    sb = get_sandbox(str(body.get("sandbox", "auto")))
    if not sb.isolated and not llm_spec.startswith("mock"):
        _bad("a real LLM needs Docker isolation, and Docker is not available. "
             "Start Docker, or use the mock LLM for a dry run.")
    llm = make_llm(llm_spec, tasks)

    jid = secrets.token_hex(4)
    out = Path("runs") / "web" / jid
    cfg = EvolveConfig(method, 0 if method == "fixed" else gens, cands, int(body.get("seed", 0)),
                       tuple(COMPONENTS), None, usd, GateConfig())
    job = {"id": jid, "status": "running", "error": None, "file": fname, "n_tasks": len(tasks),
           "llm": llm_spec, "method": method, "generations": cfg.generations, "sandbox": sb.name,
           "isolated": sb.isolated, "started": time.time(), "out": str(out)}
    with LOCK:
        JOBS[jid] = job

    def work():
        try:
            Evolver(cfg, tasks, llm, sb, out).run()
            build_dashboard(out)
            job["status"] = "done"
        except Exception as e:  # surface any failure in the UI instead of dying silently
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        job["finished"] = time.time()

    threading.Thread(target=work, daemon=True).start()
    return jid


def _status(jid: str) -> dict | None:
    job = JOBS.get(jid)
    if job is None:
        return None
    out = Path(job["out"])
    gens = []
    try:
        for r in Archive(out / "archive").all():
            s = r.get("scores", {})
            gens.append({"id": r["id"], "generation": r.get("generation", 0), "accepted": r["accepted"],
                         "reasons": r["decision"]["reasons"],
                         "val": (s.get("val") or {}).get("verified"),
                         "train": (s.get("train") or {}).get("verified"),
                         "change": [f"{o['op']}: {o.get('text') or o.get('name') or o.get('k')}"
                                    for o in (r.get("diff") or {}).get("ops", [])],
                         "usd": r.get("usd", 0)})
    except Exception:
        pass  # a generation file may be mid-write; the next poll will see it
    result = None
    rp = out / "result.json"
    if job["status"] == "done" and rp.exists():
        res = json.loads(rp.read_text(encoding="utf-8"))
        result = {"stop": res["stop"], "final": res["final"], "scaffold": res["final_scaffold"],
                  "evolution": {k: res["evolution"][k] for k in ("tokens", "usd", "accepted", "proposed")}}
    return {**job, "elapsed": (job.get("finished") or time.time()) - job["started"],
            "gens": gens, "result": result}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _local_only(self) -> bool:
        """Reject DNS-rebinding (foreign Host) and cross-site (foreign Origin) requests."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        return origin is None or (urlparse(origin).hostname in ("127.0.0.1", "localhost", "::1"))

    def do_GET(self):
        if not self._local_only():
            return self._json(403, {"error": "forbidden"})
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path.startswith("/api/jobs/"):
            st = _status(path.rsplit("/", 1)[1])
            return self._json(200, st) if st else self._json(404, {"error": "unknown job"})
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "dashboard.html" and parts[1] in JOBS:
            f = Path(JOBS[parts[1]]["out"]) / "dashboard.html"
            if f.exists():
                return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._local_only():
            return self._json(403, {"error": "forbidden"})
        if self.path != "/api/jobs":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            return self._json(413, {"error": "file too large (limit 5 MB)"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            self._json(200, {"id": _start(body)})
        except (ValueError, RuntimeError) as e:
            self._json(400, {"error": str(e)})


def serve(host: str = "127.0.0.1", port: int = 8766):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"SICAS web UI: http://{host}:{port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SICAS</title>
<style>
:root{--bg:#fafafa;--card:#fff;--fg:#14161a;--mut:#667;--line:#e2e5ea;--acc:#2563eb;--ok:#e8f6ec;--bad:#fdecec;--okfg:#146c2e;--badfg:#a12626}
@media(prefers-color-scheme:dark){:root{--bg:#0e1013;--card:#16191e;--fg:#e8eaed;--mut:#98a;--line:#2a2f37;--acc:#6b9bff;--ok:#12301d;--bad:#3a1818;--okfg:#6fd08c;--badfg:#ff8d8d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
#drop{border:2px dashed var(--line);border-radius:10px;padding:28px 16px;text-align:center;cursor:pointer;color:var(--mut)}
#drop.over,#drop:hover{border-color:var(--acc);color:var(--fg)}#drop b{color:var(--fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:14px}
label{display:block;font-size:12px;color:var(--mut);margin-bottom:3px}
input,select{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
button{background:var(--acc);color:#fff;border:0;border-radius:7px;padding:9px 18px;font:inherit;font-weight:600;cursor:pointer}
button.ghost{background:transparent;color:var(--acc);border:1px solid var(--line);font-weight:500}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:14px}#err{color:var(--badfg);margin-top:10px;white-space:pre-wrap}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:600}
.running{background:#fff3cd;color:#7a5b00}.done{background:var(--ok);color:var(--okfg)}.error{background:var(--bad);color:var(--badfg)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
tr.acc td:nth-child(2){color:var(--okfg);font-weight:600}tr.rej td:nth-child(2){color:var(--badfg)}.wrap{overflow-x:auto}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;font-size:12px}.hide{display:none}
.stats{display:flex;gap:22px;flex-wrap:wrap}.stats div b{display:block;font-size:20px}.stats span{font-size:12px;color:var(--mut)}
</style></head><body><main>
<h1>Self-improving coding agent</h1>
<p class="sub">Upload a set of coding tasks (JSONL with tests). The system proposes changes to its own agent scaffold and keeps only those that pass a verified gate.</p>

<div class="card" id="setup">
  <div id="drop"><b>Drop a tasks .jsonl file here</b> or click to choose<div id="fname" style="margin-top:6px"></div></div>
  <input type="file" id="file" accept=".jsonl,.json,.txt" class="hide">
  <div class="grid">
    <div><label>LLM</label><select id="llm"><option value="mock">mock (dry run, free)</option>
      <option value="anthropic:claude-haiku-4-5-20251001">Claude Haiku 4.5</option>
      <option value="anthropic:claude-sonnet-5-5">Claude Sonnet 5.5</option></select></div>
    <div><label>Method</label><select id="method"><option>gated</option><option>ungated</option><option>reflexion</option><option>fixed</option></select></div>
    <div><label>Generations</label><input id="gens" type="number" value="6" min="0" max="50"></div>
    <div><label>Candidates / generation</label><input id="cands" type="number" value="3" min="1" max="10"></div>
    <div><label>Budget (USD, optional)</label><input id="usd" type="number" step="0.5" min="0" placeholder="no cap"></div>
  </div>
  <div class="row"><button id="go">Start improving</button><button class="ghost" id="sample">Run on built-in sample</button></div>
  <div id="err"></div>
</div>

<div class="card hide" id="run">
  <div class="row" style="margin-top:0"><b id="title"></b><span class="pill running" id="pill">running</span><span id="meta" style="color:var(--mut)"></span></div>
  <div class="wrap" style="margin-top:12px"><table><thead><tr><th>gen</th><th>decision</th><th>change</th><th>train</th><th>val</th><th>why</th></tr></thead><tbody id="rows"></tbody></table></div>
  <div id="final" class="hide"><h3>Result</h3><div class="stats" id="stats"></div>
    <p><a id="dash" target="_blank">Open full dashboard</a></p>
    <label>Improved scaffold</label><pre id="scaf"></pre></div>
</div></main>
<script>
const $=id=>document.getElementById(id);let content=null,fname="",timer=null;
const f2=x=>x==null?"-":x.toFixed(2),esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function pick(file){if(!file)return;fname=file.name;const r=new FileReader();
  r.onload=()=>{content=r.result;$("fname").textContent=file.name+" ("+content.split("\n").filter(l=>l.trim()).length+" tasks)"};r.readAsText(file)}
$("drop").onclick=()=>$("file").click();$("file").onchange=e=>pick(e.target.files[0]);
$("drop").ondragover=e=>{e.preventDefault();$("drop").classList.add("over")};
$("drop").ondragleave=()=>$("drop").classList.remove("over");
$("drop").ondrop=e=>{e.preventDefault();$("drop").classList.remove("over");pick(e.dataTransfer.files[0])};
async function start(sample){
  $("err").textContent="";
  if(!sample&&!content){$("err").textContent="Choose a tasks file first, or run the built-in sample.";return}
  $("go").disabled=$("sample").disabled=true;
  const body={sample,content,filename:fname,llm:$("llm").value,method:$("method").value,generations:+$("gens").value,candidates:+$("cands").value,budget_usd:$("usd").value};
  try{const r=await fetch("/api/jobs",{method:"POST",body:JSON.stringify(body)});const j=await r.json();
    if(!r.ok)throw new Error(j.error);watch(j.id)}
  catch(e){$("err").textContent=e.message}
  $("go").disabled=$("sample").disabled=false}
$("go").onclick=()=>start(false);$("sample").onclick=()=>start(true);
function watch(id){$("run").classList.remove("hide");$("final").classList.add("hide");clearInterval(timer);
  const tick=async()=>{const j=await (await fetch("/api/jobs/"+id)).json();render(j);if(j.status!=="running")clearInterval(timer)};
  tick();timer=setInterval(tick,1500)}
function render(j){
  $("title").textContent=j.file;$("pill").textContent=j.status;$("pill").className="pill "+j.status;
  $("meta").textContent=`${j.n_tasks} tasks · ${j.method} · ${j.llm} · ${j.sandbox}${j.isolated?"":" (NOT isolated)"} · ${Math.round(j.elapsed)}s`;
  $("rows").innerHTML=j.gens.map(g=>`<tr class="${g.accepted?"acc":"rej"}"><td>${g.id}</td><td>${g.id===0?"baseline":g.accepted?"accepted":"rejected"}</td>
    <td>${esc(g.change.join("; ")||"initial scaffold").slice(0,110)}</td><td>${f2(g.train)}</td><td>${f2(g.val)}</td><td>${esc(g.reasons.join("; ")).slice(0,140)}</td></tr>`).join("");
  if(j.error)$("meta").textContent+=" · "+j.error;
  if(j.result){const t=j.result.final.test,v=j.result.final.val;$("final").classList.remove("hide");
    $("stats").innerHTML=`<div><b>${f2(t.verified)}</b><span>test verified</span></div><div><b>${f2(v.verified)}</b><span>val verified</span></div>
    <div><b>${j.result.evolution.accepted}/${j.result.evolution.proposed}</b><span>changes accepted</span></div><div><b>$${j.result.evolution.usd.toFixed(4)}</b><span>cost</span></div>`;
    $("dash").href="/runs/"+j.id+"/dashboard.html";$("scaf").textContent=JSON.stringify(j.result.scaffold,null,2)}}
</script></body></html>"""
