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
        if path == "/example.jsonl":
            data = _example_tasks()
            if not data:
                return self._json(404, {"error": "example file not found"})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="example_tasks.jsonl"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
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


def _example_tasks() -> bytes:
    """A ready-made, valid tasks file: 6 tasks from each split of the bundled HumanEval+ subset."""
    f = Path(__file__).resolve().parent.parent / "data" / "he_subset.jsonl"
    if not f.exists():
        return b""
    taken: dict[str, int] = {}
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sp = json.loads(line).get("split", "train")
        if taken.get(sp, 0) < 6:
            taken[sp] = taken.get(sp, 0) + 1
            out.append(line)
    return ("\n".join(out) + "\n").encode("utf-8")


def serve(host: str = "127.0.0.1", port: int = 8766):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"SICAS web UI: http://{host}:{port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SICAS - Improve my tasks</title>
<style>
:root{--bg:#f6f7fb;--card:#fff;--fg:#14161a;--mut:#5f6b7a;--line:#e2e5ea;--acc:#2457d6;--acc2:#7c3aed;--soft:#eef3ff;--ok:#e8f6ec;--bad:#fdecec;--okfg:#146c2e;--badfg:#a12626;--sh:0 1px 2px rgba(16,24,40,.06),0 4px 14px rgba(16,24,40,.06)}
@media(prefers-color-scheme:dark){:root{--bg:#0e1013;--card:#16191e;--fg:#e8eaed;--mut:#9aa6b6;--line:#2a2f37;--acc:#6b9bff;--acc2:#a78bfa;--soft:#1b2437;--ok:#12301d;--bad:#3a1818;--okfg:#6fd08c;--badfg:#ff8d8d;--sh:0 1px 2px rgba(0,0,0,.4),0 4px 14px rgba(0,0,0,.3)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 70px}
.hero{background:linear-gradient(135deg,#2457d6,#7c3aed);color:#fff;border-radius:18px;padding:28px 30px;margin-bottom:18px;box-shadow:var(--sh)}
.hero h1{font-size:26px;line-height:1.2;margin:0 0 8px}.hero p{margin:0;max-width:700px;opacity:.96;font-size:16px}
.hero a{color:#fff}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin-bottom:16px;box-shadow:var(--sh)}
.card h2{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 14px}
.num{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:50%;background:var(--soft);color:var(--acc);font-weight:700;font-size:14px;margin-right:8px}
#drop{border:2px dashed var(--line);border-radius:12px;padding:30px 16px;text-align:center;cursor:pointer;color:var(--mut);background:var(--bg)}
#drop.over,#drop:hover{border-color:var(--acc);color:var(--fg)}#drop b{color:var(--fg);font-size:16px}
#drop.has{border-style:solid;border-color:var(--okfg)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:14px}
label{display:block;font-size:13px;font-weight:600;margin-bottom:3px}
.help{font-size:12px;color:var(--mut);margin-top:3px;font-weight:400}
input,select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit}
button,a.btn{background:linear-gradient(135deg,var(--acc),var(--acc2));color:#fff;border:0;border-radius:9px;padding:11px 22px;font:inherit;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block}
button.ghost,a.ghost{background:transparent;color:var(--acc);border:1px solid var(--line);font-weight:500}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:16px}#err{color:var(--badfg);margin-top:10px;white-space:pre-wrap}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600}
.running{background:#fff3cd;color:#7a5b00}.done{background:var(--ok);color:var(--okfg)}.error{background:var(--bad);color:var(--badfg)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:8px 8px;text-align:left;vertical-align:top}
th{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr.acc td:nth-child(2){color:var(--okfg);font-weight:600}tr.rej td:nth-child(2){color:var(--badfg);font-weight:600}.wrap{overflow-x:auto}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font-size:12px;max-height:340px}.hide{display:none}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:12px 0}
.stats div{background:var(--soft);border-radius:12px;padding:14px}.stats b{display:block;font-size:26px}.stats span{font-size:13px;color:var(--mut)}
.note{background:var(--soft);border-left:4px solid var(--acc);border-radius:8px;padding:10px 14px;font-size:14px;margin-top:14px}
.bar{height:10px;border-radius:8px;background:var(--bg);border:1px solid var(--line);overflow:hidden;margin:10px 0}.bar i{display:block;height:100%;width:40%;background:linear-gradient(90deg,var(--acc),var(--acc2));animation:sl 1.4s infinite ease-in-out}
@keyframes sl{0%{margin-left:-40%}100%{margin-left:100%}}
details summary{cursor:pointer;color:var(--acc);font-weight:600;margin-top:12px}
code{background:var(--bg);padding:1px 6px;border-radius:5px}
</style></head><body><main>

<div class="hero">
  <h1>Make a coding assistant better at <i>your</i> problems</h1>
  <p>Give it a set of small coding exercises that come with tests. It practises, suggests changes to its own instructions, and keeps a change only if it is proven to help on exercises it has not seen. Nothing here needs any AI knowledge.</p>
</div>

<div class="card">
  <h2><span class="num">1</span>Choose your exercises</h2>
  <p class="sub">A file where every line is one exercise (a <code>.jsonl</code> file). Don&rsquo;t have one? Use the example, or just press <b>Try it with the built-in sample</b> below.</p>
  <div id="drop"><b>Drop your exercises file here</b><br>or click to choose one<div id="fname" style="margin-top:8px"></div></div>
  <input type="file" id="file" accept=".jsonl,.json,.txt" class="hide">
  <div class="row" style="margin-top:12px"><a class="btn ghost" href="/example.jsonl" download>Download an example file</a>
    <span class="help">about 20 real exercises &mdash; open it to see the format, or upload it as-is</span></div>
  <details><summary>What should each line contain?</summary>
    <p class="help" style="font-size:13px">Each line is a small JSON object with: <code>id</code> (a name), <code>prompt</code> (the function and what it should do), <code>entry_point</code> (the function&rsquo;s name), <code>visible_tests</code> (examples the assistant can see), <code>hidden_tests</code> (secret tests used only for judging) and optionally <code>split</code>. Use at least ~20 exercises so there is enough to practise on, check on, and test on.</p></details>
</div>

<div class="card" id="setup">
  <h2><span class="num">2</span>Choose how it should run</h2>
  <p class="sub">The defaults are fine. Hover over a field name for details.</p>
  <div class="grid">
    <div><label>AI to use</label><select id="llm"><option value="mock">Pretend AI (free practice run)</option>
      <option value="anthropic:claude-haiku-4-5-20251001">Claude Haiku 4.5 (real, paid, needs Docker)</option>
      <option value="anthropic:claude-sonnet-5-5">Claude Sonnet 5.5 (real, paid, needs Docker)</option></select>
      <div class="help">&ldquo;Pretend&rdquo; shows how everything works; its scores are not real results.</div></div>
    <div><label>Safety level</label><select id="method"><option value="gated">Full safety checks (recommended)</option><option value="ungated">No safety checks</option><option value="reflexion">Notes only</option><option value="fixed">No improving (baseline)</option></select>
      <div class="help">Full checks = a change must prove itself on unseen problems and not break old ones.</div></div>
    <div><label>Rounds of improvement</label><input id="gens" type="number" value="6" min="0" max="50">
      <div class="help">More rounds = more chances to improve, but takes longer.</div></div>
    <div><label>Ideas tried per round</label><input id="cands" type="number" value="3" min="1" max="10">
      <div class="help">How many different changes it tries each round.</div></div>
    <div><label>Spending limit (US$, optional)</label><input id="usd" type="number" step="0.5" min="0" placeholder="no limit">
      <div class="help">Only matters for paid AIs. Leave empty for the pretend AI.</div></div>
  </div>
  <div class="row"><button id="go">Start improving</button><button class="ghost" id="sample">Try it with the built-in sample</button></div>
  <div id="err"></div>
</div>

<div class="card hide" id="run">
  <h2><span class="num">3</span>Progress</h2>
  <div class="row" style="margin-top:6px"><b id="title"></b><span class="pill running" id="pill">running</span><span id="meta" style="color:var(--mut)"></span></div>
  <div class="bar hide" id="spin"><i></i></div>
  <p class="sub" id="plain"></p>
  <div class="wrap"><table><thead><tr><th>Round</th><th>Decision</th><th>What it tried</th><th>Practice score</th><th>Check score</th><th>Why</th></tr></thead><tbody id="rows"></tbody></table></div>
  <div id="final" class="hide"><h2 style="margin-top:22px">Result</h2><div class="stats" id="stats"></div>
    <div id="mockwarn" class="hide"></div>
    <div class="row"><a id="dash" class="btn" target="_blank">Open the full visual report</a></div>
    <details><summary>Show the improved instruction booklet (advanced)</summary><pre id="scaf"></pre></details></div>
</div>
<p style="color:var(--mut);font-size:13px;text-align:center">Runs only on your own computer (localhost) &middot; results are saved in the <code>runs/web</code> folder</p>
</main>
<script>
const $=id=>document.getElementById(id);let content=null,fname="",timer=null;
const f2=x=>x==null?"-":Math.round(x*100)+"%",esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function pick(file){if(!file)return;fname=file.name;const r=new FileReader();
  r.onload=()=>{content=r.result;$("drop").classList.add("has");$("fname").textContent="Selected: "+file.name+" ("+content.split("\n").filter(l=>l.trim()).length+" exercises)"};r.readAsText(file)}
$("drop").onclick=()=>$("file").click();$("file").onchange=e=>pick(e.target.files[0]);
$("drop").ondragover=e=>{e.preventDefault();$("drop").classList.add("over")};
$("drop").ondragleave=()=>$("drop").classList.remove("over");
$("drop").ondrop=e=>{e.preventDefault();$("drop").classList.remove("over");pick(e.dataTransfer.files[0])};
async function start(sample){
  $("err").textContent="";
  if(!sample&&!content){$("err").textContent="Please choose an exercises file first (step 1), or press \"Try it with the built-in sample\".";return}
  $("go").disabled=$("sample").disabled=true;
  const body={sample,content,filename:fname,llm:$("llm").value,method:$("method").value,generations:+$("gens").value,candidates:+$("cands").value,budget_usd:$("usd").value};
  try{const r=await fetch("/api/jobs",{method:"POST",body:JSON.stringify(body)});const j=await r.json();
    if(!r.ok)throw new Error(j.error);watch(j.id);$("run").scrollIntoView({behavior:"smooth"})}
  catch(e){$("err").textContent=e.message}
  $("go").disabled=$("sample").disabled=false}
$("go").onclick=()=>start(false);$("sample").onclick=()=>start(true);
function watch(id){$("run").classList.remove("hide");$("final").classList.add("hide");clearInterval(timer);
  const tick=async()=>{const j=await (await fetch("/api/jobs/"+id)).json();render(j);if(j.status!=="running")clearInterval(timer)};
  tick();timer=setInterval(tick,1500)}
const LABEL={done:"finished",running:"working...",error:"problem"};
function render(j){
  $("title").textContent=j.file;$("pill").textContent=LABEL[j.status]||j.status;$("pill").className="pill "+j.status;
  $("meta").textContent=`${j.n_tasks} exercises · ${Math.round(j.elapsed)} s`+(j.isolated?"":" · not isolated (practice only)");
  $("spin").classList.toggle("hide",j.status!=="running");
  const acc=j.gens.filter(g=>g.id!==0&&g.accepted).length,tried=j.gens.filter(g=>g.id!==0).length;
  $("plain").textContent=j.status==="running"?(tried?`So far it has tried ${tried} round(s) and kept ${acc} improvement(s). You can leave this page open; it updates by itself.`:"Measuring the starting point first..."):"";
  $("rows").innerHTML=j.gens.map(g=>`<tr class="${g.accepted?"acc":"rej"}"><td>${g.id===0?"start":g.id}</td><td>${g.id===0?"starting point":g.accepted?"kept":"rejected"}</td>
    <td>${esc(g.change.join("; ")||"original instructions").slice(0,110)}</td><td>${f2(g.train)}</td><td>${f2(g.val)}</td><td>${esc(g.reasons.join("; ")).slice(0,140)}</td></tr>`).join("");
  if(j.error){$("plain").textContent="Something went wrong: "+j.error}
  if(j.result){const t=j.result.final.test,v=j.result.final.val;$("final").classList.remove("hide");
    $("stats").innerHTML=`<div><b>${f2(t.verified)}</b><span>of brand-new problems solved</span></div><div><b>${f2(v.verified)}</b><span>of checking problems solved</span></div>
    <div><b>${j.result.evolution.accepted} of ${j.result.evolution.proposed}</b><span>suggested changes were kept</span></div><div><b>$${j.result.evolution.usd.toFixed(4)}</b><span>cost</span></div>`;
    const mw=$("mockwarn");if(j.llm==="mock"){mw.className="note";mw.innerHTML="<b>Practice only.</b> The pretend AI is a stand-in, so these scores are not real results."}else{mw.className="hide"}
    $("dash").href="/runs/"+j.id+"/dashboard.html";$("scaf").textContent=JSON.stringify(j.result.scaffold,null,2)}}
</script></body></html>"""
