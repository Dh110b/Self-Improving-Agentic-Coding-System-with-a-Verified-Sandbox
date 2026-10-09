"""Static HTML dashboard: version history, gate decisions, scores and cost per generation."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .archive import Archive


def _val(rec: dict, split: str, key: str):
    s = rec.get("scores", {}).get(split)
    return s[key] if s else None


def _chart(points: list[tuple[int, float]], w=520, h=140) -> str:
    if not points:
        return ""
    n = max(1, len(points) - 1)
    xy = " ".join(f"{20 + i * (w - 40) / n:.1f},{h - 20 - v * (h - 40):.1f}" for i, (_, v) in enumerate(points))
    dots = "".join(f'<circle cx="{20 + i * (w - 40) / n:.1f}" cy="{h - 20 - v * (h - 40):.1f}" r="3"/>'
                   for i, (_, v) in enumerate(points))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="accepted-lineage val score">'
            f'<line x1="20" y1="{h - 20}" x2="{w - 20}" y2="{h - 20}" stroke="currentColor" opacity=".3"/>'
            f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{xy}"/>'
            f'<g fill="#2563eb">{dots}</g></svg>')


def build(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    arch = Archive(run / "archive")
    recs = arch.all()
    cur = arch.current_id()
    lineage = set(arch.lineage(cur))
    res = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
    rows = []
    pts = []
    for r in recs:
        v = _val(r, "val", "verified")
        if v is None:
            v = _val(r, "train", "visible")
        if r["id"] in lineage and v is not None:
            pts.append((r["id"], v))
        d = r.get("diff") or {}
        ops = "; ".join(f"{o['op']}: {html.escape(str(o.get('text') or o.get('name') or o.get('k')))[:70]}"
                        for o in d.get("ops", [])) or "initial"
        status = "accepted" if r["accepted"] else "rejected"
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        rows.append(f"<tr class='{status}'><td>{r['id']}</td><td>{r['parent'] if r['parent'] is not None else ''}</td>"
                    f"<td>{status}{' (current)' if r['id'] == cur else ''}</td><td>{ops}</td>"
                    f"<td>{fmt(_val(r, 'train', 'visible'))}</td><td>{fmt(_val(r, 'train', 'verified'))}</td>"
                    f"<td>{fmt(_val(r, 'val', 'verified'))}</td><td>{fmt(_val(r, 'reg', 'verified'))}</td>"
                    f"<td>{html.escape('; '.join(r['decision']['reasons']))[:160]}</td>"
                    f"<td>{r.get('tokens', 0):,}</td><td>${r.get('usd', 0):.4f}</td></tr>")
    final = res.get("final", {})
    final_html = "".join(f"<li><b>{k}</b>: verified {v['verified']:.2f}, hidden {v['hidden']:.2f}, "
                         f"visible {v['visible']:.2f}, flagged {v['flagged']:.2f} (n={v['n']})</li>"
                         for k, v in final.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SICAS Dashboard</title>
<style>:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--ok:#e8f6ec;--bad:#fdecec}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--ok:#14301d;--bad:#3a1818}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:0;padding:16px;max-width:1200px;margin-inline:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
tr.accepted{{background:var(--ok)}}tr.rejected{{background:var(--bad)}}.wrap{{overflow-x:auto}}h1{{font-size:20px}}p,li{{color:var(--mut)}}</style></head>
<body><h1>Verification-gated scaffold evolution - {html.escape(str(res.get('method', '')))} (seed {res.get('seed', '')})</h1>
<p>Stop reason: {html.escape(str(res.get('stop', '')))}. Current scaffold: generation {cur}.</p>
<h2>Score along the accepted lineage</h2>{_chart(pts)}
<h2>Final evaluation (test touched once)</h2><ul>{final_html}</ul>
<h2>Generation history</h2><div class="wrap"><table><tr><th>id</th><th>parent</th><th>status</th><th>change</th>
<th>train (visible)</th><th>train (verified)</th><th>val (verified)</th><th>reg (verified)</th><th>gate reasons</th><th>tokens</th><th>cost</th></tr>
{''.join(rows)}</table></div></body></html>"""
    out = run / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out
