"""Static, self-contained HTML and CSV; no polling, external scripts or uploads."""

import csv
from html import escape
import json
from pathlib import Path

from .common import write
from .scores import METRICS, cohort


def plot(rows, keys, drifts):
    n, w, h = len(rows), 460, 210
    x = lambda i: 36 + i * 410 / (n - 1)
    y = lambda v: 180 - v * 1.5
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Task scores">']
    for v in (0, 25, 50, 75, 100):
        parts.append(f'<path d="M36,{y(v)} H446" stroke="#e1eaf2"/><text x="6" y="{y(v)+4}">{v}</text>')
    for d in drifts:
        parts.append(f'<path d="M{x(d-1)},25 V180" stroke="#7394ad" stroke-dasharray="4 4"/>'
                     f'<text x="{x(d-1)}" y="17">T{d}</text>')
    for key, color, dash in ((keys[0], "#205f93", ""), (keys[1], "#71b2d6", 'stroke-dasharray="5 3"')):
        values = [r[key] for r in rows]
        pen, path = False, []
        for i, v in enumerate(values):
            if v is None:
                pen = False
                continue
            path.append(f'{"L" if pen else "M"}{x(i):.2f},{y(v):.2f}')
            pen = True
            parts.append(f'<circle cx="{x(i)}" cy="{y(v)}" r="2" fill="{color}"><title>'
                         f'T{i+1} {key}: {v:.2f}</title></circle>')
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="2" {dash}/>')
    for i in sorted(set([0, n - 1, *range(3, n, 4)])):
        parts.append(f'<text x="{x(i)}" y="203">{i+1}</text>')
    return "".join(parts) + "</svg>"


def fmt(v):
    return "N/A" if v is None else f"{v:.2f}"


def generate(inputs, output):
    output = Path(output)
    result = cohort(inputs)
    write(output / "results.json", result)
    columns = ["model", *METRICS, "AVG"]
    with (output / "model_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"model": r["model"], **r["metrics"]} for r in result["models"])
    with (output / "task_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "case_id", "position", "w_exp", "wo_exp", "S", "R"])
        writer.writeheader()
        writer.writerows({"model": p["model"], "case_id": p["case_id"], **row} for p in inputs for row in p["positions"])
    table = '<div class="scroll"><table><tr><th>Model</th>' + "".join(f"<th>{c}</th>" for c in columns[1:]) + "</tr>"
    for row in sorted(result["models"], key=lambda r: -(r["metrics"]["AVG"] or 0)):
        table += f'<tr><td>{escape(row["model"])}</td>' + "".join(f'<td>{fmt(row["metrics"][c])}</td>' for c in columns[1:]) + "</tr>"
    table += "</table></div>"
    sections = []
    for cid in result["environments"]:
        cards = []
        for pair in sorted((p for p in inputs if p["case_id"] == cid), key=lambda p: p["model"]):
            rows = pair["positions"]
            tasks = '<table><tr><th>Task</th><th>w exp</th><th>w/o exp</th><th>S</th><th>R</th></tr>'
            tasks += "".join(f'<tr><td>T{r["position"]}</td>' + "".join(f'<td>{fmt(r[k])}</td>' for k in ("w_exp", "wo_exp", "S", "R")) + "</tr>" for r in rows)
            tasks += "</table>"
            cards.append(f'<article><h3>{escape(pair["model"])}</h3><div class="plots">'
                         '<div><h4>Performance: w exp / w/o exp (dashed)</h4>' + plot(rows, ("w_exp", "wo_exp"), pair["drifts"]) + '</div>'
                         '<div><h4>Experience: S / R (dashed)</h4>' + plot(rows, ("S", "R"), pair["drifts"]) + '</div></div>'
                         '<p>' + " | ".join(f'{k} {fmt(pair["metrics"][k])}' for k in columns[1:]) + '</p>'
                         '<details><summary>Task-level values</summary>' + tasks + '</details>'
                         '<details><summary>Experience recovery details</summary><pre>' + escape(json.dumps(pair["events"], indent=2, ensure_ascii=False)) + '</pre></details></article>')
        sections.append(f'<details class="env" id="{escape(cid)}"><summary>{escape(cid)}</summary>' + "".join(cards) + '</details>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReScopeBench Results</title><style>
:root{color:#193d5b;background:#eff5f9;font:14px/1.6 "Avenir Next",Georgia,serif}body{margin:0;padding:32px;max-width:1600px;margin:auto;background:radial-gradient(at top left,#d9eaf7,transparent 50%)}h1{font-size:34px}article,.env,.summary{background:white;border:1px solid #d2e2ec;border-radius:9px;padding:20px;margin:18px 0}summary{cursor:pointer;font-weight:bold}.plots{display:grid;grid-template-columns:1fr 1fr;gap:22px}svg{width:100%;height:auto}svg text{font:10px sans-serif;fill:#55758d}h4{margin:0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:7px;border-bottom:1px solid #dfebf3;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e9f3fa}pre{overflow:auto;max-height:420px;font-size:11px}a{color:#25638f}@media(max-width:700px){body{padding:12px}.plots{grid-template-columns:1fr}article,.env{padding:12px}}
</style><h1>ReScopeBench: Performance &amp; Experience</h1>'''
    html += f'<p>{len(result["models"])} models, {len(result["environments"])} version-matched environments. Static report; no automatic refresh.</p>'
    html += '<p>B: w/o-exp mean; W: w-exp mean; D: w-exp drift-onset score. S/R: saved structural/rule content. U: onset update coverage; H: joint-recovery timeliness. AVG: seven-column arithmetic mean, higher is better.</p>'
    html += '<p>Environment-equal aggregation. S/R are post-task snapshots, not evidence of actual reuse. Task variability is not repeated-run stability. N/A is never treated as zero.</p>'
    html += '<p><a href="model_metrics.csv">Model CSV</a> | <a href="task_scores.csv">Task CSV</a> | <a href="results.json">Full data</a></p>' + table + "".join(sections) + "</html>"
    (output / "index.html").write_text(html, encoding="utf-8")
    return {"models": len(result["models"]), "environments": len(result["environments"]), "html": str(output / "index.html")}
