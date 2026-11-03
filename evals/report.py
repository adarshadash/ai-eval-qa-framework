"""Writes eval results as JSON (machines/CI), Markdown (PR comments) and HTML (humans)."""
from __future__ import annotations

import html
import json
from pathlib import Path


def write_reports(run: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path, html_path = out_dir / "report.json", out_dir / "report.md", out_dir / "report.html"
    json_path.write_text(json.dumps(run, indent=2, default=str))
    md_path.write_text(_markdown(run))
    html_path.write_text(_html(run))
    return [json_path, md_path, html_path]


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _markdown(run: dict) -> str:
    lines = [f"# AI Eval Report - {'PASSED' if run['gate_passed'] else 'FAILED'}", "",
             f"Mode: `{run['mode']}` | Run: {run['timestamp']} | SUT model: `{run['models']['sut']}` | "
             f"Judge: `{run['models']['judge']}`", "",
             "| Suite | Status | Score | Required | Key metrics |", "|---|---|---|---|---|"]
    for s in run["suites"]:
        status = "monitor" if not s["gating"] else ("PASS" if s["passed"] else "FAIL")
        metrics = ", ".join(f"{k}={_fmt(v)}" for k, v in list(s["metrics"].items())[:5])
        lines.append(f"| {s['name']} | {status} | {s['score']:.3f} | {_fmt(s['required'])} | {metrics} |")
    failures = [(s["name"], c) for s in run["suites"] for c in s["cases"] if not c["passed"]]
    if failures:
        lines += ["", "## Failed cases", ""]
        for suite, c in failures:
            failed = [k for k, v in c["checks"].items() if not v]
            lines.append(f"- **{suite}/{c['case_id']}** - failed: {', '.join(failed)}")
    if run["regressions"]:
        lines += ["", "## Regressions vs baseline", ""] + [f"- {r}" for r in run["regressions"]]
    return "\n".join(lines) + "\n"


def _html(run: dict) -> str:
    e = html.escape
    rows = []
    for s in run["suites"]:
        status = "monitor" if not s["gating"] else ("pass" if s["passed"] else "fail")
        metrics = "<br>".join(f"{e(k)}: {e(_fmt(v))}" for k, v in s["metrics"].items())
        case_rows = "".join(
            f"<tr class='{'ok' if c['passed'] else 'bad'}'><td>{e(c['case_id'])}</td>"
            f"<td>{'PASS' if c['passed'] else 'FAIL'}</td>"
            f"<td>{e(', '.join(k for k, v in c['checks'].items() if not v) or '-')}</td>"
            f"<td><code>{e(json.dumps(c['details'], default=str)[:400])}</code></td></tr>"
            for c in s["cases"])
        traces = ""
        if s.get("traces"):
            traces = "".join(f"<tr class='{'bad' if t['flags'] else 'ok'}'><td>{e(t['id'])}</td>"
                             f"<td>{e(', '.join(t['flags']) or 'clean')}</td><td>{t['faithfulness']}</td>"
                             f"<td><code>{e(json.dumps(t['unsupported_claims']))}</code></td></tr>"
                             for t in s["traces"])
            traces = f"<table><tr><th>Trace</th><th>Flags</th><th>Faithfulness</th><th>Unsupported</th></tr>{traces}</table>"
        table = (f"<table><tr><th>Case</th><th>Result</th><th>Failed checks</th><th>Details</th></tr>{case_rows}</table>"
                 if case_rows else "")
        rows.append(f"<section><h2><span class='badge {status}'>{status.upper()}</span> {e(s['name'])} "
                    f"<small>score {s['score']:.3f} / required {e(_fmt(s['required']))} - {s['duration_s']}s</small></h2>"
                    f"<p class='metrics'>{metrics}</p>{table}{traces}</section>")
    verdict = "PASSED" if run["gate_passed"] else "FAILED"
    regressions = "".join(f"<li>{e(r)}</li>" for r in run["regressions"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Eval Report</title><style>
body{{font-family:system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1f2328;background:#fff}}
h1{{margin-bottom:4px}} small{{color:#57606a;font-weight:400;font-size:14px}}
.badge{{font-size:12px;padding:2px 8px;border-radius:10px;color:#fff;vertical-align:middle}}
.pass{{background:#1a7f37}} .fail{{background:#cf222e}} .monitor{{background:#6e7781}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 24px}}
td,th{{border:1px solid #d0d7de;padding:6px;text-align:left;vertical-align:top}}
tr.bad td{{background:#ffebe9}} code{{font-size:11px;word-break:break-all}} .metrics{{font-size:13px;color:#57606a}}
</style></head><body>
<h1>AI Eval Report: <span class="badge {'pass' if run['gate_passed'] else 'fail'}">{verdict}</span></h1>
<p>Mode <b>{e(run['mode'])}</b> - {e(run['timestamp'])} - SUT {e(run['models']['sut'])} - Judge {e(run['models']['judge'])}</p>
{'<h3>Regressions</h3><ul>' + regressions + '</ul>' if regressions else ''}
{''.join(rows)}</body></html>"""
