"""Generate a self-refreshing HTML dashboard for a running benchmark batch."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
from functools import lru_cache
import html
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from rescopebench import board_registry


ARMS = ("native", "reset")
ARM_LABELS = {"native": "w exp", "reset": "w/o exp"}


def _is_experience_tool(name: Any) -> bool:
    return isinstance(name, str) and (name == "memory" or name.startswith("skill"))


def _experience_tool_calls(result: dict[str, Any], attempt: Path) -> int:
    recorded = result.get("native_experience_tool_calls")
    if type(recorded) is int:
        # This counter includes native review and survives exchange cleanup.
        return recorded
    foreground = sum(
        1
        for message in result.get("trajectory") or []
        if isinstance(message, dict) and message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if _is_experience_tool((call.get("function") or {}).get("name"))
    )
    review = 0
    for tool_path in (attempt / "episode" / "exchange" / "native-tools").glob("*.json"):
        row = _read(tool_path, {})
        if isinstance(row, dict) and row.get("phase") == "native_review":
            review += 1
    return foreground + review


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


@lru_cache(maxsize=8192)
def _episode_summary(path: Path, _modified_ns: int, _size: int) -> tuple[int, dict[str, Any]] | None:
    # Cache only the small display projection, never the full trajectory. File
    # identity changes invalidate the entry; execution never consumes this cache.
    promotion = _read(path)
    if not isinstance(promotion, dict):
        return None
    position = promotion.get("position")
    result = promotion.get("result")
    if type(position) is not int or not isinstance(result, dict):
        return None
    experience_calls = _experience_tool_calls(result, path.parent)
    business_calls = result.get("business_tool_calls")
    history_calls = result.get("history_tool_calls")
    total_tool_calls = (
        business_calls + history_calls + experience_calls
        if all(type(value) is int for value in (business_calls, history_calls))
        else None
    )
    return position, {
        "score": result.get("score"),
        "completed_artifacts": result.get("completed_artifacts"),
        "agent_tool_rounds": result.get("agent_tool_rounds"),
        "business_tool_calls": business_calls,
        "history_tool_calls": history_calls,
        "experience_tool_calls": experience_calls,
        "total_tool_calls": total_tool_calls,
        "task_model_calls": result.get("task_model_calls"),
        "native_review_model_calls": result.get("native_review_model_calls"),
        "model_calls": result.get("model_calls"),
        "termination": result.get("termination"),
        "promotion": str(path),
    }


def _episode_rows(run: Path, arm: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((run / arm).glob("[0-9][0-9][0-9][0-9]-*/attempt-*/promotion.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        summary = _episode_summary(path, stat.st_mtime_ns, stat.st_size)
        if summary is not None:
            position, row = summary
            rows[position] = row
    return rows


def _current_position(run: Path, arm: str) -> int:
    pointer = _read(run / arm / "current.json", {})
    checkpoint = pointer.get("checkpoint") if isinstance(pointer, dict) else None
    manifest = _read(run / arm / checkpoint / "manifest.json", {}) if checkpoint else {}
    position = manifest.get("position") if isinstance(manifest, dict) else 0
    return position if type(position) is int else 0


def _failure(run: Path, arm: str, position: int) -> str | None:
    paths = sorted((run / arm).glob(f"{position:04d}-*/attempt-*/failure.json"))
    if not paths:
        return None
    value = _read(paths[-1], {})
    if not isinstance(value, dict):
        return "failure"
    return str(value.get("error") or value.get("error_type") or "failure")[:240]


def _has_live_runner(run: Path, arm: str) -> bool:
    """Return whether another process currently owns the arm runner lock."""
    lock = run / arm / "runner.lock"
    if not lock.is_file():
        return False
    try:
        with lock.open("a") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def _has_live_controller(run: Path, arm: str) -> bool:
    """Distinguish preparation/cleanup between locked episodes from a dead runner."""
    launch = _read(run / "launched.json", {})
    for child in launch.get("children", []):
        if child.get("arm") != arm or type(child.get("pid")) is not int:
            continue
        try:
            result = subprocess.run(["ps", "-p", str(child["pid"]), "-o", "command="],
                                    capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            continue
        try:
            command = shlex.split(result.stdout.strip())
            matching_arm = (command[command.index("--run") + 1] == str(run.resolve())
                            and command[command.index("--arm") + 1] == arm)
        except (ValueError, IndexError):
            continue
        entrypoints = [Path(part).name for part in command]
        workers = {"evolution_model_comparison.py", "isolated_memory_contract_run.py", "memory_profile_run.py", "official_native_run.py"}
        controller = "evolution_smoke_run.py" in entrypoints or (
            bool(workers.intersection(entrypoints)) and "worker" in command)
        if result.returncode == 0 and controller and matching_arm:
            return True
    return False


def _operator_stopped(
    operator_status: dict[str, Any], arm_data: dict[str, dict[str, Any]], task_count: int = 16
) -> bool:
    """Completed arms do not hide an explicit stop of the remaining stream."""
    operator_arms = operator_status.get("arms")
    if not isinstance(operator_arms, dict):
        return False
    stopped_states = {"waiting", "stopped", "stopped_by_operator"}
    unfinished = [arm for arm in ARMS if arm_data[arm]["position"] < task_count]
    return bool(unfinished) and all(
        operator_arms.get(arm) in stopped_states
        and arm_data[arm].get("state") not in {"running", "preparing"}
        for arm in unfinished
    )


def collect(run_root: Path) -> dict[str, Any]:
    controller = _read(run_root / "controller" / "status.json", {})
    registry = board_registry.read_registry(run_root)
    registry_runs = registry.get("runs", {})
    controller_rows = {
        (row.get("case"), row.get("arm")): row
        for row in controller.get("arms", [])
        if isinstance(row, dict)
    }
    cases = []
    runs = run_root / "runs"
    for run in sorted(path for path in runs.iterdir() if path.is_dir()):
        review = registry_runs.get(run.name, {})
        if not isinstance(review, dict):
            review = {}
        plan = _read(run / "run-plan.json", {})
        operator_status = _read(run / "operator-status.json", {})
        operator_arms = (
            operator_status.get("arms", {}) if isinstance(operator_status, dict) else {}
        )
        tasks = plan.get("tasks", []) if isinstance(plan, dict) else []
        task_count = len(tasks) or 16
        schedule = plan.get("lifecycle_schedule")
        drift_positions = [row["before_task"] for row in schedule["events"]] if schedule else [9]
        arm_data = {}
        for arm in ARMS:
            durable = _episode_rows(run, arm)
            position = _current_position(run, arm)
            # Symlinked cases retain the base batch controller. Repaired cases are
            # real directories and must never inherit their predecessor's state.
            control = controller_rows.get((run.name, arm), {}) if run.is_symlink() else {}
            # An unfinished attempt survives crashes and manual termination, so
            # its directory is evidence to resume, not evidence of liveness.
            # The non-blocking runner lock is the authoritative local signal.
            if position == task_count:
                state = "complete"
            elif _has_live_runner(run, arm):
                state = "running"
            elif _has_live_controller(run, arm):
                state = "preparing"
            elif arm in operator_arms:
                state = operator_arms[arm]
            else:
                control_state = control.get("state")
                state = control_state if control_state in {
                    "queued", "waiting", "stopped", "stopped_by_operator", "failed"
                } else "waiting"
            active_position = position + 1 if state == "running" and position < task_count else None
            arm_data[arm] = {
                "state": state,
                "position": position,
                "active_position": active_position,
                "failure": _failure(run, arm, position + 1),
                "episodes": durable,
            }
        cases.append(
            {
                "run_id": run.name,
                "case_id": plan.get("case_id", run.name),
                "case_version": plan.get("case_version", "unknown"),
                "model": plan.get("model", "unknown"),
                "created_at": plan.get("created_at") or review.get("registered_at") or "",
                "tasks": tasks,
                "task_count": task_count,
                "drift_positions": drift_positions,
                "arms": arm_data,
                "finalized": review.get("finalized") is True,
                "finalized_at": review.get("finalized_at"),
                "review_note": review.get("review_note"),
                "stopped": _operator_stopped(operator_status, arm_data, task_count),
                "stop_reason": operator_status.get("reason"),
            }
        )
    return {
        "run_root": str(run_root),
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "controller": controller,
        "cases": cases,
    }


def _number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _polylines(values: list[float | None], x0: float, x1: float, y0: float, y1: float) -> list[str]:
    segments: list[list[str]] = []
    current: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            if current:
                segments.append(current)
                current = []
            continue
        x = x0 + index * ((x1 - x0) / max(1, len(values) - 1))
        y = y0 + (100 - float(value)) * ((y1 - y0) / 100)
        current.append(f"{x:.1f},{y:.1f}")
    if current:
        segments.append(current)
    return [" ".join(segment) for segment in segments]


def _chart(native: list[float | None], reset: list[float | None], chart_id: str,
           drift_positions: list[int] | None = None) -> str:
    width, height = 760, 220
    x0, x1, y0, y1 = 48.0, 742.0, 18.0, 176.0
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="score curve">',
        f'<rect class="plot-bg" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/>',
    ]
    for score in (0, 25, 50, 75, 100):
        y = y0 + (100 - score) * ((y1 - y0) / 100)
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end">{score}</text>')
    count = max(len(native), len(reset), 1)
    spacing = (x1 - x0) / max(1, count - 1)
    for index in range(count):
        x = x0 + index * spacing
        if count > 24 and index % 2 and index != count - 1:
            continue
        parts.append(f'<text class="axis" x="{x:.1f}" y="202" text-anchor="middle">T{index + 1:02d}</text>')
    for event_index, position in enumerate([9] if drift_positions is None else drift_positions, 1):
        drift_x = x0 + (position - 1) * spacing
        parts.append(f'<line class="drift" data-before-task="{position}" x1="{drift_x:.1f}" y1="{y0}" x2="{drift_x:.1f}" y2="{y1}"/>')
        parts.append(f'<text class="drift-label" x="{drift_x + 5:.1f}" y="{y0 + 11:.1f}">E{event_index} / T{position:02d}</text>')
    for values, css, arm in ((native, "native-line", "native"), (reset, "reset-line", "reset")):
        for points in _polylines(values, x0, x1, y0, y1):
            parts.append(f'<polyline class="score-line {css}" points="{points}"/>')
        for index, value in enumerate(values):
            if value is None:
                continue
            x = x0 + index * spacing
            y = y0 + (100 - float(value)) * ((y1 - y0) / 100)
            parts.append(
                f'<circle class="point {css}" cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
                f'onclick="{html.escape(f"openTask({json.dumps(chart_id)},{index + 1})", quote=True)}"><title>{ARM_LABELS[arm]} T{index + 1:02d}: {_number(value)}</title></circle>'
            )
    parts.append('</svg>')
    return "".join(parts)


FOCUS_STATE_SCRIPT = r"""
function initialFocusState(stored, runIds, defaults) {
  const restored = stored && stored.version === 2;
  const state = restored ? {...stored} : {version: 2, focus: defaults, libraryOpen: true, query: '', scroll: 0, tables: []};
  const available = new Set(runIds);
  const known = new Set(Array.isArray(state.knownRuns) ? state.knownRuns : []);
  const added = restored ? runIds.filter(id => !known.has(id)) : [];
  const selected = Array.isArray(state.focus) ? state.focus : [];
  state.focus = [...new Set([...added, ...selected])].filter(id => available.has(id));
  state.knownRuns = runIds;
  state.tables = Array.isArray(state.tables) ? state.tables.filter(id => available.has(id)) : [];
  return state;
}
"""


FOCUS_SCRIPT = FOCUS_STATE_SCRIPT + r"""
const cards = new Map([...document.querySelectorAll('.case-card')].map(c => [c.dataset.runId, c]));
const monitoring = document.getElementById('monitoring');
const library = document.getElementById('library');
const shelf = document.getElementById('library-cards');
const search = document.getElementById('run-search');
let stored = null;
try { stored = JSON.parse(localStorage.getItem(BOARD_KEY)); } catch (_) {}
let state = initialFocusState(stored, [...cards.keys()], DEFAULT_FOCUS);
function saveSelection() {
  state.scroll = window.scrollY;
  try { localStorage.setItem(BOARD_KEY, JSON.stringify(state)); }
  catch (_) { document.getElementById('storage-note').textContent = 'Browser storage unavailable; selection may reset on refresh.'; }
}
function filterLibrary() {
  const q = search.value.trim().toLowerCase();
  let visible = 0;
  for (const card of shelf.children) {
    card.hidden = !!q && !card.dataset.search.includes(q);
    if (!card.hidden) visible++;
  }
  document.getElementById('library-empty').hidden = visible > 0;
}
function arrange() {
  for (const id of state.focus) {
    const card = cards.get(id);
    monitoring.appendChild(card);
    card.hidden = false;
    card.open = true;
  }
  for (const [id, card] of cards) {
    if (state.focus.includes(id)) continue;
    shelf.appendChild(card);
    card.open = false;
  }
  document.getElementById('focus-count').textContent = state.focus.length;
  document.getElementById('library-count').textContent = cards.size - state.focus.length;
  document.getElementById('selection-summary').textContent = `Showing ${state.focus.length} of ${cards.size} runs. ${cards.size - state.focus.length} collapsed below.`;
  document.getElementById('focus-empty').hidden = state.focus.length > 0;
  filterLibrary();
}
function choose(id, selected, scroll) {
  state.focus = state.focus.filter(x => x !== id);
  if (selected) state.focus.unshift(id);
  arrange();
  saveSelection();
  if (scroll && selected) cards.get(id).scrollIntoView({behavior: 'smooth', block: 'start'});
}
for (const [id, card] of cards) {
  card.addEventListener('toggle', () => {
    if (card.open !== state.focus.includes(id)) choose(id, card.open, true);
  });
  card.querySelector('[data-unfocus]').addEventListener('click', () => choose(id, false, false));
  const table = card.querySelector('.task-details');
  table.open = state.tables.includes(id);
  table.addEventListener('toggle', () => {
    state.tables = state.tables.filter(x => x !== id);
    if (table.open) state.tables.push(id);
    saveSelection();
  });
}
library.open = state.libraryOpen === true;
search.value = typeof state.query === 'string' ? state.query : '';
library.addEventListener('toggle', () => { state.libraryOpen = library.open; saveSelection(); });
search.addEventListener('input', () => { state.query = search.value; filterLibrary(); saveSelection(); });
document.getElementById('focus-running').addEventListener('click', () => {
  state.focus = [...cards].filter(([, c]) => c.dataset.running === 'true').map(([id]) => id);
  arrange(); saveSelection();
});
document.getElementById('focus-all').addEventListener('click', () => {
  state.focus = [...cards.keys()];
  arrange(); saveSelection();
});
document.getElementById('collapse-all').addEventListener('click', () => {
  state.focus = []; arrange(); saveSelection();
});
document.getElementById('browse-runs').addEventListener('click', () => {
  library.open = true;
  library.scrollIntoView({behavior: 'smooth', block: 'start'});
  search.focus({preventScroll: true});
});
function openTask(c, p) {
  if (c === '__average__') return;
  const d = TASKS[c + ':' + p];
  if (!d) return;
  document.getElementById('modal-title').textContent = d.case + ' / T' + String(p).padStart(2, '0');
  document.getElementById('modal-meta').textContent = d.version + ' / ' + (d.event_id || '');
  document.getElementById('modal-query').textContent = d.query;
  document.getElementById('modal-results').textContent = JSON.stringify({native: d.native, reset: d.reset}, null, 2);
  document.getElementById('modal').classList.add('open');
}
function closeTask() { document.getElementById('modal').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeTask(); });
const savedScroll = Number(state.scroll) || 0;
arrange();
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
requestAnimationFrame(() => window.scrollTo(0, savedScroll));
window.addEventListener('beforeunload', saveSelection);
setInterval(() => {
  if (document.hidden || document.getElementById('modal').classList.contains('open') || document.activeElement === search) return;
  saveSelection();
  window.location.reload();
}, 10000);
"""


def _default_focus(cases: list[dict[str, Any]]) -> list[str]:
    recent = sorted(cases, key=lambda case: case.get("created_at", ""), reverse=True)
    if len(recent) <= 8:
        return [case["run_id"] for case in recent]
    active = [case for case in recent if any(case["arms"][arm]["state"] in {"running", "preparing"} for arm in ARMS)]
    ordered = active + [case for case in recent if case not in active]
    return [case["run_id"] for case in ordered[:4]]


def render(data: dict[str, Any]) -> str:
    cases = data["cases"]
    finalized = sum(case.get("finalized") is True for case in cases)
    running_arms = sum(
        case["arms"][arm]["state"] in {"running", "preparing"} for case in cases for arm in ARMS
    )
    total_done = sum(len(case["arms"][arm]["episodes"]) for case in cases for arm in ARMS)
    total = sum(case.get("task_count", 16) * 2 for case in cases)
    max_tasks = max((case.get("task_count", 16) for case in cases), default=16)

    average: dict[str, list[float | None]] = {arm: [] for arm in ARMS}
    sample_counts: dict[str, list[int]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for position in range(1, max_tasks + 1):
            values = [
                case["arms"][arm]["episodes"][position]["score"]
                for case in cases
                if position in case["arms"][arm]["episodes"]
                and isinstance(case["arms"][arm]["episodes"][position].get("score"), (int, float))
            ]
            average[arm].append(round(sum(values) / len(values), 2) if values else None)
            sample_counts[arm].append(len(values))

    task_payload: dict[str, Any] = {}
    cards = []
    for case in cases:
        case_id = case["case_id"]
        case_key = f"{case_id}@{case['run_id']}"
        task_count = case.get("task_count", 16)
        native_values = [case["arms"]["native"]["episodes"].get(i, {}).get("score") for i in range(1, task_count + 1)]
        reset_values = [case["arms"]["reset"]["episodes"].get(i, {}).get("score") for i in range(1, task_count + 1)]
        rows = []
        for position in range(1, task_count + 1):
            task = case["tasks"][position - 1] if position <= len(case["tasks"]) else {}
            key = f"{case_key}:{position}"
            task_payload[key] = {
                "case": case_id,
                "version": case["case_version"],
                "position": position,
                "event_id": task.get("event_id"),
                "query": task.get("query", "Task content unavailable"),
                "native": case["arms"]["native"]["episodes"].get(position),
                "reset": case["arms"]["reset"]["episodes"].get(position),
            }
            cells = []
            for arm in ARMS:
                episode = case["arms"][arm]["episodes"].get(position)
                if episode:
                    experience_label = "ephemeral experience" if arm == "reset" else "experience"
                    delivered = episode.get("completed_artifacts")
                    delivery_label = (
                        f'<small title="Unlike the milestone score, this counts artifacts meeting every required fact.">'
                        f'{delivered} fully matching artifacts</small>'
                        if type(delivered) is int else ""
                    )
                    cell = (
                        f'<strong>{_number(episode.get("score"))}</strong>'
                        f'{delivery_label}'
                        f'<small title="Tool rounds count foreground assistant turns containing tools; '
                        f'business, history, and experience are mutually exclusive tool-call categories; '
                        f'tool rounds count foreground assistant turns containing one or more tools; LLM calls include '
                        f'post-task review requests.">'
                        f'{_number(episode.get("total_tool_calls"))} tools = '
                        f'{_number(episode.get("business_tool_calls"))} business + '
                        f'{_number(episode.get("history_tool_calls"))} history + '
                        f'{_number(episode.get("experience_tool_calls"))} {experience_label}<br>'
                        f'{_number(episode.get("model_calls"))} LLM calls '
                        f'({_number(episode.get("task_model_calls"))} task + '
                        f'{_number(episode.get("native_review_model_calls"))} review) · '
                        f'{_number(episode.get("agent_tool_rounds"))} task tool rounds</small>'
                    )
                elif case["arms"][arm]["active_position"] == position:
                    cell = '<span class="pulse">running</span>'
                else:
                    cell = '<span class="muted">pending</span>'
                cells.append(f'<td class="{arm}">{cell}</td>')
            rows.append(
                f'<tr onclick="{html.escape(f"openTask({json.dumps(case_key)},{position})", quote=True)}"><th>T{position:02d}</th>'
                + "".join(cells)
                + '<td class="open">view task</td></tr>'
            )
        states = " / ".join(
            f'{ARM_LABELS[arm]} {case["arms"][arm]["position"]}/{task_count} · {html.escape(str(case["arms"][arm]["state"]))}'
            for arm in ARMS
        )
        if case.get("finalized"):
            review_label = "FINALIZED"
            review_class = "finalized"
            review_title = case.get("review_note") or case.get("finalized_at") or "Finalized"
        elif case.get("stopped"):
            review_label = "STOP"
            review_class = "stopped"
            review_title = case.get("stop_reason") or case.get("review_note") or "Stopped by operator"
        else:
            review_label = "IN REVIEW"
            review_class = "in-review"
            review_title = case.get("review_note") or "Not finalized"
        cards.append(
            f'''<details class="case-card" id="case-{html.escape(case_key)}" data-run-id="{html.escape(case['run_id'])}" data-running="{str(any(case['arms'][a]['state'] in {'running', 'preparing'} for a in ARMS)).lower()}" data-search="{html.escape(f"{case_id} {case['case_version']} {case['run_id']} {case['model']}".lower())}">
<summary><span><b>{html.escape(case_id)}</b><span class="review-badge {review_class}" title="{html.escape(str(review_title))}">{review_label}</span><small>{html.escape(case["case_version"])} · {html.escape(case["run_id"])}</small></span><span class="state">{states}<span class="selection-hint">Expand to monitor above</span></span></summary>
<div class="case-body"><div class="case-controls"><div class="legend"><i class="n"></i>w exp <i class="r"></i>w/o exp</div><button type="button" data-unfocus>Move below</button></div>
{_chart(native_values, reset_values, case_key, case.get('drift_positions', [9]))}
<details class="task-details"><summary>Scores, calls and task details</summary><div class="table-wrap"><table><thead><tr><th>Task</th><th>w exp</th><th>w/o exp</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details></div>
</details>'''
        )

    avg_samples = " · ".join(
        f"T{i:02d} n={max(sample_counts['native'][i-1], sample_counts['reset'][i-1])}"
        for i in range(1, max_tasks + 1)
        if max(sample_counts['native'][i-1], sample_counts['reset'][i-1])
    )
    payload = json.dumps(task_payload, ensure_ascii=True).replace("</", "<\\/")
    default_focus = json.dumps(_default_focus(cases)).replace("</", "<\\/")
    board_key = json.dumps("rescopebench-board-v2:" + data["run_root"]).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ReScopeBench Live</title>
<style>
:root{{--ink:#17231f;--paper:#f7f8fa;--card:#fff;--native:#1769aa;--reset:#d14b3f;--line:#d9dde3;--muted:#68717d}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:var(--paper);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px 22px 70px}} header{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:3px solid var(--ink);padding-bottom:18px}}
h1{{font-size:clamp(30px,5vw,62px);line-height:.9;margin:0;letter-spacing:-.045em}} .kicker,.mono,small,.state,table,.axis{{font-family:'SFMono-Regular',Consolas,monospace}} .kicker{{text-transform:uppercase;letter-spacing:.16em;font-size:11px}}
.summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}} .metric{{background:rgba(255,253,247,.82);border:1px solid var(--line);padding:14px}} .metric b{{display:block;font-size:25px}} .metric span{{font:11px 'SFMono-Regular',monospace;color:var(--muted)}}
.overview,.case-card{{background:var(--card);border:1px solid var(--line);box-shadow:none}} .overview{{padding:18px;margin:18px 0}} .legend{{font:12px 'SFMono-Regular',monospace;color:var(--muted);display:flex;gap:9px;align-items:center}} .legend i{{width:24px;height:3px;display:inline-block}} .legend .n{{background:var(--native)}} .legend .r{{background:repeating-linear-gradient(90deg,var(--reset) 0 7px,transparent 7px 11px)}}
.chart{{display:block;width:100%;height:auto;margin-top:8px}} .plot-bg{{fill:#fff;stroke:#c8ced6;stroke-width:1}} .grid{{stroke:#e3e6ea;stroke-width:1}} .axis{{font-size:10px;fill:#657080}} .score-line{{fill:none;stroke-width:2.25;stroke-linejoin:miter;stroke-linecap:round}} .score-line.native-line{{stroke:var(--native)}} .score-line.reset-line{{stroke:var(--reset);stroke-dasharray:7 5}} .point.native-line{{fill:#fff;stroke:var(--native)}} .point.reset-line{{fill:#fff;stroke:var(--reset)}} .point{{cursor:pointer;stroke-width:2}} .drift{{stroke:#6b7280;stroke-width:1.25;stroke-dasharray:3 4}} .drift-label{{font:9px 'SFMono-Regular',monospace;fill:#6b7280}}
.case-card{{margin:10px 0}} summary{{cursor:pointer;list-style:none;padding:15px 18px;display:flex;justify-content:space-between;align-items:center;gap:20px}} summary::-webkit-details-marker{{display:none}} summary b{{font-size:19px}} summary small{{display:block;color:var(--muted);margin-top:4px}} .state{{font-size:11px;text-align:right}} .case-body{{border-top:1px solid var(--line);padding:14px 18px 20px}}
.review-badge{{display:inline-block;margin-left:9px;padding:3px 6px;border:1px solid currentColor;font:9px 'SFMono-Regular',monospace;letter-spacing:.08em;vertical-align:3px}} .review-badge.finalized{{color:#1f7a4d;background:#edf8f1}} .review-badge.in-review{{color:#7b6542;background:#fbf5e8}} .review-badge.stopped{{color:#a32222;background:#fff0f0;border-color:#cf4b4b;font-weight:700}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:9px;border-bottom:1px solid #e7e2d7;text-align:left}} tbody tr{{cursor:pointer}} tbody tr:hover{{background:#eef5ef}} td strong{{font-size:16px}} td small{{display:block;color:var(--muted);margin-top:3px}} .native strong{{color:var(--native)}} .reset strong{{color:var(--reset)}} .open{{color:var(--muted)}} .pulse{{color:#9b4d24}}
.muted{{color:#9aa09b}} .samples{{font:10px 'SFMono-Regular',monospace;color:var(--muted);line-height:1.6}}
#modal{{position:fixed;inset:0;background:rgba(15,25,20,.65);display:none;padding:4vh 4vw;z-index:10}} #modal.open{{display:block}} .modal-card{{background:var(--card);max-width:920px;max-height:92vh;overflow:auto;margin:auto;padding:22px;border-top:6px solid var(--native)}} button{{float:right;border:1px solid var(--ink);background:transparent;padding:7px 11px;cursor:pointer}} pre{{white-space:pre-wrap;word-break:break-word;background:#ece9df;padding:16px;font:12px/1.55 'SFMono-Regular',Consolas,monospace}}
.board-actions,.case-controls{{display:flex;align-items:center;gap:9px;flex-wrap:wrap}} .case-controls{{justify-content:space-between}} .board-actions button,.case-controls button{{float:none;background:#fff;color:var(--native);border:1px solid var(--line);border-radius:4px;font-size:12px}} .board-actions button:hover,.case-controls button:hover{{background:#edf4fb;border-color:var(--native)}} .focus-header{{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-top:28px}} .focus-header h2{{margin:0;font-size:24px}} .count{{font:12px 'SFMono-Regular',monospace;background:#eaf1f7;padding:4px 8px;border-radius:12px;color:var(--native)}} .help{{color:var(--muted);font-size:12px;line-height:1.6}} .empty{{padding:22px;border:1px dashed var(--line);color:var(--muted)}} #monitoring .case-card{{border-left:3px solid var(--native);scroll-margin-top:18px}} .case-card[open] .selection-hint{{display:none}} .selection-hint{{display:block;color:var(--native);margin-top:7px}} #library{{margin-top:30px;border-top:2px solid var(--line)}} #library>summary{{padding:20px 0}} #library>summary b{{font-size:18px}} #library-cards .case-card summary{{padding:10px 14px}} #library-cards .case-card summary b{{font-size:14px}} .library-search{{display:block;margin:0 0 14px;color:var(--muted);font-size:12px}} #run-search{{display:block;width:100%;max-width:560px;border:1px solid var(--line);padding:11px;margin-top:6px;background:#fff;font:14px inherit}} .task-details>summary{{padding:12px 0;color:var(--native);font-size:12px;border-top:1px solid var(--line)}} .overview>summary{{padding:0}} .overview h2{{font-size:18px}} [hidden]{{display:none!important}}
summary>span,header>div{{min-width:0}} summary small,.mono{{overflow-wrap:anywhere}} .state{{flex-shrink:0}}
@media(max-width:720px){{.summary-grid{{grid-template-columns:1fr 1fr}} header,summary,.focus-header{{align-items:start;flex-direction:column}} .state{{text-align:left}}}}
</style></head><body><main>
<header><div><div class="kicker">Live benchmark telemetry</div><h1>ReScopeBench<br>Formal Run</h1></div><div class="mono">Updated {html.escape(data['updated'])}<br>{html.escape(data['run_root'])}</div></header>
<section class="summary-grid"><div class="metric"><b>{len(cases)}</b><span>RUNS</span></div><div class="metric"><b>{finalized}/{len(cases)}</b><span>FINALIZED</span></div><div class="metric"><b>{total_done}/{total}</b><span>DURABLE EPISODES</span></div><div class="metric"><b>{running_arms}</b><span>ACTIVE ARMS</span></div></section>
<div class="focus-header"><h2>Monitoring <span class="count" id="focus-count">0</span></h2><div class="board-actions"><button type="button" id="focus-all">Show all ({len(cases)})</button><button type="button" id="focus-running">Monitor running only</button><button type="button" id="collapse-all">Collapse all</button><button type="button" id="browse-runs">Browse runs</button></div></div>
<p class="help" id="selection-summary">{len(cases)} registered runs; completed runs remain available.</p>
<p class="help" id="storage-note">New runs are automatically added here. Completed runs stay selected until you collapse them. Expand a run below to monitor it, or use Show all. Your selection is remembered. Refresh pauses while reading a task or searching.</p>
<div id="focus-empty" class="empty">No runs selected. Browse the list below or select Monitor running.</div><section id="monitoring"></section>
<details id="library"><summary><b>Other runs <span class="count" id="library-count">{len(cases)}</span></b><span class="help">Compact list / expand a run to monitor it</span></summary><label class="library-search">Find by environment, version or model<input id="run-search" type="search" placeholder="Search runs" autocomplete="off"></label><div id="library-empty" class="empty" hidden>No matching runs.</div><section id="library-cards">{''.join(cards)}</section></details>
<details class="overview"><summary><h2>All-run overview</h2><span class="help">Includes unselected runs and mixed versions</span></summary><p class="help">Display summary only, not a matched model or version comparison. Task positions are not event-aligned across environments.</p><div class="legend"><i class="n"></i>w exp mean <i class="r"></i>w/o exp mean</div>{_chart(average['native'],average['reset'],'__average__', [])}<div class="samples">Completed sample counts: {html.escape(avg_samples)}</div></details>
</main><div id="modal" onclick="if(event.target===this)closeTask()"><div class="modal-card"><button onclick="closeTask()">Close</button><h2 id="modal-title"></h2><div id="modal-meta" class="mono"></div><pre id="modal-query"></pre><pre id="modal-results"></pre></div></div>
<script>const TASKS={payload}; const DEFAULT_FOCUS={default_focus}; const BOARD_KEY={board_key};
{FOCUS_SCRIPT}</script></body></html>'''


def write_dashboard(run_root: Path, output: Path) -> None:
    document = render(collect(run_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--watch", type=float, default=0)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (run_root / "runs").is_dir():
        parser.error("run root has no runs directory")
    while True:
        write_dashboard(run_root, output)
        if args.watch <= 0:
            return
        time.sleep(max(2.0, args.watch))


if __name__ == "__main__":
    main()
