"""Candidate runner: loads a policy bundle, runs tasks in isolated workspaces, writes ScoreReports.

Bundle layout (policy-only): <candidate>/bundle.json + <candidate>/<policy_file>.
`skills_dir` in bundle.json, if present, is ignored.

Live mode calls run_agent.AIAgent (signature verified against the pinned run_agent.py).
Mock mode runs offline: no model, deterministic from --seed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.scorer.report import aggregate, build_score_report, write_score_report
from bench.tasks.catalog import (Task, apply_reference, list_tasks_for_split,
                                 prepare_workspace)
from bench.tasks.validators import validate_task
from bench.trajectories.capture import TrajectoryWriter, trajectory_ref

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CFG = REPO_ROOT / "bench" / "config" / "baseline.json"
MAX_POLICY_BYTES = 64 * 1024


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class Bundle:
    candidate_id: str
    parent_id: Optional[str]
    policy_text: str
    policy_sha256: str
    path: Path


def load_baseline_config() -> Dict[str, Any]:
    with open(BASELINE_CFG, encoding="utf-8") as f:
        return json.load(f)


def load_bundle(path: Path | str) -> Bundle:
    root = Path(path).resolve()
    meta_path = root / "bundle.json"
    if not meta_path.is_file():
        raise BundleError(f"bundle.json not found in {root}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BundleError(f"bundle.json is not valid JSON: {e}") from e
    cid = meta.get("candidate_id")
    pf = meta.get("policy_file")
    if not isinstance(cid, str) or not cid.strip():
        raise BundleError("bundle.json: candidate_id must be a non-empty string")
    if not isinstance(pf, str) or not pf.strip():
        raise BundleError("bundle.json: policy_file must be a non-empty string")
    policy_path = (root / pf).resolve()
    if root not in policy_path.parents:
        raise BundleError("bundle.json: policy_file must be inside the bundle directory")
    if not policy_path.is_file():
        raise BundleError(f"policy file not found: {pf}")
    raw = policy_path.read_bytes().replace(bytes([13,10]),bytes([10]))
    if len(raw) > MAX_POLICY_BYTES:
        raise BundleError(f"policy file exceeds {MAX_POLICY_BYTES} bytes")
    return Bundle(cid, meta.get("parent_id"), raw.decode("utf-8"), hashlib.sha256(raw).hexdigest(), root)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(candidate_id: str, split: str) -> str:
    return f"{candidate_id}-{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


def _ref(out_dir: Path, run_id: str, name: str) -> str:
    """trajectory_ref: POSIX path relative to repo root when out_dir is inside it, else the default layout."""
    target = out_dir.resolve() / "trajectories" / f"{name}.jsonl"
    try:
        return target.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return trajectory_ref(run_id, name)


def _mock_agent(task: Task, ws: Path, seed: int, repeat: int, traj: TrajectoryWriter) -> Dict[str, int]:
    h = int(hashlib.sha256(f"{seed}:{task.task_id}:{repeat}".encode()).hexdigest(), 16)
    solves = h % 4 != 0  # ~75% solve rate, deterministic
    if solves:
        apply_reference(task, ws)
    traj.event("mock_agent", solved=solves)
    return {"tokens": 1000 + h % 500, "steps": 3 + h % 5}


def _live_agent(task: Task, bundle: Bundle, ws: Path, model: str, provider: Optional[str],
                timeout: float, max_steps: int, traj: TrajectoryWriter) -> Dict[str, int]:
    from run_agent import AIAgent  # lazy: mock mode and tests never import Hermes

    kwargs: Dict[str, Any] = dict(
        model=model, max_iterations=max_steps, run_budget_seconds=timeout,
        quiet_mode=True, save_trajectories=False,
        ephemeral_system_prompt=bundle.policy_text,
        skip_context_files=True, skip_memory=True, skip_background_review=True,
        cwd=str(ws),
    )
    if provider:
        kwargs["provider"] = provider
    agent = AIAgent(**kwargs)
    result = agent.run_conversation(task.prompt)
    for m in result.get("messages", []):
        traj.event("message", message=m)
    traj.event("agent_result", completed=result.get("completed"), api_calls=result.get("api_calls"),
               final_response=result.get("final_response"))
    tokens = getattr(agent, "session_total_tokens", 0) or (
        getattr(agent, "session_input_tokens", 0) + getattr(agent, "session_output_tokens", 0))
    steps = result.get("api_calls") or getattr(agent, "session_api_calls", 0) or 0
    return {"tokens": int(tokens), "steps": int(steps)}


def run_task(task: Task, bundle: Bundle, *, run_id: str, out_dir: Path, repeat: Optional[int],
             seed: int, timeout: float, max_steps: int, model: Optional[str],
             provider: Optional[str], mock: bool) -> Dict[str, Any]:
    name = task.task_id if repeat is None else f"{task.task_id}__r{repeat}"
    traj_path = out_dir / "trajectories" / f"{name}.jsonl"
    outcome, reason, usage = "failed", None, {"tokens": 0, "steps": 0}
    t0 = time.monotonic()
    with TrajectoryWriter(traj_path) as traj, tempfile.TemporaryDirectory(
            prefix="benchws_", ignore_cleanup_errors=True) as td:
        ws = prepare_workspace(task, Path(td) / "ws")
        traj.event("task_start", task_id=task.task_id, split=task.split, candidate_id=bundle.candidate_id,
                   policy_sha256=bundle.policy_sha256, model=model, provider=provider, mock=mock,
                   seed=seed, timeout=timeout, max_steps=max_steps)
        try:
            if mock:
                usage = _mock_agent(task, ws, seed, repeat or 1, traj)
            else:
                usage = _live_agent(task, bundle, ws, model, provider, timeout, max_steps, traj)
            res = validate_task(task, ws)
            traj.event("validation", passed=res.passed, reason=res.reason)
            elapsed = time.monotonic() - t0
            if res.passed:
                outcome, reason = "passed", None
            elif elapsed >= timeout:
                outcome, reason = "timeout", f"time budget {timeout:.0f}s exhausted; validator: {res.reason}"
            else:
                outcome, reason = "failed", res.reason or "validator failed"
        except Exception as e:  # agent/infra error, not a task failure
            outcome, reason = "error", f"{type(e).__name__}: {e}"[:300]
            traj.event("error", error=reason)
        elapsed = time.monotonic() - t0
        traj.event("task_end", outcome=outcome, time_seconds=round(elapsed, 3), **usage)
    report = build_score_report(
        run_id=run_id, task_id=task.task_id, candidate_id=bundle.candidate_id, split=task.split,
        outcome=outcome, tokens=usage["tokens"], steps=usage["steps"], time_seconds=elapsed,
        trajectory_ref=_ref(out_dir, run_id, name), failure_reason=reason)
    write_score_report(report, out_dir / "reports" / f"{name}.json")
    return report


def run_split(candidate_path: Path | str, split: str, *, repeat: int, seed: int,
              out: Optional[Path | str], timeout: float, max_steps: int,
              model: Optional[str], provider: Optional[str], mock: bool) -> Dict[str, Any]:
    """Run every task of `split` `repeat` times. Raises PermissionError for sealed heldout."""
    tasks = list_tasks_for_split(split)  # enforces the held-out gate
    bundle = load_bundle(candidate_path)
    run_id = make_run_id(bundle.candidate_id, split)
    out_dir = Path(out) if out else REPO_ROOT / "bench" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    per_repeat: List[List[Dict[str, Any]]] = []
    for r in range(1, repeat + 1):
        rep = None if repeat == 1 else r
        per_repeat.append([
            run_task(t, bundle, run_id=run_id, out_dir=out_dir, repeat=rep, seed=seed + r - 1,
                     timeout=timeout, max_steps=max_steps, model=model, provider=provider, mock=mock)
            for t in tasks])
    summary = {
        "run_id": run_id, "candidate_id": bundle.candidate_id, "parent_id": bundle.parent_id,
        "policy_sha256": bundle.policy_sha256, "split": split, "repeat": repeat, "seed": seed,
        "timeout_seconds": timeout, "max_steps": max_steps, "model": model, "provider": provider,
        "mock": mock, "started_at": started, "finished_at": _utc_now(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "pinned_commit": load_baseline_config().get("pinned_commit"),
        "task_ids": [t.task_id for t in tasks],
        "aggregate": aggregate(per_repeat),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    summary["out_dir"] = str(out_dir)
    return summary
