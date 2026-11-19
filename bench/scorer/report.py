"""ScoreReport v1 builder, validator, writer and multi-repeat aggregator.

Mirrors interfaces/score_report.v1.schema.json (13 fields, additionalProperties false).
Stdlib only.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 1
SPLITS = ("train", "heldout")
OUTCOMES = ("passed", "failed", "error", "timeout")
FIELDS = (
    "schema_version", "run_id", "task_id", "candidate_id", "split", "outcome",
    "passed", "score", "tokens", "steps", "time_seconds", "failure_reason",
    "trajectory_ref",
)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def validate_score_report(r: Dict[str, Any]) -> None:
    """Raise ValueError listing every violation; return None if valid."""
    errs: List[str] = []
    if not isinstance(r, dict):
        raise ValueError("ScoreReport must be a dict")
    extra = set(r) - set(FIELDS)
    missing = set(FIELDS) - set(r)
    if extra:
        errs.append(f"unexpected fields: {sorted(extra)}")
    if missing:
        errs.append(f"missing fields: {sorted(missing)}")
    if errs:
        raise ValueError("; ".join(errs))

    if r["schema_version"] != 1 or not _is_int(r["schema_version"]):
        errs.append("schema_version must be integer 1")
    for k in ("run_id", "task_id", "candidate_id"):
        if not _nonempty_str(r[k]):
            errs.append(f"{k} must be a non-empty string")
    if r["split"] not in SPLITS:
        errs.append(f"split must be one of {SPLITS}")
    if r["outcome"] not in OUTCOMES:
        errs.append(f"outcome must be one of {OUTCOMES}")
    if not isinstance(r["passed"], bool):
        errs.append("passed must be boolean")
    if not _is_num(r["score"]) or not (0.0 <= r["score"] <= 1.0):
        errs.append("score must be a number in [0.0, 1.0]")
    for k in ("tokens", "steps"):
        if not _is_int(r[k]) or r[k] < 0:
            errs.append(f"{k} must be an integer >= 0")
    if not _is_num(r["time_seconds"]) or r["time_seconds"] < 0:
        errs.append("time_seconds must be a number >= 0")

    if isinstance(r["passed"], bool):
        if r["passed"]:
            if r["failure_reason"] is not None:
                errs.append("failure_reason must be null when passed is true")
            if r["outcome"] != "passed":
                errs.append("outcome must be 'passed' when passed is true")
        else:
            if not _nonempty_str(r["failure_reason"]):
                errs.append("failure_reason must be a non-empty string when passed is false")
            if r["outcome"] not in ("failed", "error", "timeout"):
                errs.append("outcome must be failed/error/timeout when passed is false")

    ref = r["trajectory_ref"]
    if not _nonempty_str(ref):
        errs.append("trajectory_ref must be a non-empty string")
    elif "\\" in ref or ref.startswith("/") or ":" in ref.split("/")[0] or ".." in ref.split("/"):
        errs.append("trajectory_ref must be a POSIX relative path (no backslash, drive, or '..')")

    if errs:
        raise ValueError("; ".join(errs))


def build_score_report(
    *,
    run_id: str,
    task_id: str,
    candidate_id: str,
    split: str,
    outcome: str,
    tokens: int,
    steps: int,
    time_seconds: float,
    trajectory_ref: str,
    failure_reason: Optional[str] = None,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    """Build and validate one ScoreReport. passed is derived from outcome."""
    passed = outcome == "passed"
    if passed and failure_reason is not None:
        raise ValueError("failure_reason must be None when outcome is 'passed'")
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "candidate_id": candidate_id,
        "split": split,
        "outcome": outcome,
        "passed": passed,
        "score": (1.0 if passed else 0.0) if score is None else float(score),
        "tokens": tokens,
        "steps": steps,
        "time_seconds": float(time_seconds),
        "failure_reason": None if passed else failure_reason,
        "trajectory_ref": trajectory_ref,
    }
    validate_score_report(report)
    return report


def write_score_report(report: Dict[str, Any], path: Path | str) -> Path:
    """Validate then write one report as JSON (UTF-8, LF)."""
    validate_score_report(report)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return p


def load_score_report(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)
    validate_score_report(r)
    return r


def _mean_std(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,  # sample std
    }


def aggregate(repeats: Iterable[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate ScoreReports. Input: one list of reports per repeat.

    suite: mean/std across repeats of the per-repeat pass rate (and tokens/steps/time totals).
    per_task: pass_rate over repeats plus mean/std of tokens, steps, time_seconds.
    """
    runs = [list(rs) for rs in repeats if rs]
    for rs in runs:
        for r in rs:
            validate_score_report(r)
    pass_rates = [sum(1 for r in rs if r["passed"]) / len(rs) for rs in runs]
    scores = [statistics.fmean(r["score"] for r in rs) for rs in runs]
    totals = {
        k: [sum(r[k] for r in rs) for rs in runs] for k in ("tokens", "steps", "time_seconds")
    }
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for rs in runs:
        for r in rs:
            by_task.setdefault(r["task_id"], []).append(r)
    per_task = {}
    for tid, rs in sorted(by_task.items()):
        per_task[tid] = {
            "n": len(rs),
            "pass_rate": sum(1 for r in rs if r["passed"]) / len(rs),
            "flaky": 0 < sum(1 for r in rs if r["passed"]) < len(rs),
            "tokens": _mean_std([r["tokens"] for r in rs]),
            "steps": _mean_std([r["steps"] for r in rs]),
            "time_seconds": _mean_std([r["time_seconds"] for r in rs]),
            "failure_reasons": sorted({r["failure_reason"] for r in rs if r["failure_reason"]}),
        }
    return {
        "repeats": len(runs),
        "suite": {
            "pass_rate": _mean_std(pass_rates),
            "pass_rate_per_repeat": pass_rates,
            "score": _mean_std(scores),
            "tokens_total": _mean_std(totals["tokens"]),
            "steps_total": _mean_std(totals["steps"]),
            "time_seconds_total": _mean_std(totals["time_seconds"]),
        },
        "per_task": per_task,
    }
