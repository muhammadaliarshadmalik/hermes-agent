"""MockScorer: a deterministic fake scorer so B2 can build before B1's real scorer exists.

Team B README, week 1: B2 builds ``MockScorer``, which returns fixed fixture
``ScoreReport`` records and is deterministic given a seed.

PROVISIONAL: the report fields follow the Team B README's illustrative example.
The additional ``outcome`` field is proposed by B2 and must be aligned with B1's
ScoreReport schema (interfaces/score_report.v1.schema.json) before the week 2
interface freeze. That schema is currently empty.

Standard library only. The mock never reads or runs a candidate: it only uses the
candidate's folder name as its id.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
OUTCOMES = ("passed", "failed", "error", "timeout")
SPLITS = ("train", "heldout")


def candidate_id_from_path(candidate: str | Path) -> str:
    """Return the candidate id, which is the last path component.

    Accepts both ``/`` and ``\\`` separators so Windows and POSIX paths behave the same.
    """
    text = str(candidate).replace("\\", "/").rstrip("/")
    name = text.rsplit("/", 1)[-1]
    if not name.strip():
        raise ValueError("candidate path has no name")
    return name


def _unit(seed: int, *parts: str) -> float:
    """Deterministic float in [0, 1) derived from the seed and the given parts."""
    key = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _int_between(seed: int, low: int, high: int, *parts: str) -> int:
    """Deterministic integer in [low, high], inclusive."""
    return low + int(_unit(seed, *parts) * (high - low + 1))


class MockScorer:
    """Returns fake ScoreReports for a candidate and a split."""

    def __init__(
        self,
        fixture: dict[str, Any],
        seed: int = 0,
        allow_heldout: bool = False,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(allow_heldout, bool):
            raise ValueError("allow_heldout must be a boolean")
        self._fixture = self._validate_fixture(copy.deepcopy(fixture))
        self.seed = seed
        # This is an API guard against accidental access, not a security boundary.
        # The runner/sandbox must keep held-out fixtures and results unavailable
        # to the variation agent.
        self.allow_heldout = allow_heldout

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        seed: int = 0,
        allow_heldout: bool = False,
    ) -> MockScorer:
        fixture_path = Path(path)
        if not fixture_path.is_file():
            raise FileNotFoundError(f"fixture file not found: {fixture_path}")
        with fixture_path.open(encoding="utf-8") as handle:
            fixture = json.load(handle)
        return cls(fixture, seed=seed, allow_heldout=allow_heldout)

    def run(self, candidate: str | Path, split: str) -> list[dict[str, Any]]:
        """Return one report per task in the split, in fixture order."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if split == "heldout" and not self.allow_heldout:
            raise PermissionError(
                "held-out split is blocked; only the final evaluation may enable it"
            )

        candidate_id = candidate_id_from_path(candidate)
        bonus = self._fixture["candidate_bonus"].get(candidate_id, 0.0)
        scripted = self._fixture["scripted_outcomes"].get(candidate_id, {})
        run_id = f"mock-{self.seed}-{candidate_id}-{split}"

        reports = []
        for task in self._fixture["tasks"]:
            if task["split"] != split:
                continue
            task_id = task["task_id"]
            if task_id in scripted:
                outcome = scripted[task_id]
            else:
                probability = min(1.0, max(0.0, task["base_pass_prob"] + bonus))
                draw = _unit(self.seed, candidate_id, task_id, "outcome")
                outcome = "passed" if draw < probability else "failed"
            reports.append(self._build_report(run_id, candidate_id, split, task_id, outcome))
        return reports

    def _build_report(
            self, run_id: str, candidate_id: str, split: str, task_id: str, outcome: str
    ) -> dict[str, Any]:
        passed = outcome == "passed"

        reasons = {
            "failed": f"mock check failed for {task_id}",
            "error": f"mock harness error on {task_id}",
            "timeout": f"mock task {task_id} exceeded its time limit",
        }

        seed = self.seed

        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "task_id": task_id,
            "candidate_id": candidate_id,
            "split": split,
            "outcome": outcome,
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "tokens": _int_between(
                seed, 800, 6000, candidate_id, task_id, "tokens"
            ),
            "steps": _int_between(
                seed, 2, 30, candidate_id, task_id, "steps"
            ),
            "time_seconds": round(
                1.0 + _unit(seed, candidate_id, task_id, "time") * 59.0,
                2,
            ),
            "failure_reason": None if passed else reasons[outcome],
            "trajectory_ref": (
                f"bench/runs/{run_id}/trajectories/{task_id}.jsonl"
            ),
        }

    @staticmethod
    def _validate_fixture(fixture: Any) -> dict[str, Any]:
        if not isinstance(fixture, dict):
            raise ValueError("fixture must be a JSON object")
        schema_version = fixture.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != SCHEMA_VERSION:
            raise ValueError(f"fixture schema_version must be {SCHEMA_VERSION}")
        tasks = fixture.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("fixture needs a non-empty 'tasks' list")

        seen: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("every task must be an object")
            task_id = task.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("every task needs a non-empty string 'task_id'")
            if task_id in seen:
                raise ValueError(f"duplicate task_id: {task_id}")
            seen.add(task_id)
            split = task.get("split")
            if not isinstance(split, str) or split not in SPLITS:
                raise ValueError(f"task {task_id}: split must be one of {SPLITS}")
            prob = task.get("base_pass_prob")
            if (
                isinstance(prob, bool)
                or not isinstance(prob, (int, float))
                or not math.isfinite(prob)
                or not 0 <= prob <= 1
            ):
                raise ValueError(f"task {task_id}: base_pass_prob must be a finite number in [0, 1]")

        bonuses = fixture.get("candidate_bonus", {})
        if not isinstance(bonuses, dict):
            raise ValueError("'candidate_bonus' must be an object")
        for candidate_id, bonus in bonuses.items():
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError("candidate_bonus keys must be non-empty candidate ids")
            if (
                isinstance(bonus, bool)
                or not isinstance(bonus, (int, float))
                or not math.isfinite(bonus)
            ):
                raise ValueError(f"candidate_bonus for {candidate_id!r} must be a finite number")

        scripted_outcomes = fixture.get("scripted_outcomes", {})
        if not isinstance(scripted_outcomes, dict):
            raise ValueError("'scripted_outcomes' must be an object")
        for candidate_id, outcomes in scripted_outcomes.items():
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError("scripted_outcomes keys must be non-empty candidate ids")
            if not isinstance(outcomes, dict):
                raise ValueError(f"scripted outcomes for {candidate_id!r} must be an object")
            for task_id, outcome in outcomes.items():
                if not isinstance(task_id, str) or task_id not in seen:
                    raise ValueError(f"scripted outcome for unknown task: {task_id}")
                if not isinstance(outcome, str) or outcome not in OUTCOMES:
                    raise ValueError(
                        f"scripted outcome for {candidate_id}/{task_id} must be one of {OUTCOMES}"
                    )
        return fixture