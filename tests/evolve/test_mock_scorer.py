"""Tests for evolve.mock_scorer (B2, week 1).

Run from the repo root:
    python -m uv run pytest tests/evolve/test_mock_scorer.py -v
"""

import json
from pathlib import Path

import pytest

from evolve.mock_scorer import MockScorer, candidate_id_from_path

FIXTURE = Path(__file__).parent / "fixtures" / "mock_scorer_fixture.json"
CANDIDATE = "evolve/artifacts/candidate-000"


def make_scorer(seed: int = 0, allow_heldout: bool = False) -> MockScorer:
    return MockScorer.from_file(FIXTURE, seed=seed, allow_heldout=allow_heldout)


def pass_count(reports: list[dict]) -> int:
    return sum(1 for report in reports if report["passed"])


# --- determinism -------------------------------------------------------------


def test_same_seed_gives_identical_reports():
    first = make_scorer(seed=7).run(CANDIDATE, "train")
    second = make_scorer(seed=7).run(CANDIDATE, "train")
    assert first == second


def test_different_seed_changes_results():
    first = make_scorer(seed=1).run(CANDIDATE, "train")
    second = make_scorer(seed=2).run(CANDIDATE, "train")
    signature_1 = [(r["outcome"], r["tokens"], r["steps"]) for r in first]
    signature_2 = [(r["outcome"], r["tokens"], r["steps"]) for r in second]
    assert signature_1 != signature_2


def test_different_candidates_get_different_results():
    base = make_scorer().run("candidate-000", "train")
    other = make_scorer().run("candidate-009", "train")
    assert [r["tokens"] for r in base] != [r["tokens"] for r in other]


# --- shape and invariants ----------------------------------------------------


def test_one_report_per_task_in_the_split():
    reports = make_scorer().run(CANDIDATE, "train")
    assert len(reports) == 8
    assert [r["task_id"] for r in reports] == [f"task_0{i}" for i in range(1, 9)]
    assert {r["split"] for r in reports} == {"train"}


def test_report_invariants_hold_for_every_report():
    scorer = make_scorer(allow_heldout=True)
    reports = []
    for candidate in ("candidate-000", "candidate-001", "candidate-002", "candidate-003"):
        for split in ("train", "heldout"):
            reports.extend(scorer.run(candidate, split))
    for report in reports:
        assert report["passed"] == (report["outcome"] == "passed")
        if report["passed"]:
            assert report["failure_reason"] is None
        else:
            assert isinstance(report["failure_reason"], str) and report["failure_reason"]
        assert report["outcome"] in ("passed", "failed", "error", "timeout")

        assert isinstance(report["tokens"], int) and report["tokens"] > 0
        assert isinstance(report["steps"], int) and report["steps"] > 0
        assert report["time_seconds"] > 0
        assert report["score"] in (0.0, 1.0)
        assert report["trajectory_ref"]


def test_reports_are_json_serializable():
    reports = make_scorer().run(CANDIDATE, "train")
    lines = [json.dumps(report) for report in reports]
    assert [json.loads(line) for line in lines] == reports


# --- splits and held-out isolation ------------------------------------------


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        make_scorer().run(CANDIDATE, "test")


def test_heldout_is_blocked_by_default():
    with pytest.raises(PermissionError):
        make_scorer().run(CANDIDATE, "heldout")


def test_heldout_runs_only_when_explicitly_allowed():
    reports = make_scorer(allow_heldout=True).run(CANDIDATE, "heldout")
    assert len(reports) == 4
    assert {r["split"] for r in reports} == {"heldout"}


# --- candidate handling ------------------------------------------------------


def test_candidate_id_comes_from_the_path():
    assert candidate_id_from_path("evolve/artifacts/candidate-001") == "candidate-001"
    assert candidate_id_from_path("evolve\\artifacts\\candidate-001") == "candidate-001"
    assert candidate_id_from_path("evolve/artifacts/candidate-001/") == "candidate-001"
    assert candidate_id_from_path(Path("candidate-005")) == "candidate-005"


def test_better_candidate_passes_at_least_as_many_tasks():
    scorer = make_scorer(seed=0)
    base = pass_count(scorer.run("candidate-000", "train"))
    better = pass_count(scorer.run("candidate-001", "train"))
    worse = pass_count(scorer.run("candidate-002", "train"))
    assert better >= base >= worse
    assert better > worse


def test_scripted_outcomes_override_the_random_draw():
    reports = make_scorer().run("candidate-003", "train")
    by_task = {r["task_id"]: r for r in reports}
    assert by_task["task_01"]["outcome"] == "error"
    assert by_task["task_02"]["outcome"] == "timeout"
    assert by_task["task_03"]["outcome"] == "failed"
    for task_id in ("task_01", "task_02", "task_03"):
        assert by_task[task_id]["passed"] is False
        assert by_task[task_id]["failure_reason"]


# --- fixture handling --------------------------------------------------------


def test_missing_fixture_file_raises():
    with pytest.raises(FileNotFoundError):
        MockScorer.from_file(FIXTURE.parent / "does_not_exist.json")


@pytest.mark.parametrize(
    "bad_fixture",
    [
        [],
        {},
        {"tasks": []},
        {"tasks": [{"task_id": "a", "split": "validation", "base_pass_prob": 0.5}]},
        {"tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 1.5}]},
        {"tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 0.5}] * 2},
        {
            "tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 0.5}],
            "scripted_outcomes": {"c": {"missing_task": "failed"}},
        },
        {
            "tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 0.5}],
            "scripted_outcomes": {"c": {"a": "exploded"}},
        },
    ],
)
def test_malformed_fixture_is_rejected(bad_fixture):
    if isinstance(bad_fixture, dict):
        bad_fixture = {"schema_version": 1, **bad_fixture}
    with pytest.raises(ValueError):
        MockScorer(bad_fixture)
def test_scorer_does_not_mutate_the_fixture_it_was_given():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    snapshot = json.loads(json.dumps(fixture))
    scorer = MockScorer(fixture)
    fixture["tasks"].clear()
    assert len(scorer.run(CANDIDATE, "train")) == 8
    assert snapshot["tasks"]
@pytest.mark.parametrize("version", [None, True, 2, "1"])
def test_invalid_fixture_schema_version_is_rejected(version):
    fixture = {"schema_version": version, "tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 0.5}]}
    with pytest.raises(ValueError, match="schema_version"):
        MockScorer(fixture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_bonus", []),
        ("candidate_bonus", {"candidate-000": True}),
        ("candidate_bonus", {"candidate-000": float("nan")}),
        ("candidate_bonus", {"candidate-000": float("inf")}),
        ("scripted_outcomes", []),
        ("scripted_outcomes", {"candidate-000": []}),
        ("scripted_outcomes", {"candidate-000": {"a": ["passed"]}}),
    ],
)
def test_invalid_candidate_configuration_is_rejected(field, value):
    fixture = {
        "schema_version": 1,
        "tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 0.5}],
        field: value,
    }
    with pytest.raises(ValueError):
        MockScorer(fixture)


def test_scripted_unknown_task_is_rejected_with_valid_fixture_version():
    fixture = {
        "schema_version": 1,
        "tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 0.5}],
        "scripted_outcomes": {"candidate-000": {"missing": "failed"}},
    }
    with pytest.raises(ValueError, match="unknown task"):
        MockScorer(fixture)


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_non_integer_seed_is_rejected(seed):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="seed"):
        MockScorer(fixture, seed=seed)


def test_train_reports_never_include_heldout_tasks():
    scorer = make_scorer(allow_heldout=True)
    reports = scorer.run(CANDIDATE, "train")
    assert reports
    assert all(report["split"] == "train" for report in reports)
    assert not {report["task_id"] for report in reports} & {f"task_{i:02d}" for i in range(9, 13)}


def test_report_has_readme_fields_with_expected_types():
    report = make_scorer().run(CANDIDATE, "train")[0]
    assert isinstance(report["schema_version"], int)
    assert all(isinstance(report[field], str) for field in ("run_id", "task_id", "candidate_id", "split", "trajectory_ref"))
    assert isinstance(report["passed"], bool)
    assert isinstance(report["score"], float)
    assert isinstance(report["tokens"], int)
    assert isinstance(report["steps"], int)
    assert isinstance(report["time_seconds"], float)
    assert report["failure_reason"] is None or isinstance(report["failure_reason"], str)
    assert report["outcome"] in ("passed", "failed", "error", "timeout")


@pytest.mark.parametrize(
    "task",
    [
        None,
        {"task_id": " ", "split": "train", "base_pass_prob": 0.5},
        {"task_id": "a", "split": "train", "base_pass_prob": True},
        {"task_id": "a", "split": "train", "base_pass_prob": float("nan")},
        {"task_id": "a", "split": "train", "base_pass_prob": float("inf")},
    ],
)
def test_invalid_task_entries_are_rejected(task):
    with pytest.raises(ValueError):
        MockScorer({"schema_version": 1, "tasks": [task]})


@pytest.mark.parametrize("outcome", ["unknown", None, True])
def test_invalid_scripted_outcomes_are_rejected(outcome):
    fixture = {
        "schema_version": 1,
        "tasks": [{"task_id": "a", "split": "train", "base_pass_prob": 0.5}],
        "scripted_outcomes": {"candidate-000": {"a": outcome}},
    }
    with pytest.raises(ValueError, match="scripted outcome"):
        MockScorer(fixture)


@pytest.mark.parametrize('version', [1.0, 1.5])
def test_non_integer_fixture_schema_version_is_rejected(version):
    fixture = {'schema_version': version, 'tasks': [{'task_id': 'a', 'split': 'train', 'base_pass_prob': 0.5}]}
    with pytest.raises(ValueError, match='schema_version'):
        MockScorer(fixture)
