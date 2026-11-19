"""Validator self-tests: every task must FAIL on its seed workspace and PASS with its reference solution."""
import pytest

from bench.tasks.catalog import (HELDOUT_ENV, apply_reference, list_tasks_for_split,
                                 prepare_workspace)
from bench.tasks.validators import validate_task


def _all(monkeypatch):
    monkeypatch.setenv(HELDOUT_ENV, "1")
    return list_tasks_for_split("train") + list_tasks_for_split("heldout")


def test_counts_and_unique_ids(monkeypatch):
    tasks = _all(monkeypatch)
    assert len(list_tasks_for_split("train")) == 12
    assert len(tasks) == 16 and len({t.task_id for t in tasks}) == 16


def test_heldout_gate(monkeypatch):
    monkeypatch.delenv(HELDOUT_ENV, raising=False)
    with pytest.raises(PermissionError):
        list_tasks_for_split("heldout")
    monkeypatch.setenv(HELDOUT_ENV, "0")
    with pytest.raises(PermissionError):
        list_tasks_for_split("heldout")


def _ids(monkeypatch=None):
    import os
    os.environ[HELDOUT_ENV] = "1"
    try:
        return [t.task_id for t in list_tasks_for_split("train") + list_tasks_for_split("heldout")]
    finally:
        del os.environ[HELDOUT_ENV]


@pytest.mark.parametrize("task_id", _ids())
def test_seed_fails_reference_passes(task_id, tmp_path, monkeypatch):
    monkeypatch.setenv(HELDOUT_ENV, "1")
    from bench.tasks.catalog import get_task
    t = get_task(task_id)
    ws = prepare_workspace(t, tmp_path / "ws")
    bad = validate_task(t, ws)
    assert not bad.passed and bad.reason, f"seed must fail: {bad}"
    apply_reference(t, ws)
    good = validate_task(t, ws)
    assert good.passed, good.reason
    assert good.reason is None
