import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.runner.executor import BundleError, hash_directory, load_bundle
from bench.scorer.report import FIELDS, validate_score_report
from bench.tasks.catalog import HELDOUT_ENV

ROOT = Path(__file__).resolve().parents[2]


def _bundle(d: Path, with_skills_dir=False, policy="# p\n", policy_file="policy.md"):
    d.mkdir(parents=True, exist_ok=True)
    meta = {"candidate_id": "candidate-t", "parent_id": None, "description": "t", "policy_file": policy_file}
    if with_skills_dir:
        meta["skills_dir"] = "skills"
    (d / "bundle.json").write_text(json.dumps(meta), encoding="utf-8", newline="")
    (d / "policy.md").write_text(policy, encoding="utf-8", newline="")
    return d


def _cli(*args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != HELDOUT_ENV}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "bench", *args], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=300)


def test_bundle_policy_only_and_skills_dir_ignored(tmp_path):
    b = load_bundle(_bundle(tmp_path / "a"))
    assert b.candidate_id == "candidate-t" and b.policy_text == "# p\n"
    b2 = load_bundle(_bundle(tmp_path / "b", with_skills_dir=True))
    assert b2.policy_sha256 == b.policy_sha256


def test_bundle_rejects_bad(tmp_path):
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "missing")
    _bundle(tmp_path / "t", policy_file="../evil.md")
    (tmp_path / "evil.md").write_text("x")
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "t")
    with pytest.raises(BundleError):
        load_bundle(_bundle(tmp_path / "big", policy="x" * (64 * 1024 + 1)))


def test_mock_train_run_outputs_are_schema_valid(tmp_path):
    cand, out = _bundle(tmp_path / "c"), tmp_path / "out"
    p = _cli("run", "--candidate", str(cand), "--split", "train", "--repeat", "3", "--mock", "--out", str(out))
    assert p.returncode == 0, p.stderr
    reports = sorted((out / "reports").glob("*.json"))
    assert len(reports) == 36  # 12 tasks x 3 repeats
    for rp in reports:
        r = json.loads(rp.read_text(encoding="utf-8"))
        validate_score_report(r)
        assert set(r) == set(FIELDS) and r["split"] == "train" and r["candidate_id"] == "candidate-t"
    assert len(list((out / "trajectories").glob("*.jsonl"))) == 36
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["repeat"] == 3 and s["mock"] is True and s["split"] == "train"
    assert s["candidate_hash"].startswith("sha256:") and len(s["candidate_hash"]) == 7 + 64
    assert s["aggregate"]["repeats"] == 3 and len(s["task_ids"]) == 12


def test_mock_is_deterministic(tmp_path):
    cand = _bundle(tmp_path / "c")
    rates = []
    for i in range(2):
        out = tmp_path / f"o{i}"
        assert _cli("run", "--candidate", str(cand), "--split", "train", "--mock", "--out", str(out)).returncode == 0
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        rates.append(s["aggregate"]["suite"]["pass_rate_per_repeat"])
    assert rates[0] == rates[1]


def test_heldout_blocked_without_env(tmp_path):
    p = _cli("run", "--candidate", str(_bundle(tmp_path / "c")), "--split", "heldout", "--mock", "--out", str(tmp_path / "o"))
    assert p.returncode == 2 and "sealed" in p.stderr
    assert not (tmp_path / "o").exists()


def test_heldout_blocked_when_env_not_1(tmp_path):
    p = _cli("run", "--candidate", str(_bundle(tmp_path / "c")), "--split", "heldout", "--mock",
             "--out", str(tmp_path / "o"), env_extra={HELDOUT_ENV: "true"})
    assert p.returncode == 2


def test_heldout_allowed_with_env(tmp_path):
    out = tmp_path / "o"
    p = _cli("run", "--candidate", str(_bundle(tmp_path / "c")), "--split", "heldout", "--repeat", "1",
             "--mock", "--out", str(out), env_extra={HELDOUT_ENV: "1"})
    assert p.returncode == 0, p.stderr
    assert len(list((out / "reports").glob("*.json"))) == 4


def test_live_requires_model(tmp_path):
    p = _cli("run", "--candidate", str(_bundle(tmp_path / "c")), "--split", "train", "--out", str(tmp_path / "o"))
    assert p.returncode == 2 and "--model" in p.stderr


def test_bad_candidate_exit_1(tmp_path):
    p = _cli("run", "--candidate", str(tmp_path / "nope"), "--split", "train", "--mock", "--out", str(tmp_path / "o"))
    assert p.returncode == 1 and "invalid candidate bundle" in p.stderr


def test_real_candidate_000_loads():
    b = load_bundle(ROOT / "evolve" / "artifacts" / "candidate-000")
    assert b.candidate_id == "candidate-000" and b.policy_text.strip()
    assert hash_directory(ROOT / "evolve" / "artifacts" / "candidate-000").startswith("sha256:")


def test_candidate_hash_prefix_and_stability(tmp_path):
    b = _bundle(tmp_path / "c")
    h1 = hash_directory(b)
    h2 = hash_directory(b)
    assert h1 == h2 and h1.startswith("sha256:") and len(h1) == 7 + 64


def test_candidate_hash_crlf_equivalent(tmp_path):
    a = _bundle(tmp_path / "a", policy="# p\n")
    b = _bundle(tmp_path / "b", policy="# p\r\n")
    assert hash_directory(a) == hash_directory(b)


def test_candidate_hash_rejects_symlink(tmp_path):
    b = _bundle(tmp_path / "c")
    target = tmp_path / "outside.txt"
    target.write_text("x", encoding="utf-8")
    link = b / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this host")
    with pytest.raises(BundleError):
        hash_directory(b)


def test_candidate_hash_rejects_empty_dir(tmp_path):
    b = _bundle(tmp_path / "c")
    (b / "empty").mkdir()
    with pytest.raises(BundleError):
        hash_directory(b)


def test_live_agent_sets_and_restores_hermes_home(monkeypatch, tmp_path):
    from bench.runner.executor import Bundle, _live_agent
    from bench.tasks.catalog import get_task
    from bench.trajectories.capture import TrajectoryWriter

    seen = {}

    class FakeAgent:
        def __init__(self, **kw):
            seen["HERMES_HOME"] = os.environ.get("HERMES_HOME")
            self.session_total_tokens = 11

        def run_conversation(self, prompt):
            return {"messages": [], "completed": True, "api_calls": 2, "final_response": "ok"}

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent, raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    task = get_task("train-001-fix-off-by-one")
    bundle = Bundle("candidate-t", None, "# p\n", "deadbeef", tmp_path)
    traj = TrajectoryWriter(tmp_path / "t.jsonl")
    try:
        usage = _live_agent(task, bundle, tmp_path, model="m", provider=None,
                            timeout=5.0, max_steps=2, traj=traj)
    finally:
        traj.close()
    assert usage["tokens"] == 11 and usage["steps"] == 2
    assert seen["HERMES_HOME"] and "benchhome_" in seen["HERMES_HOME"]
    assert "HERMES_HOME" not in os.environ