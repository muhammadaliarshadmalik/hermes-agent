"""Tests for the baseline policy artifact (B2, week 2)."""

import re
from pathlib import Path

from evolve.versioning import hash_directory, is_valid_candidate_id

CANDIDATE = Path(__file__).resolve().parents[2] / "evolve" / "artifacts" / "candidate-000"


def test_candidate_000_is_policy_yaml_baseline():
    assert CANDIDATE.is_dir()
    assert is_valid_candidate_id(CANDIDATE.name)
    assert sorted(p.name for p in CANDIDATE.rglob("*") if p.is_file()) == ["policy.yaml"]
    text = (CANDIDATE / "policy.yaml").read_text(encoding="utf-8-sig")
    assert "schema_version: 1" in text
    assert "rules: []" in text
    assert len((CANDIDATE / "policy.yaml").read_bytes()) <= 8 * 1024


def test_baseline_hash_is_stable_across_crlf(tmp_path):
    expected = hash_directory(CANDIDATE)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", expected)
    copy = tmp_path / "candidate-000"
    copy.mkdir()
    raw = (CANDIDATE / "policy.yaml").read_text(encoding="utf-8-sig")
    (copy / "policy.yaml").write_bytes(raw.replace("\n", "\r\n").encode("utf-8"))
    assert hash_directory(copy) == expected
