"""Tests for evolve.versioning (B2, week 2): hashing, ids, snapshots, sealing.

Run from the repo root:
    python -m uv run pytest tests/evolve/test_versioning.py -v
"""

import os
import re
import shutil
from pathlib import Path

import pytest

from evolve.versioning import (
    ArtifactError,
    create_child_snapshot,
    hash_directory,
    is_sealed,
    is_valid_candidate_id,
    next_candidate_id,
    seal_snapshot,
    unseal_snapshot,
    verify_snapshot,
)


def write(root: Path, files: dict[str, bytes | str]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8", newline="")
        else:
            path.write_bytes(content)
    return root


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    yield root
    for entry in root.iterdir():  # sealed folders must be writable again for cleanup
        unseal_snapshot(entry)


def make_baseline(artifacts: Path, text: str = "schema_version: 1\nrules: []\n") -> Path:
    return write(artifacts / "candidate-000", {"policy.yaml": text})


# --- hash_directory ----------------------------------------------------------


def test_hash_has_the_expected_format(tmp_path):
    write(tmp_path / "a", {"policy.yaml": "x"})
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", hash_directory(tmp_path / "a"))


def test_hash_is_deterministic(tmp_path):
    write(tmp_path / "a", {"policy.yaml": "x", "notes/n.md": "y"})
    assert hash_directory(tmp_path / "a") == hash_directory(tmp_path / "a")


def test_same_content_in_a_different_location_gives_the_same_hash(tmp_path):
    write(tmp_path / "one", {"policy.yaml": "x", "notes/n.md": "y"})
    write(tmp_path / "two" / "deeper", {"policy.yaml": "x", "notes/n.md": "y"})
    assert hash_directory(tmp_path / "one") == hash_directory(tmp_path / "two" / "deeper")


def test_hash_does_not_depend_on_file_creation_order(tmp_path):
    write(tmp_path / "one", {"a.txt": "1", "b.txt": "2", "c.txt": "3"})
    write(tmp_path / "two", {"c.txt": "3", "a.txt": "1", "b.txt": "2"})
    assert hash_directory(tmp_path / "one") == hash_directory(tmp_path / "two")


def test_changing_one_byte_changes_the_hash(tmp_path):
    write(tmp_path / "one", {"policy.yaml": "rules: []"})
    write(tmp_path / "two", {"policy.yaml": "rules: [] "})
    assert hash_directory(tmp_path / "one") != hash_directory(tmp_path / "two")


def test_renaming_a_file_changes_the_hash(tmp_path):
    write(tmp_path / "one", {"a.txt": "same"})
    write(tmp_path / "two", {"b.txt": "same"})
    assert hash_directory(tmp_path / "one") != hash_directory(tmp_path / "two")


def test_adding_a_file_changes_the_hash(tmp_path):
    write(tmp_path / "one", {"a.txt": "same"})
    write(tmp_path / "two", {"a.txt": "same", "b.txt": ""})
    assert hash_directory(tmp_path / "one") != hash_directory(tmp_path / "two")


def test_path_and_content_boundaries_are_unambiguous(tmp_path):
    write(tmp_path / "one", {"ab": "c"})
    write(tmp_path / "two", {"a": "bc"})
    assert hash_directory(tmp_path / "one") != hash_directory(tmp_path / "two")


def test_crlf_and_lf_text_files_hash_the_same(tmp_path):
    write(tmp_path / "lf", {"policy.yaml": "schema_version: 1\nrules: []\n"})
    write(tmp_path / "crlf", {"policy.yaml": "schema_version: 1\r\nrules: []\r\n"})
    assert hash_directory(tmp_path / "lf") == hash_directory(tmp_path / "crlf")


def test_line_endings_still_matter_in_non_text_files(tmp_path):
    write(tmp_path / "lf", {"data.bin": b"a\nb"})
    write(tmp_path / "crlf", {"data.bin": b"a\r\nb"})
    assert hash_directory(tmp_path / "lf") != hash_directory(tmp_path / "crlf")


def test_empty_directory_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ArtifactError):
        hash_directory(tmp_path / "empty")


def test_missing_directory_is_rejected(tmp_path):
    with pytest.raises(ArtifactError):
        hash_directory(tmp_path / "nope")


def test_symlinks_are_refused(tmp_path):
    write(tmp_path / "outside", {"secret.txt": "s"})
    target = write(tmp_path / "cand", {"policy.yaml": "x"})
    try:
        os.symlink(tmp_path / "outside", target / "link")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this system")
    with pytest.raises(ArtifactError):
        hash_directory(target)


# --- candidate ids -----------------------------------------------------------


@pytest.mark.parametrize("good", ["candidate-000", "candidate-012", "candidate-1000"])
def test_valid_candidate_ids(good):
    assert is_valid_candidate_id(good)


@pytest.mark.parametrize(
    "bad", ["", "candidate-1", "candidate-0a1", "../candidate-001", "candidate-001/x", "cand-001"]
)
def test_invalid_candidate_ids(bad):
    assert not is_valid_candidate_id(bad)


def test_next_id_starts_at_zero_in_an_empty_folder(artifacts):
    assert next_candidate_id(artifacts) == "candidate-000"


def test_next_id_follows_the_highest_existing_number(artifacts):
    for name in ("candidate-000", "candidate-001", "candidate-005", "notes", "candidate-x"):
        (artifacts / name).mkdir()
    assert next_candidate_id(artifacts) == "candidate-006"


# --- create_child_snapshot ---------------------------------------------------


def test_child_is_a_faithful_copy_of_the_parent(artifacts):
    parent = make_baseline(artifacts)
    child = create_child_snapshot(artifacts, "candidate-000")
    assert child.name == "candidate-001"
    assert hash_directory(child) == hash_directory(parent)


def test_editing_the_child_never_changes_the_parent(artifacts):
    parent = make_baseline(artifacts)
    before = hash_directory(parent)
    child = create_child_snapshot(artifacts, "candidate-000")
    edited = "schema_version: 1\nrules:\n  - id: R001\n"
    (child / "policy.yaml").write_text(edited, encoding="utf-8")
    assert hash_directory(parent) == before
    assert hash_directory(child) != before


def test_child_of_a_sealed_parent_is_writable(artifacts):
    make_baseline(artifacts)
    seal_snapshot(artifacts / "candidate-000")
    child = create_child_snapshot(artifacts, "candidate-000")
    assert not is_sealed(child)
    (child / "policy.yaml").write_text("changed", encoding="utf-8")


def test_explicit_child_id_is_used(artifacts):
    make_baseline(artifacts)
    child = create_child_snapshot(artifacts, "candidate-000", "candidate-007")
    assert child.name == "candidate-007"


def test_existing_child_is_never_overwritten(artifacts):
    make_baseline(artifacts)
    create_child_snapshot(artifacts, "candidate-000", "candidate-001")
    with pytest.raises(FileExistsError):
        create_child_snapshot(artifacts, "candidate-000", "candidate-001")


def test_missing_parent_is_rejected(artifacts):
    with pytest.raises(FileNotFoundError):
        create_child_snapshot(artifacts, "candidate-000")


@pytest.mark.parametrize("bad", ["../escape", "candidate-1", "x/y"])
def test_bad_ids_are_rejected_before_anything_is_copied(artifacts, bad):
    make_baseline(artifacts)
    with pytest.raises(ValueError):
        create_child_snapshot(artifacts, "candidate-000", bad)
    with pytest.raises(ValueError):
        create_child_snapshot(artifacts, bad)
    assert sorted(p.name for p in artifacts.iterdir()) == ["candidate-000"]


# --- seal and verify ---------------------------------------------------------


def test_seal_returns_the_directory_hash(artifacts):
    parent = make_baseline(artifacts)
    expected = hash_directory(parent)
    assert seal_snapshot(parent) == expected


def test_sealed_snapshot_is_read_only(artifacts):
    parent = make_baseline(artifacts)
    assert not is_sealed(parent)
    seal_snapshot(parent)
    assert is_sealed(parent)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can write to read-only files")
    with pytest.raises(PermissionError):
        (parent / "policy.yaml").write_text("tampered", encoding="utf-8")


def test_sealing_does_not_change_the_hash(artifacts):
    parent = make_baseline(artifacts)
    before = hash_directory(parent)
    seal_snapshot(parent)
    assert hash_directory(parent) == before


def test_verify_detects_tampering(artifacts):
    parent = make_baseline(artifacts)
    recorded = seal_snapshot(parent)
    assert verify_snapshot(parent, recorded)
    unseal_snapshot(parent)
    (parent / "policy.yaml").write_text("tampered", encoding="utf-8")
    assert not verify_snapshot(parent, recorded)


def test_unseal_makes_a_snapshot_writable_again(artifacts):
    parent = make_baseline(artifacts)
    seal_snapshot(parent)
    unseal_snapshot(parent)
    assert not is_sealed(parent)


def test_a_chain_of_snapshots_keeps_every_recorded_hash(artifacts):
    parent = make_baseline(artifacts)
    recorded = {"candidate-000": seal_snapshot(parent)}
    for number in (1, 2, 3):
        child = create_child_snapshot(artifacts, f"candidate-{number - 1:03d}")
        edited = f"schema_version: 1\n# edit {number}\n"
        (child / "policy.yaml").write_text(edited, encoding="utf-8")
        recorded[child.name] = seal_snapshot(child)
    assert len(set(recorded.values())) == 4
    for name, expected in recorded.items():
        assert verify_snapshot(artifacts / name, expected)


def test_copying_a_sealed_tree_elsewhere_keeps_the_same_hash(artifacts, tmp_path):
    parent = make_baseline(artifacts)
    recorded = seal_snapshot(parent)
    copy_path = tmp_path / "restored"
    shutil.copytree(parent, copy_path)
    unseal_snapshot(copy_path)
    assert verify_snapshot(copy_path, recorded)
def test_restore_snapshot_verifies_copy_and_preserves_source(artifacts, tmp_path):
    from evolve.versioning import restore_snapshot
    source = make_baseline(artifacts)
    recorded = seal_snapshot(source)
    restored = restore_snapshot(source, tmp_path / "restored", recorded)
    assert verify_snapshot(restored, recorded)
    assert verify_snapshot(source, recorded)

def test_restore_rejects_collision_and_tampering(artifacts, tmp_path):
    from evolve.versioning import restore_snapshot
    source = make_baseline(artifacts)
    recorded = seal_snapshot(source)
    with pytest.raises(FileExistsError): restore_snapshot(source, source, recorded)
    unseal_snapshot(source)
    (source / "policy.yaml").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactError): restore_snapshot(source, tmp_path / "bad", recorded)
