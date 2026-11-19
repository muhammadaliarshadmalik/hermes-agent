"""Artifact versioning for B2: content hashing and immutable candidate snapshots.

A candidate is a directory such as ``evolve/artifacts/candidate-001``. Candidates are
never edited in place: a child is created by copying its parent, edited while still
writable, then sealed (made read-only) and hashed. The parent is never touched.

The hash covers sorted relative paths and file contents. Line endings in text files are
normalized (CRLF becomes LF) so the same artifact hashes identically on Windows and POSIX,
regardless of git's autocrlf setting.

Standard library only.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path

HASH_PREFIX = "sha256:"
CANDIDATE_ID_PATTERN = re.compile(r"^candidate-(\d{3,})$")
TEXT_SUFFIXES = frozenset({".yaml", ".yml", ".md", ".txt", ".json"})

_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class ArtifactError(ValueError):
    """The artifact directory is missing, empty, or contains something not allowed."""


def is_valid_candidate_id(candidate_id: str) -> bool:
    return isinstance(candidate_id, str) and CANDIDATE_ID_PATTERN.match(candidate_id) is not None


def next_candidate_id(artifacts_dir: str | Path) -> str:
    """Return the next free id, one above the highest existing ``candidate-NNN``."""
    root = Path(artifacts_dir)
    numbers = []
    if root.is_dir():
        for entry in root.iterdir():
            match = CANDIDATE_ID_PATTERN.match(entry.name)
            if match and entry.is_dir():
                numbers.append(int(match.group(1)))
    return f"candidate-{(max(numbers) + 1) if numbers else 0:03d}"


def _list_files(root: Path) -> list[Path]:
    """Return all regular files under root, sorted by relative POSIX path.

    Symlinks anywhere in the tree are refused, so a candidate can never point outside itself.
    """
    if not root.is_dir():
        raise ArtifactError(f"not a directory: {root}")
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            full = Path(dirpath) / name
            if full.is_symlink():
                raise ArtifactError(f"symlinks are not allowed in an artifact: {full}")
        files.extend(Path(dirpath) / name for name in filenames)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def hash_directory(root: str | Path) -> str:
    """Return ``sha256:<hex>`` over the directory's files (paths and normalized contents)."""
    base = Path(root)
    files = _list_files(base)
    if not files:
        raise ArtifactError(f"artifact has no files: {base}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        data = path.read_bytes()
        if path.suffix.lower() in TEXT_SUFFIXES:
            data = data.replace(b"\r\n", b"\n")
        # Length prefixes keep path/content boundaries unambiguous.
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return HASH_PREFIX + digest.hexdigest()


def _chmod_tree(root: Path, writable: bool) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in filenames:
            _set_write(Path(dirpath) / name, writable)
        for name in dirnames:
            _set_write(Path(dirpath) / name, writable)
    _set_write(root, writable)


def _set_write(path: Path, writable: bool) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | stat.S_IWUSR if writable else mode & ~_WRITE_BITS)


def unseal_snapshot(candidate_dir: str | Path) -> None:
    """Make a candidate directory writable again (staging, cleanup)."""
    _chmod_tree(Path(candidate_dir), writable=True)


def is_sealed(candidate_dir: str | Path) -> bool:
    """True if no file in the candidate has a write permission bit set."""
    return all(not (p.stat().st_mode & _WRITE_BITS) for p in _list_files(Path(candidate_dir)))


def seal_snapshot(candidate_dir: str | Path) -> str:
    """Hash the candidate, make it read-only, and return the hash.

    Files become read-only everywhere. Directories also lose write permission on POSIX
    (Windows ignores directory read-only flags, so only files are protected there).
    """
    base = Path(candidate_dir)
    artifact_hash = hash_directory(base)
    for path in _list_files(base):
        _set_write(path, writable=False)
    if os.name != "nt":
        for dirpath, dirnames, _ in os.walk(base, topdown=False):
            for name in dirnames:
                _set_write(Path(dirpath) / name, writable=False)
        _set_write(base, writable=False)
    return artifact_hash


def verify_snapshot(candidate_dir: str | Path, expected_hash: str) -> bool:
    """True if the directory's current hash equals the recorded hash."""
    return hash_directory(candidate_dir) == expected_hash


def create_child_snapshot(
    artifacts_dir: str | Path,
    parent_id: str,
    child_id: str | None = None,
) -> Path:
    """Copy a parent candidate into a new, writable child directory and return its path.

    The parent is never modified, even if it is sealed. Edit the child, then call
    ``seal_snapshot`` on it.
    """
    root = Path(artifacts_dir)
    if not is_valid_candidate_id(parent_id):
        raise ValueError(f"invalid parent id: {parent_id!r}")
    parent = root / parent_id
    if not parent.is_dir():
        raise FileNotFoundError(f"parent candidate not found: {parent}")
    if child_id is None:
        child_id = next_candidate_id(root)
    if not is_valid_candidate_id(child_id):
        raise ValueError(f"invalid child id: {child_id!r}")
    child = root / child_id
    if child.exists():
        raise FileExistsError(f"candidate already exists: {child}")

    _list_files(parent)  # refuses symlinks before anything is copied
    try:
        shutil.copytree(parent, child, symlinks=False)
        _chmod_tree(child, writable=True)
    except Exception:
        if child.exists():
            _chmod_tree(child, writable=True)
            shutil.rmtree(child, ignore_errors=True)
        raise
    return child
def restore_snapshot(source_dir: str | Path, destination_dir: str | Path, expected_hash: str) -> Path:
    """Copy a recorded snapshot to a new destination and verify its content hash."""
    import shutil
    source, destination = Path(source_dir), Path(destination_dir)
    if destination.exists():
        raise FileExistsError(f"restore destination already exists: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(f"snapshot not found: {source}")
    actual = hash_directory(source)
    if actual != expected_hash:
        raise ArtifactError(f"snapshot hash mismatch: expected {expected_hash}, got {actual}")
    shutil.copytree(source, destination)
    _chmod_tree(destination, writable=True)
    if hash_directory(destination) != expected_hash:
        _chmod_tree(destination, writable=True)
        shutil.rmtree(destination, ignore_errors=True)
        raise ArtifactError("restored snapshot hash mismatch")
    return destination
