"""Create child candidates from validated file-content patches."""
import difflib
from pathlib import Path
from evolve.versioning import create_child_snapshot, hash_directory, seal_snapshot
from evolve.guard.scope import validate_modification

def apply_patch(artifacts_dir, parent_id, changes, *, allowed_extensions=(".yaml",), max_bytes=8192, seal=True):
    root=Path(artifacts_dir); child=create_child_snapshot(root,parent_id)
    before={p.relative_to(child).as_posix():p.read_text(encoding="utf-8") for p in child.rglob("*") if p.is_file() and p.suffix.lower() in allowed_extensions}
    try:
        for name,content in changes.items():
            target=validate_modification(child,name,content,allowed_extensions=allowed_extensions,max_bytes=max_bytes)
            target.parent.mkdir(parents=True,exist_ok=True); target.write_text(content,encoding="utf-8",newline="")
        after={p.relative_to(child).as_posix():p.read_text(encoding="utf-8") for p in child.rglob("*") if p.is_file() and p.suffix.lower() in allowed_extensions}
        diff_parts = []
        for k in sorted(before.keys() | after.keys()):
            if before.get(k) != after.get(k):
                diff_parts.extend(difflib.unified_diff(before.get(k, "").splitlines(True), after.get(k, "").splitlines(True), fromfile=f"a/{k}", tofile=f"b/{k}"))
        diff = "".join(diff_parts)
        diff_path=root.parent/"diffs"/f"{child.name}.diff"; diff_path.parent.mkdir(parents=True,exist_ok=True); diff_path.write_text(diff,encoding="utf-8")
        digest=seal_snapshot(child) if seal else hash_directory(child)
        return {"candidate_dir":child,"candidate_hash":digest,"diff_ref":diff_path.as_posix()}
    except Exception:
        from evolve.versioning import unseal_snapshot
        unseal_snapshot(child)
        import shutil; shutil.rmtree(child)
        raise
