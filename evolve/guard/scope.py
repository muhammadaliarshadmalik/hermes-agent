"""Validate candidate file paths and proposed file contents before writes."""
from pathlib import Path

class ScopeViolation(ValueError): pass

def validate_modification(candidate_dir, relative_path, content, *, allowed_extensions=(".yaml",), max_bytes=8192):
    root=Path(candidate_dir).resolve(); rel=Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts: raise ScopeViolation("path traversal or absolute path is not allowed")
    target=root/rel
    current=root
    for part in rel.parts:
        current=current/part
        if current.is_symlink(): raise ScopeViolation(f"symlink is not allowed: {rel}")
    if not target.resolve().is_relative_to(root): raise ScopeViolation("path resolves outside candidate directory")
    if target.suffix.lower() not in allowed_extensions: raise ScopeViolation(f"disallowed file type: {target.suffix}")
    data=content.encode("utf-8") if isinstance(content,str) else content
    if not isinstance(data,bytes): raise ScopeViolation("content must be text or bytes")
    if len(data)>max_bytes: raise ScopeViolation(f"file exceeds maximum size ({max_bytes} bytes)")
    return target
