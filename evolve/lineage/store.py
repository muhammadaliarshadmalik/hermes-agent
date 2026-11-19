"""Append-only candidate lineage JSONL storage."""
import json
from datetime import datetime, timezone, UTC
from pathlib import Path
from evolve.versioning import is_valid_candidate_id

_REQUIRED = {"candidate_id", "parent_id", "candidate_hash", "status", "event_type", "timestamp"}

def validate_record(record):
    if not isinstance(record, dict) or not _REQUIRED <= record.keys():
        raise ValueError(f"record requires fields: {sorted(_REQUIRED)}")
    if not is_valid_candidate_id(record["candidate_id"]):
        raise ValueError("invalid candidate_id")
    parent = record["parent_id"]
    if parent is not None and not is_valid_candidate_id(parent):
        raise ValueError("invalid parent_id")
    if record["status"] not in {"created", "accepted", "rejected"}:
        raise ValueError("status must be created, accepted, or rejected")
    if not all(isinstance(record[k], str) and record[k] for k in ("candidate_hash", "event_type", "timestamp")):
        raise ValueError("hash, event_type, and timestamp must be non-empty strings")
    if "diff_ref" in record and not isinstance(record["diff_ref"], str):
        raise ValueError("diff_ref must be a string")
    if record["status"] == "rejected" and not record.get("rejection_reason"):
        raise ValueError("rejected records require rejection_reason")
    return record

def append_record(path, record):
    target=Path(path); validate_record(record); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
    return record

def read_records(path):
    target=Path(path)
    if not target.exists(): return []
    records=[]
    with target.open(encoding="utf-8") as f:
        for number,line in enumerate(f,1):
            try: records.append(validate_record(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc: raise ValueError(f"malformed lineage at {target}:{number}: {exc}") from exc
    return records

def new_record(candidate_id, parent_id, candidate_hash, status, event_type, **extra):
    return validate_record(dict(candidate_id=candidate_id,parent_id=parent_id,candidate_hash=candidate_hash,status=status,event_type=event_type,timestamp=datetime.now(UTC).isoformat(),**extra))
