from pathlib import Path
import json
import pytest
from evolve.lineage.store import append_record, read_records

BASE=dict(candidate_id='candidate-001',parent_id='candidate-000',candidate_hash='sha256:abc',status='created',event_type='created',timestamp='2026-01-01T00:00:00Z')
def test_append_and_read_multiple_preserves_history(tmp_path):
 p=tmp_path/'state'/'lineage.jsonl'; append_record(p,BASE); append_record(p,{**BASE,'candidate_id':'candidate-002','status':'rejected','rejection_reason':'regression'}); assert len(read_records(p))==2

def test_rejects_invalid_record(tmp_path):
 with pytest.raises(ValueError): append_record(tmp_path/'x.jsonl',{**BASE,'candidate_id':'../escape'})

def test_malformed_existing_jsonl_is_reported(tmp_path):
 p=tmp_path/'lineage.jsonl'; p.write_text('{bad}\n');
 with pytest.raises(ValueError,match='lineage'): read_records(p)
