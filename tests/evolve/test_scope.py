from pathlib import Path
import pytest
from evolve.guard.scope import ScopeViolation, validate_modification

def test_accepts_policy_yaml(tmp_path):
 assert validate_modification(tmp_path,'policy.yaml','rules: []').name=='policy.yaml'
@pytest.mark.parametrize('name',['../escape.yaml','C:/outside.yaml','policy.txt'])
def test_rejects_bad_path_or_extension(tmp_path,name):
 with pytest.raises(ScopeViolation): validate_modification(tmp_path,name,'x')
def test_rejects_large_file(tmp_path):
 with pytest.raises(ScopeViolation,match='maximum'): validate_modification(tmp_path,'policy.yaml','12345',max_bytes=4)
def test_rejects_symlink(tmp_path):
 outside=tmp_path/'outside'; outside.write_text('x'); link=tmp_path/'link';
 try: link.symlink_to(outside)
 except OSError: pytest.skip('symlink privilege unavailable')
 with pytest.raises(ScopeViolation,match='symlink'): validate_modification(tmp_path,'link/policy.yaml','x')
