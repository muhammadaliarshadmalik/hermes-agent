from pathlib import Path
import pytest
from evolve.patching import apply_patch

def test_patch_creates_child_and_diff_without_changing_parent(tmp_path):
 root=tmp_path/'artifacts'; parent=root/'candidate-000'; parent.mkdir(parents=True); (parent/'policy.yaml').write_text('rules: []\n'); result=apply_patch(root,'candidate-000',{'policy.yaml':'rules: [R1]\n'},seal=False); assert (parent/'policy.yaml').read_text()=='rules: []\n'; assert 'R1' in (result['candidate_dir']/'policy.yaml').read_text(); assert Path(result['diff_ref']).exists()
def test_scope_failure_removes_child(tmp_path):
 root=tmp_path/'artifacts'; parent=root/'candidate-000'; parent.mkdir(parents=True); (parent/'policy.yaml').write_text('rules: []\n')
 with pytest.raises(ValueError): apply_patch(root,'candidate-000',{'../bad.txt':'x'})
 assert sorted(p.name for p in root.iterdir())==['candidate-000']
