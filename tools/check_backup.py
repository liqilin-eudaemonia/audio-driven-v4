"""Verify frozen hashes/import closure and replay synthetic checks in isolation."""
import ast,contextlib,hashlib,io,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PIPELINE=ROOT/'pipeline'
sys.path.insert(0,str(PIPELINE))

def main():
 manifest=json.loads((ROOT/'SOURCE_MAP.json').read_text())
 for item in manifest['files']:
  path=ROOT/item['backup'];assert path.is_file() and not path.is_symlink()
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['backup']
 for name,deps in manifest['local_imports'].items():
  for dep in deps:assert (PIPELINE/dep).is_file(),(name,dep)
  compile((PIPELINE/name).read_text(),str(PIPELINE/name),'exec')
 # Verify production algorithm dependencies match the fingerprints in full-run reports.
 for report in ['v2_rule_fingerprint.json','v3_rule_fingerprint.json']:
  for name,h in json.loads((ROOT/'reports'/report).read_text())['sha256'].items():
   assert hashlib.sha256((PIPELINE/name).read_bytes()).hexdigest()==h,(report,name)
 import apply_sentence_windows_v2 as v2
 import repair_sentence_starts_v3 as v3
 import validate_sentence_windows_v2 as v2_checks
 import test_sentence_start_repair_v3 as v3_checks
 import test_listener_role_switch as role_checks
 with tempfile.TemporaryDirectory(prefix='seamless_backup_checks_') as temp:
  # Unit checks normally write a small result JSON. Redirect it away from data.
  v2.OUT=Path(temp)/'v2';v3.OUT=Path(temp)/'v3'
  with contextlib.redirect_stdout(io.StringIO()):
   v2_checks.unit_checks();v3_checks.check()
  checks2=json.loads((v2.OUT/'unit_validation.json').read_text())
  checks3=json.loads((v3.OUT/'unit_validation.json').read_text())
  roles=role_checks.run_checks()
 assert checks2['passed'] and checks3['passed'] and roles['passed']
 # No raw media, data manifests, credentials or model files in this snapshot.
 tracked=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
 banned={'.mp4','.wav','.npz','.sqlite','.jsonl','.safetensors','.pt','.pem','.key'}
 assert not any(p.suffix in banned or p.name=='.env' for p in tracked)
 result=dict(passed=True,source_hash_entries=len(manifest['files']),python_dependency_files=len(manifest['local_imports']),
  v2_checks=checks2['checks'],v3_checks=checks3['checks'],role_switch_checks=roles['checks'],
  production_rule_fingerprints_match=True,synthetic_tests_only_no_dataset_read=True,raw_media_or_model_files_included=False)
 (ROOT/'reports/backup_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
