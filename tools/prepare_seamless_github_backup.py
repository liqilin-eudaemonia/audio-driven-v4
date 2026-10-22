"""Build a source-only, allowlisted backup, preserving frozen implementation hashes."""
import ast,concurrent.futures,hashlib,importlib.metadata,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[2]/'code_backups/seamless-listen-response-v3-20261022'
ENTRY=['scan_current_full.py','apply_current_full_rules.py','apply_sentence_windows_v2.py','repair_sentence_starts_v3.py',
 'validate_sentence_windows_v2.py','validate_sentence_start_repair_v3.py','test_sentence_start_repair_v3.py','test_listener_role_switch.py',
 'build_sentence_review_preview.py','validate_sentence_review_preview.py','check_sentence_review_browser.py',
 'publish_sentence_start_repair_v3.py','check_sentence_start_repair_v3.py']
REPORTS={
 'outputs_current_full_v1/database_summary.json':'reports/source_database_summary.json',
 'outputs_current_full_v1/environment.json':'reports/source_environment.json',
 'outputs_sentence_windows_v2/summary.json':'reports/v2_summary.json',
 'outputs_sentence_windows_v2/validation.json':'reports/v2_full_validation.json',
 'outputs_sentence_windows_v2/rule_fingerprint.json':'reports/v2_rule_fingerprint.json',
 'outputs_sentence_windows_v2/review_preview_v1/selection_validation.json':'reports/v2_sample_selection_validation.json',
 'outputs_sentence_windows_v2/review_preview_v1/browser_validation.json':'reports/v2_browser_validation.json',
 'outputs_sentence_start_repair_v3/summary.json':'reports/v3_summary.json',
 'outputs_sentence_start_repair_v3/validation.json':'reports/v3_full_validation.json',
 'outputs_sentence_start_repair_v3/unit_validation.json':'reports/v3_unit_validation.json',
 'outputs_sentence_start_repair_v3/audit_summary.json':'reports/v3_audit_summary.json',
 'outputs_sentence_start_repair_v3/apply_summary.json':'reports/v3_apply_summary.json',
 'outputs_sentence_start_repair_v3/rule_fingerprint.json':'reports/v3_rule_fingerprint.json',
 'outputs_sentence_start_repair_v3/preview/browser_validation.json':'reports/v3_browser_validation.json',
 'outputs_sentence_start_repair_v3/preview/media_validation.json':'reports/v3_sample_media_validation.json'}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(4*1024*1024):h.update(block)
 return h.hexdigest()
def closure():
 seen=set();todo=list(ENTRY);edges={}
 while todo:
  name=todo.pop()
  if name in seen:continue
  seen.add(name);deps=set()
  for node in ast.walk(ast.parse((ROOT/name).read_text())):
   mods=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
   for mod in mods:
    dep=mod.split('.')[0]+'.py'
    if (ROOT/dep).is_file():deps.add(dep);todo.append(dep)
  edges[name]=sorted(deps)
 return seen,edges

def main():
 OUT.mkdir(parents=True,exist_ok=True);files,edges=closure();mapping={name:'pipeline/'+name for name in files}
 mapping.update(REPORTS);mapping.update({'sentence_review_preview.html':'pipeline/sentence_review_preview.html','CURRENT_SEGMENTATION_RULES.md':'docs/CURRENT_SEGMENTATION_RULES.md','outputs_sentence_start_repair_v3/README.md':'docs/V3_RUN_RECORD.md','prepare_seamless_github_backup.py':'tools/prepare_seamless_github_backup.py'})
 manifest=[]
 for src,dst in sorted(mapping.items()):
  dest=OUT/dst;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dest)
  manifest.append(dict(source=str(ROOT/src),backup=dst,sha256=digest(dest),size=dest.stat().st_size))
 (OUT/'SOURCE_MAP.json').write_text(json.dumps(dict(snapshot='seamless-listen-response-v3-20261022',files=manifest,local_imports=edges),ensure_ascii=False,indent=2)+'\n')
 (OUT/'.gitignore').write_text('__pycache__/\n*.pyc\noutputs*/\n*.sqlite\n*.sqlite-*\n*.mp4\n*.wav\n*.npz\n*.tar\n*.tar.gz\n*.jsonl\n*.jsonl.gz\n*.log\n*.bundle\n.env\n.venv/\n')
 versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','opencv-python','soundfile','fonttools']}
 (OUT/'requirements.txt').write_text('# Versions used by the frozen source; Python 3.13.12 on Linux.\n'+'\n'.join(f'{k}=={val}' for k,val in versions.items())+'\n')
 (OUT/'requirements-preview.txt').write_text('-r requirements.txt\nplaywright\nbrotli\n')
 print('Source files:',len(files),'Copied entries:',len(mapping),'Root:',OUT,flush=True)
 # Record hashes of local dataset artifacts, not their contents.
 artifacts=[ROOT/'outputs_current_full_v1'/n for n in ['metadata.sqlite','conversations.json','source_inventory.json']]+[ROOT/'outputs_sentence_start_repair_v3'/n for n in ['kept.jsonl','clean.jsonl','review.jsonl','unresolved_sentence_start.jsonl']]
 def record(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:records=list(ex.map(record,artifacts))
 (OUT/'DATA_ARTIFACTS.json').write_text(json.dumps(dict(included_in_git=False,artifacts=records),indent=2)+'\n')
 print('Artifact hashes complete',flush=True)
if __name__=='__main__':main()
