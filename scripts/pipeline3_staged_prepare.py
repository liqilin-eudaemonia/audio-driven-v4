"""CPU-only full-manifest preparation. Does not extract or launch training."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from omni_av_p3_staged.data import read, join_rows
from omni_av_p3.distributed import sha
from omni_av_p2.checkpoint import atomic_json_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); cfg=read(a.config)
    native=read(cfg['native_index']); splits={}; inputs={}
    for split,count in [('train',91009),('dev',48)]:
        example_path=ROOT/cfg['manifests'][split]
        metadata_path=Path(cfg['data_root'])/'ltx23_precomputed_720p_area_qualified_20260920'/f'metadata_{split}.jsonl'
        text_path=Path(cfg['text_indices'][split])
        metadata=[json.loads(x) for x in metadata_path.read_text().splitlines() if x.strip()]
        splits[split]=join_rows(native[split],read(text_path)['entries'],read(example_path),metadata,split,count)
        for path in (example_path,metadata_path,text_path):inputs[str(path)]=sha(path)
    if {e['sample_id'] for e in splits['train']} & {e['sample_id'] for e in splits['dev']}:
        raise ValueError('Train/dev overlap')
    cp=cfg['planner']
    if sha(cp['path']) != cp['sha256'] or sha(cfg['initial_connector']['path']) != cfg['initial_connector']['sha256']:
        raise ValueError('Selected initialization changed')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json_new(a.output,dict(version='p3.full-render-manifest.v1',config=cfg,
        config_sha256=sha(a.config),native_contract=native['contract'],
        native_index_sha256=sha(cfg['native_index']),source_indices=inputs,**splits))
    print(json.dumps(dict(manifest=str(a.output),train=len(splits['train']),dev=len(splits['dev']),
                         sha256=sha(a.output),gpu_started=False)))
if __name__=='__main__':main()
