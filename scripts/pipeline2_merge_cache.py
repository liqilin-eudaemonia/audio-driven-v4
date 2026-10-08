#!/usr/bin/env python3
"""Merge completed preparation shards and verify exact requested sample coverage."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omni_av.data import write_json_new


def merge(indexes, manifest):
    sources = [json.loads(p.read_text()) for p in indexes]
    expected = json.loads(manifest.read_text())
    splits = {x['split'] for x in expected}
    if len(splits) != 1 or not sources:
        raise ValueError('Need one split and nonempty shards')
    split = splits.pop()
    entries = []
    for source in sources:
        if source['contract'] != sources[0]['contract'] or source['split'] != split:
            raise ValueError('Incompatible preparation shards')
        entries.extend(source['entries'])
    ids = [e['sample_id'] for e in entries]
    wanted = [e['observation']['sample_id'] for e in expected]
    if len(set(ids)) != len(ids) or len(set(wanted)) != len(wanted) or set(ids) != set(wanted):
        raise ValueError('Duplicate or missing sample IDs across shards')
    for entry in entries:
        if entry['split'] != split or not all(Path(entry[k]).is_file() for k in ('features', 'text')):
            raise ValueError('Missing assets or wrong split')
    by_id = {e['sample_id']: e for e in entries}
    return dict(contract=sources[0]['contract'], split=split, entries=[by_id[sid] for sid in wanted])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indexes', type=Path, nargs='+', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    index = merge(args.indexes, args.manifest)
    write_json_new(args.output, index)
    print(json.dumps(dict(split=index['split'], samples=len(index['entries']))))


if __name__ == '__main__':
    main()
