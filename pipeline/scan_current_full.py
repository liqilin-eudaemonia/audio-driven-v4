"""Current listen-response rules on all local raw Seamless archives.

No vision, gain normalization, model inference or video rendering. Immutable
source inventory, metadata shards and result shards make each stage resumable.
"""
from __future__ import annotations
import argparse, concurrent.futures, gzip, hashlib, io, json, math, os, sqlite3, struct, tarfile, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/public/0_DATA/2_Avatar/seamless_interaction')
OUT=ROOT/'outputs_current_full_v1'
REV='full-listen-response-10s-raw-v1'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def wav_info(f,ref):
    base=ref['offset'];f.seek(base);head=f.read(12)
    if head[:4]!=b'RIFF' or head[8:]!=b'WAVE':raise ValueError('Unsupported WAV container')
    pos=12;fmt=None;data=None
    while pos+8<=ref['size']:
        f.seek(base+pos);tag,size=struct.unpack('<4sI',f.read(8));payload=pos+8
        if tag==b'fmt ':
            raw=f.read(size);kind,ch,sr,_,align,bits=struct.unpack('<HHIIHH',raw[:16])
            if kind==65534 and len(raw)>=40:kind=struct.unpack('<H',raw[24:26])[0]
            fmt=dict(format=kind,channels=ch,samplerate=sr,block_align=align,bits=bits)
        if tag==b'data':data=dict(data_offset=base+payload,data_bytes=min(size,ref['size']-payload))
        if fmt and data:break
        pos=payload+size+(size%2)
    if not fmt or not data:raise ValueError('WAV fmt/data missing')
    frames=data['data_bytes']//fmt['block_align']
    return dict(**fmt,**data,frames=frames,duration=frames/fmt['samplerate'])

def video_duration(f,ref):
    base=ref['offset'];stop=base+ref['size'];pos=base
    def boxes(a,b):
        while a+8<=b:
            f.seek(a);size,kind=struct.unpack('>I4s',f.read(8));header=8
            if size==1:size=struct.unpack('>Q',f.read(8))[0];header=16
            elif size==0:size=b-a
            if size<header or a+size>b:raise ValueError('Invalid MP4 box')
            yield a+header,a+size,kind
            a+=size
    for a,b,kind in boxes(pos,stop):
        if kind!=b'moov':continue
        for c,d,child in boxes(a,b):
            if child!=b'mvhd':continue
            f.seek(c);data=f.read(min(d-c,40))
            if data[0]==0:scale,duration=struct.unpack('>II',data[12:20])
            elif data[0]==1:scale,duration=struct.unpack('>IQ',data[20:32])
            else:raise ValueError('Unknown mvhd version')
            if scale<=0:raise ValueError('Bad MP4 timescale')
            return duration/scale
    raise ValueError('MP4 mvhd missing')

def parse_archive(item):
    from scan_alternate_pairs import parse_words
    path=Path(item['path']);records={};errors=[]
    try:
        with path.open('rb') as f:
            with tarfile.open(fileobj=f,mode='r:') as tar:
                for m in tar:
                    if not m.isfile():continue
                    name=Path(m.name).name
                    if '.' not in name:continue
                    stem,ext=name.rsplit('.',1)
                    if ext not in ['json','wav','mp4']:continue
                    parts=stem.split('_')
                    if len(parts)!=4:continue
                    r=records.setdefault(stem,dict(id=stem,interaction='_'.join(parts[:3]),participant=parts[3],
                        label=item['label'],split=item['split'],refs={}))
                    r['refs'][ext]=dict(archive=str(path),member=m.name,offset=m.offset_data,size=m.size)
            for r in records.values():
                r['errors']=[]
                if not all(e in r['refs'] for e in ['json','wav','mp4']):
                    r['errors'].append('missing_json_wav_or_mp4');continue
                try:
                    ref=r['refs']['json'];f.seek(ref['offset']);meta=json.loads(f.read(ref['size']))
                    words=parse_words(meta)
                    r['invalid_word_times']=sum(not(math.isfinite(a) and math.isfinite(b) and b>a) for a,b,_ in words)
                    r['words']=[w for w in words if math.isfinite(w[0]) and math.isfinite(w[1]) and w[1]>w[0]]
                    r['wav']=wav_info(f,r['refs']['wav']);r['video_duration']=video_duration(f,r['refs']['mp4'])
                    if r['wav']['format'] not in [1,3] or r['wav']['bits'] not in [8,16,24,32,64]:r['errors'].append('unsupported_wav_encoding')
                except Exception as e:r['errors'].append(type(e).__name__+': '+str(e))
        stat=path.stat()
        if stat.st_size!=item['bytes'] or stat.st_mtime_ns!=item['mtime_ns']:errors.append('source_changed_during_scan')
    except Exception as e:errors.append(type(e).__name__+': '+str(e))
    return dict(archive=str(path),records=list(records.values()),errors=errors)

def metadata_batch(task):
    number,items=task;path=OUT/'metadata_shards'/f'{number:05d}.jsonl.gz';done=path.with_suffix('.done.json')
    if done.exists() and path.exists():return json.loads(done.read_text())
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending')
    summary=dict(batch=number,archives=len(items),people=0,archive_errors=0,person_errors=0)
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:
        for item in items:
            obj=parse_archive(item);summary['people']+=len(obj['records']);summary['archive_errors']+=bool(obj['errors'])
            summary['person_errors']+=sum(bool(r['errors']) for r in obj['records'])
            f.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
    tmp.replace(path);save(done,summary);return summary

def inventory():
    path=OUT/'source_inventory.json'
    if path.exists():return json.loads(path.read_text())
    items=[]
    for p in sorted(SOURCE.glob('*/*/[0-9][0-9][0-9][0-9]/*.tar')):
        stat=p.stat();rel=p.relative_to(SOURCE).parts
        items.append(dict(path=str(p),label=rel[0],split=rel[1],bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    obj=dict(revision=REV,source=str(SOURCE),archives=items,count=len(items),
        by_subset=dict(Counter(x['label']+'/'+x['split'] for x in items)),created_unix=time.time())
    save(path,obj);return obj

def scan_metadata(workers):
    inv=inventory();items=inv['archives'];tasks=[(i//128,items[i:i+128]) for i in range(0,len(items),128)]
    total=Counter();started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for result in ex.map(metadata_batch,tasks):
            total.update({k:v for k,v in result.items() if k!='batch'})
            progress=dict(stage='metadata',done_archives=total['archives'],total_archives=len(items),counts=dict(total),elapsed_s=time.time()-started)
            save(OUT/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    save(OUT/'metadata_summary.json',dict(total))

def build_database():
    path=OUT/'metadata.sqlite';marker=OUT/'database_summary.json'
    if marker.exists() and path.exists():return
    tmp=OUT/'metadata.sqlite.pending'
    if tmp.exists():tmp.unlink()
    db=sqlite3.connect(tmp);db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
    db.execute('CREATE TABLE people (key TEXT PRIMARY KEY, label TEXT, split TEXT, interaction TEXT, participant TEXT, payload BLOB)')
    counts=Counter();problems=OUT/'source_issues.jsonl'
    with problems.open('w') as issues:
        for path_shard in sorted((OUT/'metadata_shards').glob('*.jsonl.gz')):
            with gzip.open(path_shard,'rt',encoding='utf-8') as f:
                for line in f:
                    obj=json.loads(line)
                    if obj['errors']:
                        counts['bad_archives']+=1;issues.write(json.dumps(dict(archive=obj['archive'],errors=obj['errors']))+'\n');continue
                    for r in obj['records']:
                        if r['errors']:
                            counts['bad_people']+=1;issues.write(json.dumps(dict(id=r['id'],archive=obj['archive'],errors=r['errors']))+'\n');continue
                        key='/'.join([r['label'],r['split'],r['id']]);payload=gzip.compress(json.dumps(r,separators=(',',':')).encode(),compresslevel=1)
                        try:db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(key,r['label'],r['split'],r['interaction'],r['participant'],payload));counts['people']+=1
                        except sqlite3.IntegrityError:counts['duplicate_people']+=1;issues.write(json.dumps(dict(duplicate_key=key,archive=obj['archive']))+'\n')
            db.commit()
    db.execute('CREATE INDEX interaction_idx ON people(label,split,interaction)')
    groups=db.execute('SELECT label,split,interaction,count(*),group_concat(key, "|") FROM people GROUP BY label,split,interaction ORDER BY label,split,interaction').fetchall()
    pairs=[]
    for label,split,interaction,n,keys in groups:
        if n==2:pairs.append(dict(label=label,split=split,interaction=interaction,keys=sorted(keys.split('|'))))
        else:counts['unpaired_groups' if n<2 else 'ambiguous_groups']+=1
    db.close();tmp.replace(path);save(OUT/'conversations.json',pairs)
    save(marker,dict(**counts,conversations=len(pairs),by_subset=dict(Counter(p['label']+'/'+p['split'] for p in pairs))))
    print('DATABASE',dict(counts),'conversations',len(pairs),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['metadata','database']);ap.add_argument('--workers',type=int,default=24);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='metadata':scan_metadata(args.workers)
    if args.stage=='database':build_database()

if __name__=='__main__':main()
