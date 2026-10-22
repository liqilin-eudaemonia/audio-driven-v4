"""30 full-frame listen+response previews with proposed fixed head/shoulder ROIs.

No spatially cropped videos are produced. Existing landmarks supply all geometry.
"""
import argparse
import concurrent.futures
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs_alternate_pairs/plain_screen/head_shoulders_720p30'
BUDGET = 1280*720


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2));temp.replace(path)


def run(args):
    p=subprocess.run(args, capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-2000:])
    return p.stdout


def refs(archive, stem, extensions):
    wanted={stem+'.'+e for e in extensions}; found={}
    with tarfile.open(archive,'r:') as tar:
        for m in tar:
            name=Path(m.name).name
            if name in wanted:
                found[name.rsplit('.',1)[1]]=dict(archive=archive,member=m.name,offset=m.offset_data,size=m.size)
                if len(found)==len(wanted):break
    if len(found)!=len(wanted): raise ValueError('Missing files for '+stem)
    return found


def read(ref):
    with open(ref['archive'],'rb') as f:
        f.seek(ref['offset']);return f.read(ref['size'])


def uri(ref):
    return f"subfile,,start,{ref['offset']},end,{ref['offset']+ref['size']},,:{ref['archive']}"


def canvas(ratio):
    candidates=[]
    for w in range(384,1793,32):
        for h in range(384,1793,32):
            area=w*h
            if .94*BUDGET<=area<=BUDGET:
                score=4*abs(math.log((w/h)/ratio))+(1-area/BUDGET)
                candidates.append((score,w,h))
    _,w,h=min(candidates)
    return w,h


def geometry(k, width, height, sar=1.0):
    # Work in display coordinates. Some square-coded videos have 9:16 SAR,
    # so treating raw pixel X/Y units as equal would distort the person.
    k=k.copy();k[:,:,0]*=sar;width*=sar
    face=k[:,23:91]; good=(face[:,:,2]>.4).sum(axis=1)>=40
    shoulder=k[:,[5,6]]
    good &= (shoulder[:,:,2]>.4).all(axis=1)
    good &= np.isfinite(face).all(axis=(1,2)) & np.isfinite(shoulder).all(axis=(1,2))
    if good.mean()<.75: raise ValueError(f'Only {good.mean():.1%} frames have valid face and shoulders')
    f=face[good];s=shoulder[good];xs=np.where(f[:,:,2]>.4,f[:,:,0],np.nan);ys=np.where(f[:,:,2]>.4,f[:,:,1],np.nan)
    fl,fr=np.nanpercentile(xs,[2,98],axis=1);ft,fb=np.nanpercentile(ys,[2,98],axis=1)
    fw=fr-fl;fh=fb-ft;sw=np.linalg.norm(s[:,0,:2]-s[:,1,:2],axis=1)
    # A face landmark box excludes the scalp/hair and shoulder landmarks sit
    # inside the clothing contour. Estimate margins explicitly, then audit them.
    left=np.minimum(fl-.18*fw,s[:,:,0].min(axis=1)-.18*sw)
    right=np.maximum(fr+.18*fw,s[:,:,0].max(axis=1)+.18*sw)
    top=ft-.85*fh
    bottom=np.maximum(fb+.12*fh,s[:,:,1].max(axis=1)+.30*sw)
    raw=np.array([np.quantile(left,.005),np.quantile(top,.005),np.quantile(right,.995),np.quantile(bottom,.995)])
    clipped=np.clip(raw,[0,0,0,0],[width,height,width,height])
    rw,rh=clipped[2]-clipped[0],clipped[3]-clipped[1]
    if min(rw,rh)<100:raise ValueError('Implausibly small ROI')
    tw,th=canvas(rw/rh)
    unit=math.gcd(tw,th);uw,uh=tw//unit,th//unit
    scale=2*math.ceil(max(rw/uw,rh/uh)/2)
    cw,ch=uw*scale,uh*scale
    constrained=False
    if cw>width or ch>height:
        scale=2*math.floor(min(width/uw,height/uh)/2);cw,ch=uw*scale,uh*scale;constrained=True
    cx,cy=(clipped[:2]+clipped[2:])/2
    x=int(np.clip(cx-cw/2,0,width-cw))//2*2;y=int(np.clip(cy-ch/2,0,height-ch))//2*2
    if raw[2]>=width and 0<width-(x+cw)<2:
        x=width-cw  # Do not lose the last source column merely to align origin.
    pts=f[:,:,:2];valid=f[:,:,2]>.4
    inside=(pts[:,:,0]>=x)&(pts[:,:,0]<=x+cw)&(pts[:,:,1]>=y)&(pts[:,:,1]<=y+ch)
    in_source=(pts[:,:,0]>=0)&(pts[:,:,0]<=width)&(pts[:,:,1]>=0)&(pts[:,:,1]<=height)
    face_coverage=float(inside[valid].mean())
    shoulders_inside=((s[:,:,0]>=x)&(s[:,:,0]<=x+cw)&(s[:,:,1]>=y)&(s[:,:,1]<=y+ch)).all(axis=1)
    estimated_coverage=float(((left>=x)&(right<=x+cw)&(top>=y)&(bottom<=y+ch)).mean())
    return dict(x=round(x/sar,6),y=y,width=round(cw/sar,6),height=ch,
                display_width=cw,display_height=ch,source_pixel_aspect_ratio=sar,
                target_width=tw,target_height=th,
                target_pixels=tw*th,pixel_budget=BUDGET,budget_fraction=round(tw*th/BUDGET,4),
                resize_scale=round(tw/cw,3),aspect_ratio=round(cw/ch,4),
                valid_frame_fraction=round(float(good.mean()),4),
                face_point_coverage=round(face_coverage,4),shoulder_frame_coverage=round(float(shoulders_inside.mean()),4),
                visible_face_point_coverage=round(float(inside[valid&in_source].mean()),4),
                face_points_outside_source_fraction=round(float((~in_source)[valid].mean()),4),
                estimated_envelope_coverage=round(estimated_coverage,4),
                source_boundary_limited=bool(np.any(raw!=clipped) or constrained),
                median_face_height=round(float(np.median(fh)),1),
                face_height_fraction=round(float(np.median(fh)/ch),3),
                median_shoulder_width=round(float(np.median(sw)),1),
                policy='face 2–98 percentiles; scalp/hair +0.85 face height; shoulders +0.18 width each side; chest +0.30 shoulder width; temporal 0.5–99.5 percentiles; fixed full-clip box')


def prepare(row):
    folder=OUT/'media'/row['clip_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'geometry.json').exists():
        cached=json.loads((folder/'geometry.json').read_text())
        if cached.get('roi_revision')==6:return cached
        if cached.get('roi_revision')==5 and not cached['roi']['source_boundary_limited']:
            cached['roi_revision']=6;save(folder/'geometry.json',cached);return cached
    r=dict(row);source=refs(r['listener_archive'],r['listener_id'],['mp4','npz','wav'])
    info=json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',uri(source['mp4'])]))
    v=next(s for s in info['streams'] if s['codec_type']=='video');n,d=map(int,v['avg_frame_rate'].split('/'));fps=n/d
    with np.load(io.BytesIO(read(source['npz'])),allow_pickle=False) as z:
        allk=z['boxes_and_keypoints:keypoints']
        nn,dd=map(int,v['r_frame_rate'].split('/'));nominal=nn/dd
        # A few source videos have missing frames and therefore avg_fps < 30.
        # NPZ arrays retain the nominal timeline; use its rate when length agrees.
        feature_fps=nominal if abs(len(allk)/nominal-float(v['duration']))<=.1 else fps
        if abs(len(allk)/feature_fps-float(v['duration']))>.15:raise ValueError('Landmark/video length mismatch')
        k=allk[int(r['clip_start_s']*feature_fps):math.ceil(r['clip_end_s']*feature_fps)]
    sar_text=v.get('sample_aspect_ratio','1:1');sn,sd=map(int,sar_text.split(':'));sar=sn/sd
    box=geometry(k,int(v['width']),int(v['height']),sar)
    r.update(roi=box,source_refs=source,source_width=int(v['width']),source_height=int(v['height']),
             source_fps=fps,landmark_fps=feature_fps,roi_revision=6,
             source_sar=sar,display_width=int(v['width'])*sar,display_height=int(v['height']),
             source_video_frames=int(v['nb_frames']),landmark_frames=len(allk),
             preview=f"media/{r['clip_id']}/full_frame.mp4",poster=f"media/{r['clip_id']}/poster.jpg")
    # Keep the entire original frame in every review asset.
    preview_h=2*round(720*r['display_height']/r['display_width']/2)
    run(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-ss',str(r['clip_start_s']),'-i',uri(source['mp4']),
         '-frames:v','1','-vf',f'scale=720:{preview_h}:flags=lanczos,setsar=1',str(folder/'poster.jpg')])
    save(folder/'geometry.json',r)
    return r


def track_audio(ref, row, role, samples):
    with sf.SoundFile(io.BytesIO(read(ref))) as f:
        sr=f.samplerate;f.seek(round(row['clip_start_s']*sr));a=f.read(round(row['clip_duration_s']*sr),dtype='float32',always_2d=True).mean(axis=1)
    count=round(row['clip_duration_s']*sr);a=np.pad(a,(0,max(0,count-len(a))))[:count]
    gated=np.zeros_like(a)
    for start,end in row['keep_'+role+'_audio']:
        i=max(0,round((start-row['clip_start_s'])*sr));j=min(count,round((end-row['clip_start_s'])*sr))
        if j>i:gated[i:j]=a[i:j]
    fade=min(count,round(row['end_fade_s']*sr))
    if fade:gated[-fade:]*=np.linspace(1,0,fade,dtype=np.float32)
    gated*=row['audio_gain']
    samples[role]=(gated,sr)


def render(row):
    folder=OUT/'media'/row['clip_id']
    if (folder/'completed.json').exists():
        previous=json.loads((folder/'completed.json').read_text())
        reusable_sar=previous.get('preview_square_pixels',False) or row['source_sar']==1
        if reusable_sar and (previous['preview_audio_peak_before_clip']<=1 or 'preview_mix_attenuation' in previous):
            row['preview_audio_peak_before_clip']=previous['preview_audio_peak_before_clip']
            row['preview_mix_attenuation']=previous.get('preview_mix_attenuation',1.0)
            row['preview_square_pixels']=True
            save(folder/'completed.json',row)
            return row
    audios={};track_audio(row['source_refs']['wav'],row,'listener',audios)
    speaker=refs(row['speaker_archive'],row['speaker_id'],['wav'])['wav'];track_audio(speaker,row,'speaker',audios)
    sr=audios['listener'][1]
    if audios['speaker'][1]!=sr:raise ValueError('Sample rate mismatch')
    mixed=.5*(audios['speaker'][0]+audios['listener'][0])
    row['preview_audio_peak_before_clip']=round(float(np.max(np.abs(mixed))),4)
    attenuation=min(1.0,.98/max(.001,float(np.max(np.abs(mixed)))))
    row['preview_mix_attenuation']=round(attenuation,6)
    mixed*=attenuation
    sf.write(folder/'mix.wav',mixed,sr,subtype='FLOAT')
    duration=row['clip_duration_s']
    preview_h=2*round(720*row['display_height']/row['display_width']/2)
    run(['ffmpeg','-v','error','-nostdin','-y','-threads','2','-ss',str(row['clip_start_s']),'-t',str(duration),'-i',uri(row['source_refs']['mp4']),
         '-i',str(folder/'mix.wav'),'-map','0:v','-map','1:a','-t',str(duration),
         '-vf',f'scale=720:{preview_h}:flags=lanczos,setsar=1,fps=25','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20',
         '-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart',str(folder/'full_frame.pending.mp4')])
    (folder/'full_frame.pending.mp4').replace(folder/'full_frame.mp4');(folder/'mix.wav').unlink()
    row['preview_square_pixels']=True
    save(folder/'completed.json',row)
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');args=ap.parse_args()
    selection=json.loads((OUT/'selection.json').read_text());rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(prepare,selection['rows']):
            rows.append(r);print('BOX',len(rows),r['clip_id'],r['roi']['target_width'],r['roi']['target_height'],flush=True)
    save(OUT/'geometry_manifest.json',dict(seed=selection['seed'],source_manifest=selection['source_manifest'],rows=rows))
    # Contact sheet is also full-frame with a rectangle, never a spatial crop.
    sheet=np.full((3*448,10*252,3),245,np.uint8)
    for i,r in enumerate(rows):
        im=cv2.imread(str(OUT/r['poster']));b=r['roi'];sx=im.shape[1]/r['source_width'];sy=im.shape[0]/r['source_height']
        cv2.rectangle(im,(round(b['x']*sx),round(b['y']*sy)),(round((b['x']+b['width'])*sx),round((b['y']+b['height'])*sy)),(80,230,100),4)
        factor=min(252/im.shape[1],448/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*factor),round(im.shape[0]*factor)))
        tile=np.full((448,252,3),35,np.uint8);oy=(448-im.shape[0])//2;ox=(252-im.shape[1])//2
        tile[oy:oy+im.shape[0],ox:ox+im.shape[1]]=im
        cv2.putText(tile,f'{i+1:02d} {b["target_width"]}x{b["target_height"]}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        sheet[i//10*448:(i//10+1)*448,i%10*252:(i%10+1)*252]=tile
    cv2.imwrite(str(OUT/'contact_sheet.jpg'),sheet)
    if args.render:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            rows=list(pool.map(render,rows))
        save(OUT/'manifest.json',dict(seed=selection['seed'],source_manifest=selection['source_manifest'],rows=rows))
        print('FINISHED',len(rows),flush=True)


if __name__=='__main__':main()
