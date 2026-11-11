"""Isolated, single-object SAM 2.1 MLX trial; no Hogline imports or database writes."""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/src'))
import cv2
import mlx.core as mx
import numpy as np
from sam2.build import build_model
from sam2.predictor import SAM2VideoPredictor

SOURCE = ROOT.parents[1] / 'post-live/f87db63f-9e2a-4a1b-8f97-4d3281a0f675/recording.mp4'
CASES = [
    dict(name='clear-yellow-3603', start=3603., duration=8., crop=[0, 0, 720, 990], point=[405, 551]),
    dict(name='sweepers-red-1849', start=1849., duration=8., crop=[0, 0, 720, 990], point=[464, 657]),
    dict(name='opposite-yellow-5031', start=5031., duration=8., crop=[1200, 0, 1920, 990], point=[239, 611]),
]

def run_case(model, case, fps):
    out = ROOT / 'outputs' / case['name']
    out.mkdir(parents=True, exist_ok=True)
    x0,y0,x1,y1 = case['crop']
    w,h = x1-x0,y1-y0
    command = ['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-ss',str(case['start']),
        '-i',str(SOURCE),'-t',str(case['duration']),'-vf',f'crop={w}:{h}:{x0}:{y0},fps={fps}',
        '-threads','2','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    before=time.monotonic()
    raw=subprocess.run(command,check=True,capture_output=True).stdout
    frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,h,w,3)
    print(f"{case['name']}: loaded {len(frames)} frames ({frames.nbytes/1e9:.2f} GB RGB)",flush=True)
    pred=SAM2VideoPredictor(model)
    pred.init_state(frames,preload_features=False)
    point=np.array([case['point']],dtype=np.float32)
    pred.add_new_points(0,1,point,np.array([1],dtype=np.int32))
    writer=cv2.VideoWriter(str(out/'overlay-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h))
    maskwriter=cv2.VideoWriter(str(out/'mask-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h),False)
    if not writer.isOpened() or not maskwriter.isOpened(): raise RuntimeError('Video writer unavailable')
    rows=[]; samples=[]
    infer_started=time.monotonic()
    for f,ids,masks,ious in pred.propagate_in_video(max_frames=len(frames)):
        best=int(np.argmax(ious[0])); mask=np.asarray(masks[0][best])>0
        yy,xx=np.where(mask)
        row={'frame':f,'source_seconds':case['start']+f/fps,'mask_area_px':int(mask.sum()),
             'mask_quality_estimate':float(ious[0][best]),'candidate':best,
             'centroid_crop_px':[float(xx.mean()),float(yy.mean())] if len(xx) else None,
             'bbox_crop_px':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None}
        rows.append(row)
        bgr=cv2.cvtColor(frames[f],cv2.COLOR_RGB2BGR)
        overlay=bgr.copy();overlay[mask]=(overlay[mask]*.5+np.array([70,240,70])*.5).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(40,255,40),1)
        if f==0: cv2.drawMarker(overlay,tuple(case['point']),(0,0,255),cv2.MARKER_CROSS,14,2)
        for pic,label in [(bgr,'SOURCE'),(overlay,'SAM 2.1 MLX')]:
            cv2.rectangle(pic,(0,0),(w,42),(24,24,24),-1)
            cv2.putText(pic,f"{label}  t={row['source_seconds']:.2f}s",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
        panel=np.concatenate([bgr,overlay],axis=1)
        writer.write(panel);maskwriter.write(mask.astype(np.uint8)*255)
        if f in [0,len(frames)//4,len(frames)//2,3*len(frames)//4,len(frames)-1]:
            path=out/f'frame-{f:04}.jpg';cv2.imwrite(str(path),panel);samples.append(str(path))
        # Finished feature/output tensors are not needed again in this forward-only run.
        # Preserve the initial conditioning memory and all 16 recent pointer slots.
        pred._features.pop(f,None)
        pred._output_dict.pop(f,None)
        for old in list(pred._non_cond_frame_outputs):
            if old < f-16: del pred._non_cond_frame_outputs[old]
        if f%60==0:
            print(f"{case['name']}: {f+1}/{len(frames)}, {time.monotonic()-infer_started:.1f}s, Metal peak {mx.get_peak_memory()/1e9:.2f}GB",flush=True)
    writer.release();maskwriter.release()
    elapsed=time.monotonic()-before
    (out/'observations.json').write_text(json.dumps(rows,indent=2))
    for stem in ['overlay','mask']:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out/f'{stem}-raw.mp4'),
            '-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{stem}.mp4')],check=True)
        (out/f'{stem}-raw.mp4').unlink()
    report={**case,'fps':fps,'frames':len(frames),'elapsed_seconds':elapsed,
            'metal_peak_bytes':mx.get_peak_memory(),'nonempty_frames':sum(r['mask_area_px']>0 for r in rows),
            'samples':samples,'notice':'Single manually initialized stone. Masks and centroids are model proposals, not calibrated ice coordinates or verified identities.'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    del pred,frames,raw
    gc.collect();mx.clear_cache()
    print(json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,nargs='+');ap.add_argument('--fps',type=int,default=60)
    args=ap.parse_args()
    mx.set_cache_limit(512*1024**2)
    mx.set_memory_limit(12*1024**3)
    weights=ROOT/'base_plus.safetensors'
    print('GPU:',mx.device_info(),flush=True)
    model=build_model('base_plus',str(weights))
    # Fail if any expected parameter is missing, rather than accepting random defaults.
    model.load_weights(str(weights),strict=True)
    mx.eval(model.parameters())
    print('Weights loaded strictly',flush=True)
    selected=[CASES[i] for i in args.case] if args.case is not None else CASES
    reports=[run_case(model,c,args.fps) for c in selected]
    (ROOT/'last-run.json').write_text(json.dumps({'source':str(SOURCE),'model_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'vendor_revision':subprocess.check_output(['git','-C',str(ROOT/'vendor'),'rev-parse','HEAD'],text=True).strip(),
        'reports':reports},indent=2))
