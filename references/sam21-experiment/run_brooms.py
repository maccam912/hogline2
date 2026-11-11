"""Broom trial with independent object memories and shared image features."""
import argparse
import gc
import json
import subprocess
import time

from run_experiment import ROOT, SOURCE, CASES, cv2, mx, np, build_model, SAM2VideoPredictor
from sam2.predictor import _imagenet_normalize_chw

# Head and shaft points selected from the original first frames; negatives exclude people/stone.
PROMPTS = [
    [dict(name='left_broom', points=[[225,596],[204,560],[178,506],[144,443],[135,522]],labels=[1,1,1,1,0]),
     dict(name='right_broom', points=[[493,570],[507,520],[519,474],[553,515],[405,551]],labels=[1,1,1,0,0])],
    [dict(name='left_broom', points=[[383,612],[373,568],[359,517],[397,550]],labels=[1,1,1,0]),
     dict(name='right_broom', points=[[494,616],[550,570],[609,529],[657,494],[599,581],[464,657]],labels=[1,1,1,1,0,0])],
    [dict(name='left_broom', points=[[197,637],[165,579],[126,514],[95,464],[98,558]],labels=[1,1,1,1,0]),
     dict(name='right_broom', points=[[264,618],[280,582],[304,537],[319,504],[239,611],[348,548]],labels=[1,1,1,1,0,0])],
]
COLORS=[(255,190,30),(220,65,255)] # BGR: cyan/blue, pink

def render_frame(frame, masks, stone, source_seconds, prompts=None):
    bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR);overlay=bgr.copy()
    for mask,color in [(stone,(60,240,60)),*zip(masks,COLORS)]:
        overlay[mask]=(overlay[mask]*.35+np.asarray(color)*.65).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,color,1)
    if prompts:
        for prompt,color in zip(prompts,COLORS):
            for pt,label in zip(prompt['points'],prompt['labels']):
                cv2.drawMarker(overlay,tuple(pt),color if label else (0,0,255),cv2.MARKER_CROSS,9,1)
    for pic,label in [(bgr,'SOURCE'),(overlay,'STONE + BROOMS')]:
        cv2.rectangle(pic,(0,0),(pic.shape[1],70),(20,20,20),-1)
        cv2.putText(pic,f'{label}  t={source_seconds:.2f}s',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    for x,label,color in [(10,'stone',(60,240,60)),(150,'left broom',COLORS[0]),(350,'right broom',COLORS[1])]:
        cv2.putText(overlay,label,(x,55),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
    return np.concatenate([bgr,overlay],axis=1)

def run(model,ci,preview=False):
    case=CASES[ci];prompts=PROMPTS[ci]
    out=ROOT/'brooms'/case['name'];out.mkdir(parents=True,exist_ok=True)
    x0,y0,x1,y1=case['crop'];w,h=x1-x0,y1-y0
    started=time.monotonic()
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-ss',str(case['start']),'-i',str(SOURCE),
         '-t',str(case['duration']),'-vf',f'crop={w}:{h}:{x0}:{y0},select=not(mod(n\\,2))',
         '-fps_mode','passthrough','-threads','2']
    if preview: cmd+=['-frames:v','1']
    raw=subprocess.check_output(cmd+['-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,h,w,3)
    assert len(frames)==(1 if preview else 240),len(frames)
    base=SAM2VideoPredictor(model);base.init_state(frames,preload_features=False)
    predictors=[base]
    other=SAM2VideoPredictor(model)
    other._images=base._images;other._original_sizes=base._original_sizes;other._num_frames=base._num_frames
    other._features[0]=base._features[0];predictors.append(other)
    for pred,prompt in zip(predictors,prompts):
        pred.add_new_points(0,1,np.array(prompt['points'],np.float32),np.array(prompt['labels'],np.int32))
    generators=[p.propagate_in_video(max_frames=len(frames)) for p in predictors]
    stonecap=cv2.VideoCapture(str(ROOT/'outputs'/case['name']/'mask.mp4'))
    writers=[]
    if not preview:
        video=cv2.VideoWriter(str(out/'overlay-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(2*w,h))
        for prompt in prompts:
            writers.append(cv2.VideoWriter(str(out/(prompt['name']+'-raw.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h),False))
        assert video.isOpened() and all(v.isOpened() for v in writers)
    rows=[]
    for f in range(len(frames)):
        if f:
            norm=_imagenet_normalize_chw(base._images[f].transpose(2,0,1))
            feats=model.forward_image(mx.array(norm)[None]);mx.eval(*feats.values())
            for pred in predictors: pred._features[f]=feats
        masks=[];observations=[]
        for pred,generator,prompt in zip(predictors,generators,prompts):
            fi,ids,candidates,scores=next(generator);assert fi==f
            candidate=int(np.argmax(scores[0]));mask=candidates[0][candidate]>0
            masks.append(mask);yy,xx=np.where(mask)
            observations.append(dict(object=prompt['name'],area_px=int(mask.sum()),
                quality_estimate=float(scores[0][candidate]),
                bbox_crop_px=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None))
        ok,sm=stonecap.read();assert ok
        stone=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)>127
        stonecap.grab() # Previous stone run has 60 fps; current frames use indices 0, 2, 4, ...
        panel=render_frame(frames[f],masks,stone,case['start']+f/30,prompts if f==0 else None)
        if preview:
            cv2.imwrite(str(out/'prompt-preview.jpg'),panel)
            print(case['name'],observations,flush=True)
        else:
            video.write(panel)
            for writer,mask in zip(writers,masks): writer.write(mask.astype(np.uint8)*255)
            if f%30==0 or f==len(frames)-1: cv2.imwrite(str(out/f'frame-{f:04}.jpg'),panel)
        rows.append(dict(frame=f,source_seconds=case['start']+f/30,objects=observations))
        for pred in predictors:
            pred._features.pop(f,None);pred._output_dict.pop(f,None)
            for old in list(pred._non_cond_frame_outputs):
                if old<f-16: del pred._non_cond_frame_outputs[old]
        if f%30==0: print(f"{case['name']}: {f+1}/{len(frames)} frames, {time.monotonic()-started:.1f}s, GPU {mx.get_peak_memory()/1e9:.2f} GB",flush=True)
    stonecap.release()
    if not preview:
        video.release()
        for writer in writers:writer.release()
        elapsed=time.monotonic()-started
        (out/'observations.json').write_text(json.dumps(rows,indent=2))
        for stem in ['overlay','left_broom','right_broom']:
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out/f'{stem}-raw.mp4'),
                '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{stem}.mp4')],check=True)
            (out/f'{stem}-raw.mp4').unlink()
        report=dict(case=case,prompts=prompts,frames=len(frames),fps=30,elapsed_seconds=elapsed,
                    peak_metal_bytes=mx.get_peak_memory(),
                    method='Independent per-broom memories; shared frozen per-frame image features. Initial head/shaft positive points and exclusion points only. No later corrections. Stone overlay reused from earlier 60fps masks, thresholded and sampled every second frame.',
                    limitations='Unverified masks, not calibrated broom tips. Mask presence and quality scores do not establish correct segmentation.')
        (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    del base,other,predictors,generators,raw,frames
    gc.collect();mx.clear_cache()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');ap.add_argument('--case',type=int,nargs='+',default=[0,1,2]);args=ap.parse_args()
    mx.set_cache_limit(512*1024**2);mx.set_memory_limit(12*1024**3)
    model=build_model('base_plus',str(ROOT/'base_plus.safetensors'));model.load_weights(str(ROOT/'base_plus.safetensors'),strict=True);mx.eval(model.parameters())
    for ci in args.case:run(model,ci,args.preview)
