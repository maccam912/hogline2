import {project, rayX, rayBand, compareTarget, pairAt} from './tracking_aim_math.mjs';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function request(url, method='GET', body) {
  const r=await fetch(url,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  const data=await r.json(); if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail)); return data;
}
export function mountAim(container, base, runId, manifest, overlay) {
  let disposed=false, data=null, ray=null, pair=null, target=null, evidenceImage=null, broomImage=null;
  let evidenceToken=0, broomToken=0, selectedTime=0, broomTime=0, saving=false, timer=null;
  const endpoint='/api'+base+'/runs/'+runId+'/aim';
  const q=s=>container.querySelector(s);
  container.innerHTML='<h2>3. Review hog-line aim</h2><p role="status">Loading local observations…</p>';
  const fpsText=String(manifest.media.fps).split('/').map(Number), fps=fpsText[0]/(fpsText[1]||1);
  const frameUrl=t=>'/api'+base+'/frame?seconds='+encodeURIComponent(t.toFixed(6));
  const view=id=>data.views.find(v=>v.id===id);
  const farView=()=>view(q('.aim-camera').value);
  const note=s=>{q('.aim-message').textContent=s;};
  function stop(){clearTimeout(timer);timer=null;if(q('.aim-play'))q('.aim-play').textContent='Play pairs';}
  function cropImage(canvas, img, crop) {
    const [x,y,x1,y1]=crop;canvas.width=x1-x;canvas.height=y1-y;
    const ctx=canvas.getContext('2d');ctx.drawImage(img,x,y,x1-x,y1-y,0,0,x1-x,y1-y);return ctx;
  }
  function cross(ctx,p,color,r=7){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[0]-r,p[1]);ctx.lineTo(p[0]+r,p[1]);ctx.moveTo(p[0],p[1]-r);ctx.lineTo(p[0],p[1]+r);ctx.stroke();}
  function trace(ctx, v, crop, points, color, dash=[], fill=false){
    const mapped=points.map(p=>project(v.ice_to_image,p)).filter(p=>p&&p.every(Number.isFinite));
    if(mapped.length<2)return;
    ctx.beginPath();mapped.forEach((p,i)=>ctx[i?'lineTo':'moveTo'](p[0]-crop[0],p[1]-crop[1]));
    if(fill){ctx.closePath();ctx.fillStyle=color;ctx.fill();}
    else{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
  }
  function drawFar(){
    const v=farView(), c=q('.aim-house');if(!v||!broomImage)return;
    const ctx=cropImage(c,broomImage,v.crop), tee=data.receiving_end==='A'?0:data.tee_distance_m;
    const ys=Array.from({length:33},(_,i)=>tee-4+i/4);
    if(ray.status==='estimated'&&q('.aim-fixed').checked){
      if(q('.aim-band').checked)trace(ctx,v,v.crop,[...ys.map(y=>[rayX(ray,y)-rayBand(ray,y),y]),...ys.slice().reverse().map(y=>[rayX(ray,y)+rayBand(ray,y),y])],'#ffb02e40',[],true);
      trace(ctx,v,v.crop,ys.map(y=>[rayX(ray,y),y]),'#ff9e16');
    }
    if(pair&&q('.aim-pairs').checked)trace(ctx,v,v.crop,ys.filter(y=>pair.direction*(y-pair.origin[1])>=0).map(y=>[rayX(pair,y),y]),'#d128ed',[6,3]);
    if(target&&target.view_id===v.id){
      const p=target.point.map((x,i)=>x-v.crop[i]);cross(ctx,p,'#00ffff',10);
      if(ray.status==='estimated'){
        const ice=project(v.image_to_ice,target.point), desired={origin:ray.origin,slope_dx_dy:(ice[0]-ray.origin[0])/(ice[1]-ray.origin[1])};
        trace(ctx,v,v.crop,ys.map(y=>[rayX(desired,y),y]),'#00dfe5',[3,5]);
      }
    }
    const format=value=>`${(Math.abs(value)*100).toFixed(0)} cm ${value>=0?'right':'left'} of broom`;
    let summary=ray.status==='estimated'?`Fixed ray: ${(ray.angle_degrees).toFixed(2)}° in sheet coordinates. `:'';
    if(target&&target.view_id===v.id){
      if(ray.status==='estimated'){const result=compareTarget(ray,v,target.point);summary=`Fixed ray: ${format(result.offsetRight)}. Fit sensitivity ±${(result.band*100).toFixed(0)} cm. `;}
      if(pair){const ice=project(v.image_to_ice,target.point);summary+=pair.direction*(ice[1]-pair.origin[1])>0?`Current pair: ${format(compareTarget(pair,v,target.point).offsetRight)}. `:'Current pair has passed the target depth. ';}
      summary+=`Broom marked at source ${target.source_seconds.toFixed(3)} s. Left/right is looking from the delivery end.`;
    }else summary+='Click the broom’s ice-contact point to compare. Rays outside this camera crop are not visible.';
    q('.aim-comparison').textContent=summary;
  }
  function deliveryCrop(){
    const v=view(data.delivery_view), points=ray.status==='estimated'?ray.samples.map(s=>s.point):[];
    if(pair)points.push(pair.point,pair.previous_point);
    if(!points.length)return v.crop;
    const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
    return [Math.max(v.crop[0],Math.floor(Math.min(...xs)-95)),Math.max(v.crop[1],Math.floor(Math.min(...ys)-65)),Math.min(v.crop[2],Math.ceil(Math.max(...xs)+95)),Math.min(v.crop[3],Math.ceil(Math.max(...ys)+65))];
  }
  function drawEvidence(){
    if(!evidenceImage)return;const v=view(data.delivery_view), crop=deliveryCrop(), ctx=cropImage(q('.aim-evidence'),evidenceImage,crop);
    if(ray.status==='estimated'){
      trace(ctx,v,crop,[[-3,ray.origin[1]],[3,ray.origin[1]]],'#20d9ff',[5,4]);
      trace(ctx,v,crop,[-2,2].map(d=>[rayX(ray,ray.origin[1]+d),ray.origin[1]+d]),'#ff9e16');
    }
    if(pair){cross(ctx,pair.point.map((x,i)=>x-crop[i]),'#ef37ff');cross(ctx,pair.previous_point.map((x,i)=>x-crop[i]),'#ffffff',5);}
    const sample=ray.samples?.find(s=>Math.abs(s.seconds-selectedTime)<.005);
    if(sample)cross(ctx,sample.point.map((x,i)=>x-crop[i]),'#63ff94');
  }
  async function loadImage(seconds, kind){
    const token=kind==='broom'?++broomToken:++evidenceToken;
    const image=new Image();image.src=frameUrl(seconds);
    try{await image.decode();if(disposed||token!==(kind==='broom'?broomToken:evidenceToken))return false;
      if(kind==='broom'){broomImage=image;broomTime=seconds;q('.aim-time').value=seconds.toFixed(6);q('.aim-house').dataset.seconds=seconds;drawFar();}
      else{evidenceImage=image;selectedTime=seconds;pair=pairAt(ray.pair_rays||[],seconds);drawEvidence();drawFar();q('.aim-evidence-time').textContent=`Delivery source ${seconds.toFixed(3)} s${pair?` · pair Δt ${pair.dt.toFixed(3)} s`:' · no consecutive forward pair here'}`;}
      return true;
    }catch{if(!disposed&&token===(kind==='broom'?broomToken:evidenceToken))note('Could not load that source frame.');return false;}
  }
  async function selectTime(seconds, seek=false){
    q('.aim-frame-link').href=frameUrl(seconds);
    if(seek){overlay.pause();overlay.currentTime=Math.max(0,seconds-manifest.source_start);}
    return loadImage(seconds,'evidence');
  }
  function sampleButtons(){
    const holder=q('.aim-samples');holder.replaceChildren();
    for(const s of ray.samples||[]){const b=document.createElement('button');b.className='secondary';b.textContent=s.seconds.toFixed(3)+' s';b.onclick=()=>{stop();selectTime(s.seconds,true);};holder.appendChild(b);}
  }
  async function choose(){
    stop();++evidenceToken;++broomToken;evidenceImage=null;broomImage=null;pair=null;
    for(const c of container.querySelectorAll('canvas'))c.getContext('2d').clearRect(0,0,c.width,c.height);
    ray=data.rays.find(r=>r.stone_id===q('.aim-stone').value);if(!ray)return;
    target=ray.review?.target||null;
    if(target)q('.aim-camera').value=target.view_id;
    note(!farView()?'No receiving-house overhead camera is calibrated for this run.':ray.review_stale?'Previous broom mark is stale; place it again.':target?'Saved broom mark loaded.':'');
    q('.aim-save').disabled=!ray.evidence_sha256||!target||!farView();q('.aim-clear').disabled=!target;q('.aim-house').style.cursor=ray.evidence_sha256?'crosshair':'default';
    q('.aim-details').textContent=ray.status==='estimated'?`Estimated crossing ${ray.crossing_seconds.toFixed(3)} s (between ${ray.crossing_bracket.map(t=>t.toFixed(3)).join(' and ')} s). ${ray.sample_count} accepted observations over ${ray.span_m.toFixed(2)} m at about ${ray.sample_rate_hz.toFixed(1)} fps. No whole-shot smoothing. ${ray.pair_rays.length} consecutive forward pairs available.`:`Fixed ray unavailable: ${ray.reason}. ${ray.pair_rays?.length||0} consecutive forward pairs available for inspection.`;
    sampleButtons();
    const pairs=ray.pair_rays||[], t=ray.reference_seconds??pairs[0]?.seconds??manifest.source_start;
    q('.aim-pair-index').max=Math.max(0,pairs.length-1);q('.aim-pair-index').disabled=!pairs.length;q('.aim-play').disabled=!pairs.length;
    const index=pairs.findIndex(p=>p.seconds>=t);q('.aim-pair-index').value=Math.max(0,index);
    await Promise.all([selectTime(pairs[Math.max(0,index)]?.seconds??t),loadImage(target?.source_seconds??t,'broom')]);
  }
  async function stepPair(index){
    overlay.pause();q('.aim-follow').checked=false;
    const pairs=ray.pair_rays||[];index=Math.max(0,Math.min(pairs.length-1,index));if(!pairs.length)return;
    q('.aim-pair-index').value=index;await selectTime(pairs[index].seconds);
  }
  async function play(){
    if(timer!==null){stop();return;}
    q('.aim-play').textContent='Pause pairs';
    const tick=async()=>{if(disposed)return;const index=Number(q('.aim-pair-index').value), pairs=ray.pair_rays||[];
      if(index>=pairs.length-1){stop();return;}
      const started=performance.now();await stepPair(index+1);if(timer===null||disposed)return;
      timer=setTimeout(tick,Math.max(0,pairs[index+1].dt*1000-(performance.now()-started)));
    };
    timer=setTimeout(tick,0);
  }
  const sync=()=>{if(disposed||!ray||!q('.aim-follow')?.checked)return;const seconds=manifest.source_start+overlay.currentTime;pair=pairAt(ray.pair_rays||[],seconds);drawFar();q('.aim-live').textContent=pair?`Replay pair ending ${pair.seconds.toFixed(3)} s`:'Replay: no consecutive forward pair';};
  overlay.addEventListener('timeupdate',sync);
  (async()=>{try{
    data=await request(endpoint);if(disposed)return;
    container.innerHTML=`<h2>3. Review hog-line aim</h2>
      <p>Orange: fixed heading at the delivery hog line. Magenta: the unsmoothed direction between consecutive observations. Cyan: your broom target. The straight rays show direction, not the stone’s eventual curled path.</p>
      <div class="controls"><label>Stone<select class="aim-stone">${data.rays.map(r=>`<option value="${esc(r.stone_id)}">${esc(r.stone_id)}${r.status==='estimated'?'':' · fixed ray unavailable'}</option>`).join('')}</select></label><label>Far-house camera<select class="aim-camera">${data.receiving_views.map(id=>`<option>${esc(id)}</option>`).join('')}</select></label></div>
      <p class="aim-details muted"></p>
      <div class="toolbar aim-toggles"><label><input type="checkbox" class="aim-fixed" checked> Fixed ray</label><label><input type="checkbox" class="aim-pairs" checked> Pair ray</label><label><input type="checkbox" class="aim-band" checked> Fit sensitivity band</label><label><input type="checkbox" class="aim-follow" checked> Follow replay for pair ray</label></div>
      <div class="aim-grid"><div><h3>Far house · mark the broom</h3><canvas class="aim-house" width="458" height="456" aria-label="Far house with aim rays; click broom ice-contact point"></canvas>
      <div class="controls"><label>Broom source time (seconds)<input class="aim-time" type="number" min="0" step="0.016667"></label><button class="aim-show secondary">Show frame</button></div>
      <div class="toolbar"><button class="aim-earlier secondary">−1 second</button><button class="aim-prev-frame secondary">−1 frame</button><button class="aim-next-frame secondary">+1 frame</button><button class="aim-later secondary">+1 second</button></div>
      <p class="aim-comparison" aria-live="polite"></p><button class="aim-save" disabled>Save broom mark</button> <button class="aim-clear secondary" disabled>Clear broom mark</button><p class="aim-message" role="status"></p></div>
      <div><h3>Delivery evidence · inspect the wiggle</h3><canvas class="aim-evidence" width="400" height="300" aria-label="Delivery source frame and tracked stone points"></canvas><p class="aim-evidence-time muted"></p>
      <div class="toolbar"><button class="aim-back secondary">Previous pair</button><button class="aim-play">Play pairs</button><button class="aim-next secondary">Next pair</button></div><label>Analyzed pair<input class="aim-pair-index" type="range" min="0" value="0" step="1"></label><p class="aim-live muted"></p>
      <p class="muted">Magenta cross: current observation. White: previous observation. Green: sample used in the fixed fit. Blue line: calibrated hog line.</p><details><summary>Frames used for the fixed ray</summary><div class="toolbar aim-samples"></div></details><a class="aim-frame-link" target="_blank">Open original delivery frame</a></div></div>
      <p class="muted">The band shows fit sensitivity to two-pixel tracking jitter and local scatter; it excludes systematic camera-calibration and stone-height error. The crossing uses the tracker’s approximate ice-plane anchor. Pair rays use the delivery camera only and disappear across missing/rejected frames, reversals or contacts. Playback may slow while source frames decode. Broom marks are manual; they do not certify delivery accuracy.</p>`;
    if(!data.rays.length){q('.aim-details').textContent='No tracked stones in this run.';return;}
    if(!data.receiving_views.length)note('No receiving-house overhead camera is calibrated for this run.');
    q('.aim-stone').onchange=choose;
    q('.aim-camera').onchange=()=>{target=null;q('.aim-save').disabled=true;drawFar();note('Click the broom in this camera.');};
    for(const cls of ['.aim-fixed','.aim-pairs','.aim-band'])q(cls).onchange=drawFar;
    q('.aim-follow').onchange=()=>{if(q('.aim-follow').checked)stop();sync();};
    q('.aim-house').onclick=e=>{
      if(saving||!ray?.evidence_sha256||!broomImage||!farView())return;
      const c=e.target,r=c.getBoundingClientRect(),v=farView();
      target={view_id:v.id,source_seconds:broomTime,point:[v.crop[0]+(e.clientX-r.left)*c.width/r.width,v.crop[1]+(e.clientY-r.top)*c.height/r.height]};
      q('.aim-save').disabled=false;q('.aim-clear').disabled=false;note('Broom mark changed — save to keep it.');drawFar();
    };
    async function showBroom(t){
      const center=ray.reference_seconds??selectedTime,low=Math.max(0,center-30),high=Math.min(manifest.media.duration-1/fps,center+2);
      if(!Number.isFinite(t)||t<low||t>high){note(`Choose a source time from ${low.toFixed(2)} to ${high.toFixed(2)} s.`);return;}
      await loadImage(t,'broom');
    }
    q('.aim-show').onclick=()=>showBroom(Number(q('.aim-time').value));
    for(const [cls,dt] of [['.aim-earlier',-1],['.aim-later',1],['.aim-prev-frame',-1/fps],['.aim-next-frame',1/fps]])q(cls).onclick=()=>showBroom(broomTime+dt);
    q('.aim-pair-index').oninput=()=>{stop();stepPair(Number(q('.aim-pair-index').value));};
    q('.aim-back').onclick=()=>{stop();stepPair(Number(q('.aim-pair-index').value)-1);};
    q('.aim-next').onclick=()=>{stop();stepPair(Number(q('.aim-pair-index').value)+1);};
    q('.aim-play').onclick=play;
    async function save(clear=false){
      if(saving||!ray?.evidence_sha256)return;saving=true;
      const selected=ray, marked=clear?null:target;
      q('.aim-save').disabled=true;q('.aim-clear').disabled=true;
      try{const review=await request(endpoint+'/'+encodeURIComponent(selected.stone_id),'PUT',{evidence_sha256:selected.evidence_sha256,revision:selected.review_revision,target:marked});
        selected.review=review;selected.review_revision=review.revision;
        if(!disposed&&selected===ray){target=review.target;note(clear?'Broom mark cleared.':'Broom mark saved.');drawFar();}
      }catch(e){if(!disposed&&selected===ray)note(e.message);}
      finally{saving=false;if(!disposed){q('.aim-save').disabled=!target||!ray.evidence_sha256;q('.aim-clear').disabled=!target;}}
    }
    q('.aim-save').onclick=()=>save();q('.aim-clear').onclick=()=>save(true);
    q('.aim-stone').value=data.rays.find(r=>r.status==='estimated')?.stone_id||data.rays[0].stone_id;
    await choose();
  }catch(e){if(!disposed)container.textContent='Aim review unavailable: '+e.message;}})();
  return ()=>{disposed=true;stop();++evidenceToken;++broomToken;overlay.removeEventListener('timeupdate',sync);};
}
