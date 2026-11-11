import {mountAim} from './tracking_aim.js';
import {mountColab} from './tracking_colab.js';
import {mountGamePanel, mountShotReviews} from './game.js';
import {mapPoint, nearHouse, reviewEvents, sheetLayout} from './tracking_map.mjs';
const $=s=>document.querySelector(s), identifier=new URLSearchParams(location.search).get('recording');
const base='/recordings/'+encodeURIComponent(identifier)+'/tracking';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let rawSets=[],runEvents=[],disposeAim=null;
let calibration,record,viewIndex=0,background=new Image(),cropClicks=null,rows=[],manifest=null,activeRun=null,forwardRows=[],wholeRows=null,trajectorySummary=null;
async function api(path,method='GET',body){let r=await fetch('/api'+path,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});let d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail));return d}
function say(s){$('#message').textContent=s}function fail(e){say(e.message)}
const renderColab=mountColab({base,say,fail,poll,openRun});
function describeGeometry(){
  const g=sheetLayout(calibration);
  const estimated=calibration.views.filter(v=>v.mapping_status==='inherited_estimate'&&!v.camera_calibration);
  $('#layout-description').textContent=(calibration.layout_source_recording?`Shared layout reused from ${calibration.layout_source_recording}: panel boundaries, overhead alignment and house directions are ready. `:'')+(estimated.length?'Perspective zoom and aim use starting estimates from that recording. Optional alignment adjustments are below.':'');
  const fitted=calibration.views.filter(v=>v.camera_calibration).map(v=>v.id);
  $('#geometry-description').textContent=`Sheet ${g.length_m.toFixed(3)} × ${g.width_m.toFixed(3)} m; tee–hog ${g.hog_from_tee_m.toFixed(3)} m; tee–hack ${g.hack_from_tee_m.toFixed(3)} m; house diameter ${(2*g.house_radius_m).toFixed(3)} m. ${calibration.geometry_evidence||'Standard dimensions; not site-surveyed.'} ${fitted.length?'Independent fitted camera mappings: '+fitted.join(', ')+'. Editing landmarks, crops, orientation or dimensions clears affected fits.':'Camera mappings use the marked house rings.'}`;
}
$('#load-venue').onclick=async()=>{try{
  const venue=$('#venue').value.trim();
  const dimensions=await api('/tracking/venues/'+encodeURIComponent(venue));
  Object.assign(calibration,dimensions);
  $('#tee-distance').value=calibration.tee_distance_m;
  $('#tee-source').value=calibration.tee_distance_source;
  $('#fixed-length').checked=!!calibration.tee_distance_fixed;
  describeGeometry();say('Venue dimensions loaded. Save landmarks to use them for future runs.');
}catch(e){fail(e)}};
function current(){return calibration.views[viewIndex]}
function populate(){const v=current();$('#view').innerHTML=calibration.views.map((v,i)=>`<option value="${i}">${esc(v.id)}</option>`).join('');$('#view').value=viewIndex;$('#view-name').value=v?.id||'';$('#view-kind').value=v?.kind||'perspective';$('#view-end').value=v?.end||'A';$('#view-rotation').value=v?.rotation_degrees||0;$('#alignment-check-label').hidden=v?.kind!=='perspective';$('#alignment-checked').checked=v?.mapping_status!=='inherited_estimate';drawCalibration()}
function drawCalibration(){const c=$('#calibration-canvas'),ctx=c.getContext('2d');if(!background.naturalWidth)return;c.width=background.naturalWidth;c.height=background.naturalHeight;ctx.drawImage(background,0,0);calibration.views.forEach((v,i)=>{let [x,y,x1,y1]=v.crop;ctx.strokeStyle=i===viewIndex?'#e5f794':'#66a69d';ctx.lineWidth=3;ctx.strokeRect(x,y,x1-x,y1-y);ctx.fillStyle=ctx.strokeStyle;ctx.font='20px sans-serif';ctx.fillText(v.id,x+8,y+26);if(i===viewIndex)for(const [name,p]of Object.entries(v.landmarks)){ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.fill();ctx.fillText(name,p[0]+10,p[1]-8)}});$('#point-status').textContent=current()?Object.entries(current().landmarks).map(([n,p])=>`${n}: ${p.map(v=>Math.round(v)).join(', ')}`).join(' · '):'Add a view, mark its crop, then click its landmarks.'}
function loadFrame(){calibration.source_seconds=Number($('#frame-time').value);background.src='/api'+base+'/frame?seconds='+calibration.source_seconds}
background.onload=drawCalibration;background.onerror=()=>say('Could not load that source frame.');
$('#load-frame').onclick=loadFrame;$('#view').onchange=()=>{viewIndex=+$('#view').value;cropClicks=null;populate()};
$('#new-view').onclick=()=>{calibration.views.push({id:'view_'+(calibration.views.length+1),kind:'perspective',end:'A',crop:[0,0,record.media.width,record.media.height],landmarks:{}});viewIndex=calibration.views.length-1;populate();say('Mark the two crop corners, then the button and four ring edges.')};
$('#alignment-checked').onchange=()=>{if(current()?.kind==='perspective')current().mapping_status=$('#alignment-checked').checked?'marked':'inherited_estimate';describeGeometry()};
$('#remove-view').onclick=()=>{calibration.views.splice(viewIndex,1);viewIndex=0;populate()};
for(const [id,key]of [['view-name','id'],['view-kind','kind'],['view-end','end'],['view-rotation','rotation_degrees']])$('#'+id).onchange=()=>{if(current()){current()[key]=key==='rotation_degrees'?Number($('#'+id).value):$('#'+id).value;populate()}};
$('#crop').onclick=()=>{cropClicks=[];say('Click the top-left and bottom-right corners of this camera panel.')};
$('#calibration-canvas').onclick=e=>{const v=current();if(!v)return;const rect=e.target.getBoundingClientRect();const p=[Math.round((e.clientX-rect.left)*e.target.width/rect.width),Math.round((e.clientY-rect.top)*e.target.height/rect.height)];if(cropClicks){cropClicks.push(p);if(cropClicks.length===2){const[a,b]=cropClicks;v.crop=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];cropClicks=null;say('Crop marked. Select a landmark and click it.')}}else v.landmarks[$('#landmark').value]=p;drawCalibration()};
$('#save-calibration').onclick=async()=>{try{calibration.first_delivery_end=$('#first-end').value;calibration.tee_distance_m=Number($('#tee-distance').value);calibration.tee_distance_source=$('#tee-source').value;calibration.venue_id=$('#venue').value.trim()||null;calibration.tee_distance_fixed=$('#fixed-length').checked;calibration=await api(base+'/calibration','PUT',calibration);describeGeometry();say('Landmarks saved. Future runs use this calibration revision.')}catch(e){fail(e)}};
$('#run').onclick = async () => {
  const execution = $('#execution').value;
  try {
    const result = await api(base + '/runs', 'POST', {
      execution,
      colab_model: $('#colab-model').value,
      end_number: Number($('#end').value),
      fps: Number($('#fps').value),
      duration_seconds: $('#length').value === 'full' ? null : Number($('#length').value),
    });
    activeRun = result.run_id;
    say(execution === 'colab'
      ? 'Preparing the video bundle on this Mac. Download it below, then run the notebook in Colab.'
      : 'Starting SAM2.1 on this Mac.');
    await poll();
  } catch (error) { fail(error); }
};
$('#stop').onclick=async()=>{try{await api(base+'/runs/'+activeRun+'/stop','POST');await poll()}catch(e){fail(e)}};
async function poll(){try{const list=await api(base+'/runs');renderColab(list);$('#runs').innerHTML=list.map(r=>`<button class="secondary" data-run="${r.run_id}" ${r.state!=='complete'?'disabled':''}>End ${r.end_number} · ${r.duration.toFixed(0)}s · cal ${r.calibration_revision} · ${r.raw_comparison?esc(r.backend)+' · raw velocity':r.execution==='colab'?'Colab SAM2.1'+(r.backend.includes('large')?' large':' base-plus'):r.backend==='sam2.1-mlx-granite'?'SAM2.1 on this Mac':r.trajectory_model?'Whole-clip motion':r.motion_model?'Kalman motion':'Original tracker'} · ${esc(r.stage)}${r.stones!==undefined?' · '+r.stones+' IDs':''}${r.error?' · '+esc(r.error):''}</button>`).join('');$('#runs').querySelectorAll('button').forEach(b=>b.onclick=()=>openRun(b.dataset.run));const busy=list.find(r=>['queued','running'].includes(r.state));$('#run').disabled=!!busy;$('#stop').disabled=!busy;if(busy)activeRun=busy.run_id;}catch(e){fail(e)}}
async function openRun(id){disposeAim?.();disposeAim=null;$('#aim-review').textContent='';try{const prefix='/api'+base+'/runs/'+id+'/';const response=await fetch(prefix+'manifest.json');if(!response.ok)throw Error('This run is not ready to review yet. Choose a completed run below.');manifest=await response.json();say('');rawSets=[];$('#speed-mode').value=manifest.raw_comparison?'raw':'filtered';$('#speed-mode option[value=filtered]').disabled=!!manifest.raw_comparison;$('#speed-mode option[value=raw]').disabled=!manifest.raw_comparison;$('#speed-view').innerHTML=manifest.calibration.views.map(v=>`<option>${esc(v.id)}</option>`).join('');$('#speed-view').value='perspective_a';if(manifest.raw_comparison){const runs=await api(base+'/runs');for(const run of runs.filter(r=>r.state==='complete'&&r.comparison_id===manifest.comparison_id)){const raw=await(await fetch('/api'+base+'/runs/'+run.run_id+'/raw-velocity.json')).json();rawSets.push({backend:run.backend,samples:raw.samples});}rawSets.sort((a,b)=>a.backend.localeCompare(b.backend));}const text=await(await fetch(prefix+'observations.jsonl')).text();forwardRows=text.trim().split('\n').filter(Boolean).map(s=>JSON.parse(s));rows=forwardRows;wholeRows=null;trajectorySummary=null;$('#trajectory-mode').value='forward';$('#trajectory-mode option[value=whole]').disabled=true;$('#focus-stone').innerHTML=(manifest.stones||[]).map(s=>`<option>${esc(s.id)}</option>`).join('');if(manifest.trajectory_model){const trajectory=await(await fetch(prefix+'trajectory.json')).json();wholeRows=trajectory.frames;trajectorySummary=await(await fetch(prefix+'trajectory-summary.json')).json();$('#trajectory-mode option[value=whole]').disabled=false;$('#trajectory-mode').value='whole';rows=wholeRows;}updateTrajectoryNotice();$('#overlay').hidden=false;$('#overlay').src=prefix+'overlay.mp4';$('#run-description').textContent=`End ${manifest.end.number} · source ${manifest.source_start.toFixed(2)}s · ${manifest.fps} fps · calibration revision ${manifest.calibration.revision} · ${manifest.execution==='colab'?'Colab GPU · ':''} ${manifest.raw_comparison?manifest.backend+" · unfiltered masks and velocities":manifest.motion_model?"Kalman motion with uncertainty":"Original tracker — rerun to apply motion filtering"}`;if(manifest.calibration.views.some(v=>v.mapping_status==='inherited_estimate'&&!v.camera_calibration))$('#run-description').textContent+=' · Perspective zoom/aim estimated from shared layout';$('#artifact-links').innerHTML=`<a href="#aim-review">Review aim rays</a><a href="${prefix}overlay.mp4" download>Download overlay</a><a href="${prefix}manifest.json" target="_blank">Run manifest</a><a href="${prefix}events.json" target="_blank">Identity and contact events</a>${manifest.raw_comparison?`<a href="${prefix}raw-velocity.json" download>Raw velocity data</a>`:''}${manifest.camera_report?`<a href="${prefix}comparison.html" target="_blank">Before / after camera calibration</a>`:''}${manifest.granite_report?`<a href="${prefix}comparison.html" target="_blank">Granite masks and camera alignment</a>`:''}${manifest.comparison_report?`<a href="${prefix}comparison.html" target="_blank">Compare velocity windows</a><a href="${prefix}comparison.mp4" target="_blank">Comparison highlights</a>`:''}${manifest.trajectory_model?`<a href="${prefix}trajectory.json" download>Whole-clip coordinates</a><a href="${prefix}geometry-report.json" target="_blank">Geometry audit</a>`:''}`;mountShotReviews($('#shot-reviews'),identifier,id,manifest);disposeAim=mountAim($('#aim-review'),base,id,manifest,$('#overlay'));const events=await(await fetch(prefix+'events.json')).json();runEvents=events;$('#events').innerHTML='<h3>Review an event</h3><div class=toolbar>'+(manifest.raw_comparison?events:reviewEvents(events)).map(e=>`<button class=secondary data-seconds="${e.seconds}">${e.seconds.toFixed(1)}s · ${esc(e.type.replaceAll('_',' '))} · ${esc(e.stone||(e.stones||[]).join(' ↔ '))}${e.count>1?' ×'+e.count:''}</button>`).join('')+'</div>';$('#events').querySelectorAll('button').forEach(b=>b.onclick=()=>{$('#overlay').currentTime=Math.max(0,Number(b.dataset.seconds)-manifest.source_start-1);drawMap()});drawMap()}catch(e){fail(e)}}
function drawMap(){
  const c=$('#map'),ctx=c.getContext('2d'),g=sheetLayout(manifest?.calibration||calibration||{}),teeDistance=g.tee_distance_m,scale=Math.min((c.height-80)/g.length_m,(c.width-120)/g.width_m),x0=150,y0=c.height-40+g.min_y*scale;
  const delivery=manifest?.calibration.views.find(v=>v.id===manifest.delivery_view)?.end||'B';
  const near=nearHouse($('#map-orientation').value,delivery);
  const xy=p=>{const q=mapPoint(p,near,teeDistance);return [x0+q[0]*scale,y0-q[1]*scale]};
  $('#orientation-description').textContent=`Looking from ${near} toward ${near==='A'?'B':'A'}. Delivery: ${delivery} → ${delivery==='A'?'B':'A'}.`;
  ctx.fillStyle='#eaf3f0';ctx.fillRect(0,0,c.width,c.height);
  ctx.strokeStyle='#8a9f9b';ctx.strokeRect(x0-g.width_m/2*scale,y0-g.max_y*scale,g.width_m*scale,g.length_m*scale);
  $('#map-dimensions').textContent=`Sheet ${g.length_m.toFixed(2)} × ${g.width_m.toFixed(2)} m · buttons ${teeDistance.toFixed(3)} m. Dimensions follow this run’s calibration; lines are ice-plane references.`;
  for(const y of [0,teeDistance]){
    const [x,p]=xy([0,y]);
    for(const [r,color]of [[g.house_radius_m,'#4284b4'],[g.house_radius_m*2/3,'#f3f6f1'],[g.house_radius_m/3,'#a5c96b'],[.152,'#f3f6f1']]){
      ctx.beginPath();ctx.arc(x,p,r*scale,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
    }
    ctx.fillStyle='#35524b';ctx.font='14px sans-serif';ctx.fillText(y===0?'A':'B',x-65,p+5);
  }
  for(const marking of g.markings) for(const y of marking.ys){
    const py=xy([0,y])[1],half=marking.name==='Hack'?.457/2:g.width_m/2;
    ctx.strokeStyle=marking.name==='Hog'?'#aa6670':'#829b94';ctx.lineWidth=marking.name==='Hog'?2:1;
    ctx.beginPath();ctx.moveTo(x0-half*scale,py);ctx.lineTo(x0+half*scale,py);ctx.stroke();
    ctx.fillStyle='#35524b';ctx.font='10px sans-serif';ctx.fillText(marking.name,x0+g.width_m/2*scale+5,py+3);
  }
  ctx.lineWidth=1;
  ctx.setLineDash([4,5]);ctx.strokeStyle='#afbfba';ctx.beginPath();ctx.moveTo(x0,25);ctx.lineTo(x0,820);ctx.stroke();ctx.setLineDash([]);
  if(!rows.length)return;
  const t=manifest.source_start+$('#overlay').currentTime;
  let lo=0,hi=rows.length-1;
  while(lo<hi){const m=Math.ceil((lo+hi)/2);if(rows[m].source_seconds<=t)lo=m;else hi=m-1}
  const row=rows[lo],next=rows[Math.min(rows.length-1,lo+1)];
  $('#map-status').textContent=row.stones.map(s=>`${s.id} · ${s.estimate_kind||(s.visible?'observed':'hidden')} · ${s.trajectory_phase||s.state}${s.speed_mps!==undefined?' · '+s.speed_mps.toFixed(2)+' m/s':''}${s.uncertainty95_m!==undefined?' · uncertainty '+s.uncertainty95_m.toFixed(2)+' m':''}`).join(' | ');
  for(const stone of row.stones){
    const color=stone.color==='red'?'#df3c56':'#b89b00';
    let previous=null,previousView=null;
    for(let i=Math.max(0,lo-Math.ceil(manifest.fps*12));i<=lo;i++){
      const sample=rows[i].stones.find(s=>s.id===stone.id);
      if(!sample||sample.estimate_kind==='lost'){previous=null;continue}
      const point=xy(sample.position);
      if(previous&&(!manifest.raw_comparison||previousView===sample.raw_view)){ctx.strokeStyle=color;ctx.globalAlpha=.4;ctx.setLineDash(sample.visible?[]:[3,4]);ctx.beginPath();ctx.moveTo(...previous);ctx.lineTo(...point);ctx.stroke()}
      previous=point;previousView=sample.raw_view;
    }
    ctx.globalAlpha=1;ctx.setLineDash([]);
    let position=stone.position;
    const following=next.stones.find(s=>s.id===stone.id);
    if(!manifest.raw_comparison&&following&&next.source_seconds>row.source_seconds&&stone.estimate_kind!=='lost'&&following.estimate_kind!=='lost'){
      const f=Math.max(0,Math.min(1,(t-row.source_seconds)/(next.source_seconds-row.source_seconds)));
      position=position.map((v,i)=>v+(following.position[i]-v)*f);
    }
    const [x,y]=xy(position);
    if(stone.position_covariance){
      const rx=Math.sqrt(5.991*stone.position_covariance[0][0])*scale,ry=Math.sqrt(5.991*stone.position_covariance[1][1])*scale;
      ctx.globalAlpha=.13;ctx.fillStyle=color;ctx.beginPath();ctx.ellipse(x,y,rx,ry,0,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
    }
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();
    if(stone.estimate_kind==='lost'){ctx.moveTo(x-4,y-4);ctx.lineTo(x+4,y+4);ctx.moveTo(x+4,y-4);ctx.lineTo(x-4,y+4);ctx.stroke()}
    else{ctx.arc(x,y,5,0,Math.PI*2);if(stone.visible)ctx.fill();else{ctx.setLineDash([2,2]);ctx.stroke();ctx.setLineDash([])}}
    ctx.fillStyle='#243d35';ctx.font='11px sans-serif';ctx.fillText(stone.id,x+8,y+3);
  }
  if($('#show-raw').checked)for(const o of (forwardRows[lo]?.observations||[])){const[x,y]=xy(o.xy);ctx.strokeStyle=o.accepted===false?'#788989':'#6e43a3';ctx.strokeRect(x-2,y-2,4,4)}
  ctx.font='13px sans-serif';ctx.fillStyle='#243d35';ctx.fillText('Source '+row.source_seconds.toFixed(2)+'s',15,25);drawSpeed();
}
function updateTrajectoryNotice(){
  $('#trajectory-mode').disabled=!!manifest?.raw_comparison;
  $('#trajectory-mode option[value=forward]').textContent=manifest?.raw_comparison?'Raw mask positions':'Forward filter';
  $('#map-legend').textContent=manifest?.raw_comparison?'Raw positions, without interpolation. The map prefers an overhead measurement when available; trails break at camera changes. Use the per-camera graph to compare velocity.':'Solid: measurement-supported estimate. Hollow: predicted or held. Cross: lost. Shaded ellipse: approximate 95% model uncertainty. Trails show filtered motion; raw measurements are optional. The map projects onto the ice plane.';
  if(manifest?.raw_comparison){$('#trajectory-notice').textContent='Matched camera windows; raw mask geometry. No motion gate, Kalman filter, whole-clip smoothing, or velocity interpolation. The speed graph shows completed runs from this experiment. The map selects one camera per frame.';return;}
  $('#trajectory-notice').textContent=wholeRows?`${trajectorySummary.segments.length} travel segments use future observations; ${trajectorySummary.house_anchors.filter(a=>a.kind==='overhead').length} resting positions anchored by overhead views. Button spacing: ${trajectorySummary.tee_distance_m.toFixed(3)} m. ${manifest.calibration.views.some(v=>v.camera_calibration)?'Independent camera calibration revision '+manifest.calibration.revision+' applied. Further fit audit: ':'Unapplied camera-fit candidate: '}${trajectorySummary.geometry_status.replaceAll('_',' ')}. ${trajectorySummary.warnings.some(w=>w.includes('fit failed'))?'Some segments need review.':''}`:'This run contains the forward estimate only.'+(manifest?.trajectory_error?' Refinement needs attention: '+manifest.trajectory_error:'');
}
$('#trajectory-mode').onchange=()=>{rows=$('#trajectory-mode').value==='whole'&&wholeRows?wholeRows:forwardRows;drawMap()};
$('#focus-stone').onchange=()=>drawSpeed();
function drawFilteredSpeed(c,name){
  const ctx=c.getContext('2d');
  ctx.fillStyle='#eaf3f0';ctx.fillRect(0,0,c.width,c.height);if(!manifest||!name)return;
  const x=t=>45+(t-manifest.source_start)/manifest.duration*830,y=s=>160-Math.min(6,s)/6*135;
  ctx.font='12px sans-serif';ctx.fillStyle='#35524b';ctx.fillText(name+' · speed (m/s)',45,16);
  for(const speed of [0,2,4,6]){ctx.strokeStyle='#ccd9d5';ctx.beginPath();ctx.moveTo(45,y(speed));ctx.lineTo(875,y(speed));ctx.stroke();ctx.fillText(String(speed),20,y(speed)+4)}
  for(const seconds of [0,manifest.duration/2,manifest.duration])ctx.fillText(seconds.toFixed(0)+'s',x(manifest.source_start+seconds)-8,182);
  for(const [data,color]of [[forwardRows,'#98a5a3'],[wholeRows,'#267b58']]){
    if(!data)continue;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let started=false;
    for(const row of data){const stone=row.stones.find(s=>s.id===name);if(!stone||stone.estimate_kind==='lost'){started=false;continue;}const p=[x(row.source_seconds),y(stone.speed_mps??Math.hypot(...stone.velocity))];if(started)ctx.lineTo(...p);else{ctx.moveTo(...p);started=true}}
    ctx.stroke();
  }
  ctx.lineWidth=1;ctx.strokeStyle='#357eaa';ctx.fillStyle='#245e83';ctx.setLineDash([2,4]);
  for(const event of runEvents.filter(e=>e.type==='view_join'&&e.stone===name)){
    const px=x(event.seconds);ctx.beginPath();ctx.moveTo(px,26);ctx.lineTo(px,160);ctx.stroke();
    ctx.fillText(event.view.replace('perspective_','').replace('overhead_','OH ').toUpperCase(),px+3,30);
  }
  ctx.lineWidth=1;ctx.strokeStyle='#be8d35';ctx.setLineDash([4,4]);
  for(const event of trajectorySummary?.contacts||[]){if(!event.stones.includes(name))continue;ctx.beginPath();ctx.moveTo(x(event.seconds),25);ctx.lineTo(x(event.seconds),160);ctx.stroke()}
  ctx.setLineDash([]);ctx.strokeStyle='#274d43';const cursor=x(manifest.source_start+$('#overlay').currentTime);ctx.beginPath();ctx.moveTo(cursor,25);ctx.lineTo(cursor,160);ctx.stroke();
}
function speedPanels(){
  const selected=$('#focus-stone').value,extra=$('#other-stone-speeds');
  const names=$('#all-stone-speeds').checked?[...$('#focus-stone').options].map(o=>o.value).filter(n=>n!==selected):[];
  const key=JSON.stringify(names);
  if(extra.dataset.stones!==key){
    extra.replaceChildren();extra.dataset.stones=key;
    for(const name of names){const c=document.createElement('canvas');c.width=900;c.height=190;c.style.width='100%';c.dataset.stone=name;c.setAttribute('aria-label',name+' velocity');extra.appendChild(c);}
  }
  const panels=[[$('#speed-graph'),selected],...[...extra.children].map(c=>[c,c.dataset.stone])];
  for(const [canvas,name]of panels){
    canvas.setAttribute('aria-label',name+' velocity');canvas.style.cursor='crosshair';
    canvas.title=name+' velocity — click a time to seek the video';
    canvas.onclick=e=>{if(!manifest)return;const px=(e.clientX-canvas.getBoundingClientRect().left)*canvas.width/canvas.getBoundingClientRect().width;$('#overlay').currentTime=Math.max(0,Math.min(manifest.duration,(px-45)/830*manifest.duration));drawMap();};
  }
  return panels;
}
function drawSpeed(){
  const raw=$('#speed-mode').value==='raw';
  $('#speed-view').disabled=!raw;$('#speed-units').disabled=!raw;
  for(const [canvas,name]of speedPanels())(raw?drawRawSpeed:drawFilteredSpeed)(canvas,name);
  if(!raw)$('#speed-note').textContent='Green: whole-clip fit. Grey: forward filter. Blue: camera joins (A/B perspective; OH overhead). Amber: estimated contacts. Camera calibration differences can look like sudden braking; these curves are estimates, not direct speed measurements. Click any chart to seek the video.';
}
$('#all-stone-speeds').onchange=drawSpeed;
function drawRawSpeed(canvas,stone){
  const ctx=canvas.getContext('2d'),view=$('#speed-view').value,field=$('#speed-units').value;
  ctx.fillStyle='#eaf3f0';ctx.fillRect(0,0,canvas.width,canvas.height);if(!manifest)return;
  const series=rawSets.map(set=>({...set,samples:set.samples.filter(p=>p.stone_id===stone&&p.view_id===view)}));
  const values=series.flatMap(set=>set.samples.map(p=>p[field]).filter(v=>Number.isFinite(v)));
  const maximum=Math.max(field==='pixel_speed'?1:.1,...values)*1.05;
  const x=t=>45+(t-manifest.source_start)/manifest.duration*830,y=v=>155-v/maximum*115;
  ctx.font='11px sans-serif';ctx.fillStyle='#35524b';ctx.fillText(`${stone} · ${view} · raw ${field==='pixel_speed'?'px/s':'m/s'}`,45,14);
  for(let i=0;i<=4;i++){const value=maximum*i/4;ctx.strokeStyle='#ccd9d5';ctx.beginPath();ctx.moveTo(45,y(value));ctx.lineTo(875,y(value));ctx.stroke();ctx.fillText(value.toFixed(1),5,y(value)+4);}
  for(const sec of [0,manifest.duration/2,manifest.duration])ctx.fillText(sec.toFixed(0)+'s',x(manifest.source_start+sec)-8,181);
  series.forEach((set,index)=>{const color=set.backend.startsWith('sam2')?'#2273b6':'#cf641f';ctx.fillStyle=color;ctx.fillText(set.backend==='sam2.1-mlx-granite'?'SAM2 granite body':set.backend.startsWith('sam2')?'SAM2 reinitialized':'SAM3.1 temporal',400+index*195,14);ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.beginPath();let previous=null;
    for(const p of set.samples){if(!Number.isFinite(p[field])){previous=null;continue;}const px=x(p.source_seconds),py=y(p[field]);if(previous&&p.frame===previous.frame+1)ctx.lineTo(px,py);else ctx.moveTo(px,py);previous=p;}ctx.stroke();
  });
  ctx.lineWidth=1;ctx.strokeStyle='#274d43';const cursor=x(manifest.source_start+$('#overlay').currentTime);ctx.beginPath();ctx.moveTo(cursor,30);ctx.lineTo(cursor,155);ctx.stroke();
  $('#speed-note').textContent=manifest.backend==='sam2.1-mlx-granite'?'SAM2 granite-body tracking. Perspective speed uses the visible mask’s supported bottom edge. Unverified bodies leave gaps; no temporal smoothing or velocity clipping. This diagnostic bypasses identity gates and can switch stones after occlusion.':'Blue: SAM2 with reinitialization. Orange: SAM3.1 temporal tracking. Every consecutive same-camera mask difference is retained; no smoothing, speed clipping, or gap bridging. Empty masks break the line. Metres/second includes calibration and mask-height error; compare pixels/second too.';
}
$('#speed-mode').onchange=drawSpeed;$('#speed-view').onchange=drawSpeed;$('#speed-units').onchange=drawSpeed;
$('#map-orientation').onchange=drawMap;
$('#show-raw').onchange=drawMap;
let animation;
function animateMap(){drawMap();if(!$('#overlay').paused)animation=requestAnimationFrame(animateMap)}
$('#overlay').onplay=()=>{cancelAnimationFrame(animation);animateMap()};
$('#overlay').onpause=()=>{cancelAnimationFrame(animation);drawMap()};
$('#overlay').ontimeupdate=drawMap;
(async()=>{try{record=await api('/recordings/'+encodeURIComponent(identifier));calibration=await api(base+'/calibration');$('#frame-time').value=calibration.source_seconds;$('#tee-distance').value=calibration.tee_distance_m||34.747;$('#tee-source').value=calibration.tee_distance_source||'nominal';$('#venue').value=calibration.venue_id||'';$('#fixed-length').checked=!!calibration.tee_distance_fixed;describeGeometry();$('#first-end').value=calibration.first_delivery_end;$('#end').innerHTML=record.ends.map(e=>`<option value="${e.number}">End ${e.number}</option>`).join('');$('#calibration-panel').open=!calibration.views.length;populate();loadFrame();drawMap();await poll();const requested=new URLSearchParams(location.search).get('run');if(requested)await openRun(requested);setInterval(poll,3000)}catch(e){fail(e)}})();

mountGamePanel($('#whole-game'),identifier);
$('#whole-game').addEventListener('runs-cleared',()=>{location.href='/tracking?recording='+encodeURIComponent(identifier)});
