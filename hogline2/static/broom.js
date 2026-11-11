// Broom marking: one click per shot where the skip's broom touches the ice.
const $ = s => document.querySelector(s);
const identifier = new URLSearchParams(location.search).get('recording');
const base = '/api/recordings/' + encodeURIComponent(identifier);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
let shots = [], views = [], current = 0, seconds = 0, pending = null, image = null;
$('#back').href = '/tracking?recording=' + encodeURIComponent(identifier);

const say = text => { $('#message').textContent = text || ''; };
const shot = () => shots[current];
const playingEnd = s => (s.delivery_end === 'B' ? 'A' : 'B');
const view = () => views.find(v => v.id === $('#view').value);

function renderList() {
  let html = '', end = null;
  shots.forEach((s, i) => {
    if (s.end !== end) { end = s.end; html += `<h3>End ${end}</h3>`; }
    const done = s.mark ? (s.mark.no_broom ? '–' : '✓') : '·';
    html += `<button class="secondary${i === current ? ' current' : ''}" data-i="${i}">${done} ${esc(s.stone.replace('_', ' '))} · ${esc(s.player)}</button>`;
  });
  $('#shots').innerHTML = html;
  $('#shots').querySelectorAll('button').forEach(b => { b.onclick = () => open(Number(b.dataset.i)); });
  $('#shots .current')?.scrollIntoView({block: 'nearest'});
}

function renderReport(report) {
  if (!report.length) { $('#report').innerHTML = '<p class="muted">No tracked shots yet.</p>'; return; }
  const cm = v => (v === null || v === undefined ? '—' : (v > 0 ? '+' : '') + v + ' cm');
  let html = '<table><tr><th>Player</th><th>Marked</th><th>Turn</th><th>Shots</th><th>Typical miss</th><th>Narrow (+) / wide (−)</th><th>Weight by call (hog to hog)</th></tr>';
  for (const p of report) {
    const turns = Object.entries(p.line);
    const weight = Object.entries(p.weight).map(([k, w]) => `${esc(k)}: ${w.mean_hog_to_hog_s} s${w.spread_s === null ? '' : ' ± ' + w.spread_s} (${w.shots})`).join('<br>');
    const rows = turns.length ? turns : [['—', {shots: 0, mean_abs_cm: null, mean_narrow_cm: null}]];
    rows.forEach(([name, t], i) => {
      html += '<tr>' + (i ? '' : `<td rowspan="${rows.length}">${esc(p.player)}</td><td rowspan="${rows.length}">${p.marked} of ${p.shots}</td>`)
        + `<td>${esc(name)}</td><td>${t.shots}</td><td>${t.mean_abs_cm === null ? '—' : t.mean_abs_cm + ' cm'}</td><td>${cm(t.mean_narrow_cm)}</td>`
        + (i ? '' : `<td rowspan="${rows.length}">${weight || '—'}</td>`) + '</tr>';
    });
  }
  $('#report').innerHTML = html + '</table>';
}

function describe(s) {
  const a = s.aim, parts = [];
  if (s.hog_to_hog_s) parts.push(`hog to hog ${s.hog_to_hog_s} s`);
  if (a?.turn) parts.push(`${a.turn}, curled ${Math.abs(a.curl_right_m * 100).toFixed(0)} cm`);
  if (a && a.miss_right_cm !== undefined) {
    parts.push(`line ${Math.abs(a.miss_right_cm)} cm ${a.miss_right_cm >= 0 ? 'right' : 'left'} of the broom`
      + (a.miss_narrow_cm !== undefined ? ` (${a.miss_narrow_cm >= 0 ? 'narrow' : 'wide'})` : ''));
  } else if (s.mark?.no_broom) parts.push('no broom visible');
  else if (!a) parts.push('path after release too thin for a line');
  return parts.join(' · ');
}

function draw() {
  const canvas = $('#frame'), ctx = canvas.getContext('2d'), v = view();
  if (!image || !v) return;
  const [x0, y0, x1, y1] = v.crop;
  canvas.width = 1280; canvas.height = Math.round(1280 * (y1 - y0) / (x1 - x0));
  ctx.drawImage(image, x0, y0, x1 - x0, y1 - y0, 0, 0, canvas.width, canvas.height);
  const k = canvas.width / (x1 - x0);
  const cross = (px, colour) => {
    if (!px || px[0] < x0 || px[0] >= x1 || px[1] < y0 || px[1] >= y1) return;
    const cx = (px[0] - x0) * k, cy = (px[1] - y0) * k;
    ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(cx - 14, cy); ctx.lineTo(cx + 14, cy); ctx.moveTo(cx, cy - 14); ctx.lineTo(cx, cy + 14); ctx.stroke();
  };
  cross(shot().mark?.px, '#00d0ff');
  cross(pending, '#ff2bd6');
}

function loadFrame() {
  $('#clock').textContent = seconds.toFixed(1) + ' s';
  const next = new Image();
  next.onload = () => { image = next; draw(); };
  next.onerror = () => say('That frame could not be read from the source video.');
  next.src = base + '/tracking/frame?seconds=' + seconds.toFixed(2);
}

function open(i) {
  current = Math.max(0, Math.min(shots.length - 1, i));
  const s = shot();
  pending = null;
  seconds = s.mark?.seconds ?? Math.max(0, s.delivered_at - 3);
  const options = views.filter(v => v.end === playingEnd(s));
  $('#view').innerHTML = options.map(v => `<option value="${esc(v.id)}">${v.kind === 'overhead' ? 'Overhead (house only)' : 'End view (whole far half, guards included)'} · house ${esc(v.end)}</option>`).join('');
  $('#view').value = s.mark?.view ?? (options.find(v => v.kind === 'perspective') ?? options[0])?.id;
  $('#call').value = s.mark?.call ?? '';
  $('#title').textContent = `End ${s.end} · ${s.stone.replace('_', ' ')} · ${s.player}`;
  $('#result').textContent = describe(s);
  renderList();
  loadFrame();
}

async function save(noBroom) {
  const s = shot();
  if (!noBroom && !pending && !s.mark?.px) { say('Click the broom first, or choose No broom visible.'); return; }
  const body = {seconds, call: $('#call').value || null, no_broom: !!noBroom, px: noBroom ? null : (pending ?? s.mark.px)};
  try {
    const response = await fetch(`${base}/broom/${s.end}/${encodeURIComponent(s.stone)}`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const value = await response.json();
    if (!response.ok) throw Error(value.detail || 'Could not save the mark');
    shots[current] = value.shot;
    renderReport(value.report);
    say('');
    open(current + (current < shots.length - 1 ? 1 : 0));
  } catch (error) { say(error.message); }
}

$('#frame').onclick = e => {
  const v = view(), rect = e.target.getBoundingClientRect();
  pending = [v.crop[0] + (e.clientX - rect.left) / rect.width * (v.crop[2] - v.crop[0]),
             v.crop[1] + (e.clientY - rect.top) / rect.height * (v.crop[3] - v.crop[1])];
  draw();
};
$('#view').onchange = draw;
document.querySelectorAll('[data-step]').forEach(b => { b.onclick = () => { seconds = Math.max(0, seconds + Number(b.dataset.step)); loadFrame(); }; });
$('#save').onclick = () => save(false);
$('#none').onclick = () => save(true);
document.onkeydown = e => {
  if (['SELECT', 'INPUT'].includes(document.activeElement?.tagName)) return;
  if (e.key === 'ArrowRight') open(current + 1);
  else if (e.key === 'ArrowLeft') open(current - 1);
  else if (e.key === '.') { seconds += 0.2; loadFrame(); }
  else if (e.key === ',') { seconds = Math.max(0, seconds - 0.2); loadFrame(); }
  else if (e.key === 'Enter') save(false);
  else if (e.key === 'v') { const o = $('#view'); o.selectedIndex = (o.selectedIndex + 1) % o.options.length; draw(); }
};

(async () => {
  try {
    const response = await fetch(base + '/broom');
    const value = await response.json();
    if (!response.ok) throw Error(value.detail || 'Could not load the shots');
    shots = value.shots; views = value.views;
    $('#call').innerHTML += value.calls.map(c => `<option>${esc(c)}</option>`).join('');
    renderReport(value.report);
    if (!shots.length) { $('#title').textContent = 'No tracked ends yet'; say('Track an end with stonetrack first; its shots then appear here.'); return; }
    const first = shots.findIndex(s => !s.mark);
    open(first < 0 ? 0 : first);
  } catch (error) { say(error.message); }
})();
