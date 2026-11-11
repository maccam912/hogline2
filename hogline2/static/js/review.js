// Review shots: watch each delivery, click the skip's broom on the paused video, label the call.
//
// One <video> plays the original broadcast. It is cropped to the camera that matters for the
// current shot, and a canvas on top draws the tracked stones and the broom mark. Clicking the
// picture marks the broom; everything saves as it is changed.
import {api, attempt, clock, esc, link, replaceParams, toast} from './util.js';
import {
  BACK, HACK, HOG, STONE_COLOURS, STONE_RADIUS, TEE, cameraAt, drawBroom, drawPath, drawSheet, drawStone,
  houseCrop, pixelsPerMetre, positionAt, sheetView, stoneInImage, toImage,
} from './sheet.js';

const VIEWS = [
  ['target', 'Target house', 'the camera looking at the house this stone is thrown at'],
  ['overhead', 'Overhead', 'straight down on the target house'],
  ['thrower', 'Thrower', 'the delivery'],
  ['all', 'Whole broadcast', 'all four cameras'],
];
const other = end => (end === 'A' ? 'B' : 'A');
const title = shot => `${shot.colour === 'red' ? 'Red' : 'Yellow'} ${shot.label.slice(1)}`;

function lineText(shot) {
  const aim = shot.aim, mark = shot.mark;
  if (mark?.no_broom) return 'No broom visible';
  if (!aim) return 'Path too short to measure';
  if (aim.miss_right_cm === undefined) return 'Mark the broom to see the line';
  const size = Math.abs(aim.miss_right_cm);
  if (size < 0.5) return 'On the broom';
  const side = aim.miss_right_cm > 0 ? 'right' : 'left';
  const kind = aim.miss_narrow_cm === undefined ? '' : aim.miss_narrow_cm >= 0 ? ' · narrow' : ' · wide';
  return `${size.toFixed(0)} cm ${side} of the broom${kind}`;
}

function turnText(shot) {
  const aim = shot.aim;
  if (!aim) return '–';
  if (!aim.turn) return 'Straight (under 5 cm of curl)';
  return `${aim.turn === 'clockwise' ? 'Clockwise' : 'Counter-clockwise'}, curled ${Math.abs(aim.curl_right_m * 100).toFixed(0)} cm`;
}

export function mount(container, {id, record, params}) {
  const base = `/recordings/${encodeURIComponent(id)}`;
  const frame = (() => { const [n, d = 1] = String(record.media?.fps || '30').split('/').map(Number); return d / n; })();
  const mosaic = [0, 0, record.media?.width || 1920, record.media?.height || 1080];
  const paths = new Map(); // end number -> stones with compact paths
  let data = null, index = 0, viewMode = 'target', labels = true, raf = 0, disposed = false, scrubbing = false;

  container.innerHTML = `
    <aside class="shots">
      <div class="shots-head">
        <b id="progress"></b>
        <button id="next-open" class="quiet" title="Enter">Next unmarked</button>
      </div>
      <nav id="shot-list" aria-label="Shots"></nav>
    </aside>
    <section class="stage">
      <div class="row between">
        <div class="segmented" id="views" role="group" aria-label="Camera">
          ${VIEWS.map(([key, label, what]) => `<button data-view="${key}" title="${label}: ${what}">${label}</button>`).join('')}
        </div>
        <label class="check"><input type="checkbox" id="labels" checked> Stone labels</label>
      </div>
      <div class="screen-wrap"><div class="screen" id="screen">
        <video id="video" preload="auto" playsinline></video>
        <canvas id="overlay" aria-label="Video. Click where the skip's broom touches the ice."></canvas>
      </div></div>
      <div class="transport">
        <button id="back1" class="quiet" title="Back 1 second (←)">−1 s</button>
        <button id="backf" class="quiet" title="Back one frame (,)">−1 frame</button>
        <button id="play" title="Play / pause (space)">Play</button>
        <button id="fwdf" class="quiet" title="Forward one frame (.)">+1 frame</button>
        <button id="fwd1" class="quiet" title="Forward 1 second (→)">+1 s</button>
        <input id="scrub" type="range" step="0.01" aria-label="Position in this shot">
        <span id="time" class="mono"></span>
        <select id="rate" aria-label="Playback speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
      </div>
      <canvas id="strip" class="strip" aria-label="Top-down view of the whole sheet"></canvas>
      <p class="muted keys">
        <kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> 1 s · <kbd>,</kbd><kbd>.</kbd> frame ·
        <kbd>↑</kbd><kbd>↓</kbd> shot · <kbd>enter</kbd> next unmarked · <kbd>1</kbd>–<kbd>4</kbd> call ·
        <kbd>x</kbd> no broom · <kbd>v</kbd> camera
      </p>
    </section>
    <aside class="facts" id="facts"></aside>`;
  const $ = selector => container.querySelector(selector);
  const video = $('#video'), overlay = $('#overlay'), screen = $('#screen'), scrub = $('#scrub');
  video.src = `/api${base}/source`;

  const shot = () => data.shots[index];
  const endOf = s => data.ends.find(e => e.number === s.end);
  const camera = (kind, end) => data.cameras.find(c => c.kind === kind && c.end === end);

  /** The time span the scrubber covers for a shot: a little before the delivery to just after it stops. */
  function span(s) {
    const end = endOf(s);
    let from = Math.max(end.start_seconds, s.delivered_at - 12);
    let to = Math.min(end.end_seconds, Math.max(s.stopped_at, s.delivered_at + 8) + 4);
    if (s.mark?.seconds !== undefined) { from = Math.min(from, s.mark.seconds); to = Math.max(to, s.mark.seconds); }
    return [from, to];
  }

  function crop() {
    const s = shot(), playing = other(s.delivery_end);
    const target = camera('perspective', playing), above = camera('overhead', playing);
    const thrower = camera('perspective', s.delivery_end);
    if (viewMode === 'target' && target) return houseCrop(target);
    if (viewMode === 'overhead' && above) return above.crop;
    if (viewMode === 'thrower' && thrower) return houseCrop(thrower);
    return mosaic;
  }

  // ---- layout -------------------------------------------------------------------------------

  function layout() {
    const [x0, y0, x1, y1] = crop(), cw = x1 - x0, ch = y1 - y0;
    const wrap = screen.parentElement, maxW = wrap.clientWidth, maxH = Math.max(240, window.innerHeight - 330);
    const scale = Math.min(maxW / cw, maxH / ch);
    const w = Math.round(cw * scale), h = Math.round(ch * scale);
    screen.style.width = w + 'px';
    screen.style.height = h + 'px';
    video.style.width = mosaic[2] * scale + 'px';
    video.style.height = mosaic[3] * scale + 'px';
    video.style.left = -x0 * scale + 'px';
    video.style.top = -y0 * scale + 'px';
    const ratio = window.devicePixelRatio || 1;
    overlay.width = w * ratio; overlay.height = h * ratio;
    const strip = $('#strip');
    strip.width = strip.clientWidth * ratio; strip.height = strip.clientHeight * ratio;
    const house = $('#house');
    if (house) { house.width = house.clientWidth * ratio; house.height = house.clientHeight * ratio; }
  }

  // ---- drawing ------------------------------------------------------------------------------

  function drawOverlay(seconds) {
    const ctx = overlay.getContext('2d'), s = shot();
    const [x0, y0, x1] = crop(), k = overlay.width / (x1 - x0);
    const place = ([x, y]) => [(x - x0) * k, (y - y0) * k];
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const line = k * 1.4;
    for (const cam of data.cameras) {
      const [cx0, cy0, cx1, cy1] = cam.crop;
      ctx.save();
      ctx.beginPath();
      ctx.rect(...place([cx0, cy0]), (cx1 - cx0) * k, (cy1 - cy0) * k);
      ctx.clip();
      if (s.aim?.heading && s.mark?.px && cam.end === other(s.delivery_end)) {
        ctx.setLineDash([8 * line, 6 * line]);
        ctx.strokeStyle = 'rgba(255,170,40,.9)'; ctx.lineWidth = 2 * line;
        ctx.beginPath();
        ctx.moveTo(...place(toImage(cam, s.aim.heading[0])));
        ctx.lineTo(...place(toImage(cam, s.aim.heading[1])));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (labels) {
        for (const stone of paths.get(s.end) || []) {
          const at = positionAt(stone, seconds);
          if (!at) continue;
          const [px, py] = place(stoneInImage(cam, [at.x, at.y]));
          const radius = Math.max(7, STONE_RADIUS * pixelsPerMetre(cam, [at.x, at.y])) * k + 2 * line;
          const current = stone.stone === s.stone;
          ctx.setLineDash(at.hidden ? [4 * line, 4 * line] : []);
          for (const [colour, width] of [['rgba(0,0,0,.75)', current ? 5 : 3.5], [STONE_COLOURS[stone.colour], current ? 3 : 1.5]]) {
            ctx.strokeStyle = colour; ctx.lineWidth = width * line;
            ctx.beginPath(); ctx.arc(px, py, radius, 0, 2 * Math.PI); ctx.stroke();
          }
          ctx.setLineDash([]);
          ctx.font = `600 ${Math.round(Math.max(11, 12 * line))}px system-ui, sans-serif`;
          ctx.textBaseline = 'bottom';
          ctx.lineWidth = 3 * line; ctx.strokeStyle = 'rgba(0,0,0,.8)';
          ctx.strokeText(stone.label, px + radius * 0.8, py - radius * 0.7);
          ctx.fillStyle = current ? '#fff' : 'rgba(255,255,255,.8)';
          ctx.fillText(stone.label, px + radius * 0.8, py - radius * 0.7);
        }
      }
      ctx.restore();
    }
    if (s.mark?.px) drawBroom(ctx, place(s.mark.px), '#00e0ff', 13 * (window.devicePixelRatio || 1));
  }

  function drawMap(canvas, range, sideways, seconds) {
    if (!canvas?.width) return;
    const ctx = canvas.getContext('2d'), s = shot(), pad = 6 * (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const view = sheetView({x: pad, y: pad, w: canvas.width - 2 * pad, h: canvas.height - 2 * pad}, s.delivery_end, range, sideways);
    drawSheet(ctx, view);
    const stones = paths.get(s.end) || [], current = stones.find(st => st.stone === s.stone);
    if (s.aim?.heading) {
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = '#e8891c'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(...view.fromIce(s.aim.heading[0])); ctx.lineTo(...view.fromIce(s.aim.heading[1])); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (current) drawPath(ctx, view, current, s.delivered_at - 1, Math.max(seconds, s.delivered_at), '#14181a');
    for (const stone of stones) {
      const at = positionAt(stone, seconds);
      if (at) drawStone(ctx, view, at, stone.colour, {highlight: stone.stone === s.stone, hidden: at.hidden, label: stone.label.slice(1)});
    }
    if (s.mark?.ice) drawBroom(ctx, view.fromIce(s.mark.ice), '#00a8e0', 8 * (window.devicePixelRatio || 1));
  }

  function tick() {
    if (disposed) return;
    raf = requestAnimationFrame(tick);
    if (!data?.shots.length) return;
    const seconds = video.currentTime, [from, to] = span(shot());
    if (!video.paused && seconds >= to) video.pause();
    if (!scrubbing) scrub.value = seconds;
    const offset = seconds - shot().delivered_at;
    $('#time').textContent = `${clock(seconds)}  (${offset >= 0 ? '+' : '−'}${Math.abs(offset).toFixed(1)} s)`;
    $('#play').textContent = video.paused ? 'Play' : 'Pause';
    drawOverlay(seconds);
    drawMap($('#strip'), [-HACK - 0.6, TEE + BACK + 0.6], true, seconds);
    drawMap($('#house'), [TEE - HOG - 1.2, TEE + BACK + 0.5], false, seconds);
    scrub.min = Math.min(from, seconds); scrub.max = Math.max(to, seconds);
  }

  // ---- panels -------------------------------------------------------------------------------

  function drawList() {
    const marked = data.shots.filter(s => s.mark?.px || s.mark?.no_broom).length;
    $('#progress').textContent = `${marked} of ${data.shots.length} brooms marked`;
    let html = '', end = null;
    data.shots.forEach((s, i) => {
      if (s.end !== end) { end = s.end; html += `<h3>End ${end}</h3>`; }
      const state = s.mark?.px ? ['done', 'Broom marked', '✓'] : s.mark?.no_broom ? ['skip', 'No broom visible', '–'] : ['', 'Not marked', ''];
      html += `<button class="shot${i === index ? ' current' : ''}" data-i="${i}">
        <span class="dot ${s.colour}"></span><span class="n">${s.throw}</span>
        <span class="who">${esc(s.player)}</span>
        <span class="t">${s.hog_to_hog_s ? s.hog_to_hog_s.toFixed(1) : ''}</span>
        <span class="state ${state[0]}" title="${state[1]}">${state[2]}</span></button>`;
    });
    $('#shot-list').innerHTML = html;
    $('#shot-list').querySelectorAll('.shot').forEach(b => { b.onclick = () => open(Number(b.dataset.i)); });
    $('#shot-list .current')?.scrollIntoView({block: 'nearest'});
  }

  function drawFacts() {
    const s = shot(), mark = s.mark;
    const row = (label, value) => `<div><dt>${label}</dt><dd>${value}</dd></div>`;
    $('#facts').innerHTML = `
      <p class="muted">End ${s.end} · throw ${s.throw} of ${data.shots.filter(o => o.end === s.end).length}</p>
      <h2><span class="dot ${s.colour}"></span>${title(s)} · ${esc(s.player)}</h2>
      <dl>
        ${row('Line', `<b>${esc(lineText(s))}</b>`)}
        ${row('Turn', esc(turnText(s)))}
        ${row('Hog to hog', s.hog_to_hog_s ? s.hog_to_hog_s.toFixed(2) + ' s' : '–')}
        ${row('Tee to hog', s.tee_to_hog_s ? s.tee_to_hog_s.toFixed(2) + ' s' : '–')}
        ${row('Stone', esc(s.outcome === 'in play' ? 'Stayed in play' : s.outcome || '–'))}
      </dl>
      <h3>Broom</h3>
      <p class="${mark?.px ? '' : 'callout'}">${mark?.px ? `Marked at ${clock(mark.seconds)}. Click the video again to move it.`
        : mark?.no_broom ? 'Marked as not visible.'
        : 'Pause when the skip’s broom is down, then <b>click where it touches the ice</b>.'}</p>
      <div class="row">
        ${mark?.px ? '<button id="goto" class="quiet">Show it</button><button id="clear" class="quiet">Remove</button>' : ''}
        <button id="nobroom" class="quiet${mark?.no_broom ? ' on' : ''}" aria-pressed="${!!mark?.no_broom}" title="x">No broom visible</button>
      </div>
      <h3>Called shot</h3>
      <div class="segmented" id="calls">
        ${data.calls.map((c, i) => `<button data-call="${c}" class="${mark?.call === c ? 'on' : ''}" aria-pressed="${mark?.call === c}" title="${i + 1}">${c[0].toUpperCase() + c.slice(1)}</button>`).join('')}
      </div>
      <canvas id="house" class="house" aria-label="Top-down view of the target house"></canvas>
      <p class="muted legend"><span class="swatch heading"></span>heading at release <span class="swatch broom">+</span>broom</p>
      <div class="row between">
        <button id="prev" class="quiet" ${index ? '' : 'disabled'}>← Previous</button>
        <button id="next" class="quiet" ${index < data.shots.length - 1 ? '' : 'disabled'}>Next →</button>
      </div>`;
    $('#goto')?.addEventListener('click', () => { video.pause(); video.currentTime = mark.seconds; });
    $('#clear')?.addEventListener('click', () => save({px: null, seconds: null, no_broom: false}));
    $('#nobroom').onclick = () => save({no_broom: !mark?.no_broom});
    $('#calls').querySelectorAll('button').forEach(b => { b.onclick = () => setCall(b.dataset.call); });
    $('#prev').onclick = () => open(index - 1);
    $('#next').onclick = () => open(index + 1);
    layout();
  }

  function drawViews() {
    $('#views').querySelectorAll('button').forEach(b => {
      b.classList.toggle('on', b.dataset.view === viewMode);
      b.setAttribute('aria-pressed', b.dataset.view === viewMode);
    });
  }

  // ---- actions ------------------------------------------------------------------------------

  async function open(i) {
    if (!data.shots.length) return;
    index = Math.max(0, Math.min(data.shots.length - 1, i));
    const s = shot();
    replaceParams(id, 'review', {end: s.end, shot: s.stone});
    video.pause();
    video.currentTime = s.mark?.seconds ?? Math.max(endOf(s).start_seconds, s.delivered_at - 3);
    if (s.mark?.view) viewMode = data.cameras.find(c => c.id === s.mark.view)?.kind === 'overhead' ? 'overhead' : 'target';
    else if (viewMode !== 'overhead') viewMode = 'target';
    drawViews(); drawList(); drawFacts();
    if (!paths.has(s.end)) {
      const loaded = await attempt(() => api(`${base}/ends/${s.end}/paths`));
      if (loaded) paths.set(s.end, loaded.stones);
    }
  }

  /** Save a change to the current shot's mark; `changes` overrides the saved fields. */
  async function save(changes) {
    const s = shot(), at = index, old = s.mark || {};
    const body = {seconds: old.seconds ?? null, px: old.px ?? null, call: old.call ?? null, no_broom: !!old.no_broom, ...changes};
    if (body.no_broom) { body.px = null; body.seconds = null; }
    await attempt(async () => {
      const saved = await api(`${base}/shots/${s.end}/${s.stone}`, 'PUT', body);
      data.shots[at] = saved.shot;
      if (at === index) { drawList(); drawFacts(); }
      if (changes.px) toast(`Broom saved · ${lineText(saved.shot)}`);
    });
  }

  const setCall = call => save({call: shot().mark?.call === call ? null : call});

  function markBroom(event) {
    const s = shot(), [x0, y0, x1, y1] = crop(), box = overlay.getBoundingClientRect();
    const px = [x0 + (event.clientX - box.left) / box.width * (x1 - x0), y0 + (event.clientY - box.top) / box.height * (y1 - y0)];
    const cam = cameraAt(data.cameras, px), playing = other(s.delivery_end);
    if (!cam) { toast('That is outside the camera views', 'error'); return; }
    if (cam.end !== playing) { toast('That camera shows the other house. Click the broom in the house this stone is thrown at.', 'error'); return; }
    video.pause();
    save({px, seconds: video.currentTime, no_broom: false});
  }

  function nextUnmarked() {
    const open_ = i => !(data.shots[i].mark?.px || data.shots[i].mark?.no_broom);
    const order = [...data.shots.keys()].slice(index + 1).concat([...data.shots.keys()].slice(0, index + 1));
    const found = order.find(open_);
    if (found === undefined) toast('Every shot has a broom mark');
    else open(found);
  }

  const step = seconds => { video.pause(); video.currentTime = Math.max(0, video.currentTime + seconds); };
  const toggle = () => { if (video.paused) { if (video.currentTime >= span(shot())[1] - 0.05) video.currentTime = span(shot())[0]; video.play(); } else video.pause(); };
  const setView = mode => { viewMode = mode; drawViews(); layout(); };

  function onKey(event) {
    if (!data?.shots.length || event.metaKey || event.ctrlKey || event.altKey) return;
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName) && document.activeElement.type !== 'range' && document.activeElement.type !== 'checkbox') return;
    const calls = data.calls, key = event.key;
    const actions = {
      ' ': toggle, ArrowLeft: () => step(-1), ArrowRight: () => step(1), ',': () => step(-frame), '.': () => step(frame),
      ArrowUp: () => open(index - 1), ArrowDown: () => open(index + 1), Enter: nextUnmarked,
      x: () => save({no_broom: !shot().mark?.no_broom}),
      v: () => setView(VIEWS[(VIEWS.findIndex(v => v[0] === viewMode) + 1) % VIEWS.length][0]),
    };
    const action = actions[key] || (calls[Number(key) - 1] && (() => setCall(calls[Number(key) - 1])));
    if (!action) return;
    event.preventDefault();
    document.activeElement?.blur?.();
    action();
  }

  // ---- wiring -------------------------------------------------------------------------------

  overlay.onclick = markBroom;
  $('#views').querySelectorAll('button').forEach(b => { b.onclick = () => setView(b.dataset.view); });
  $('#labels').onchange = event => { labels = event.target.checked; };
  $('#play').onclick = toggle;
  $('#back1').onclick = () => step(-1);
  $('#fwd1').onclick = () => step(1);
  $('#backf').onclick = () => step(-frame);
  $('#fwdf').onclick = () => step(frame);
  $('#rate').onchange = event => { video.playbackRate = Number(event.target.value); };
  $('#next-open').onclick = nextUnmarked;
  scrub.oninput = () => { scrubbing = true; video.currentTime = Number(scrub.value); };
  scrub.onchange = () => { scrubbing = false; };
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', layout);

  (async () => {
    try {
      data = await api(`${base}/shots`);
    } catch (error) { toast(error.message, 'error'); return; }
    if (disposed) return;
    if (!data.shots.length) {
      container.className = 'page';
      container.innerHTML = `<div class="empty"><h2>No tracked shots yet</h2><p>Track the stones first; every delivery then shows up here to review.</p><p><a class="button" href="${link(id, 'tracking')}">Go to tracking</a></p></div>`;
      return;
    }
    let start = data.shots.findIndex(s => String(s.end) === params.end && (!params.shot || s.stone === params.shot));
    if (start < 0) start = data.shots.findIndex(s => !(s.mark?.px || s.mark?.no_broom));
    await open(Math.max(0, start));
    tick();
  })();

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('resize', layout);
    video.pause();
    video.removeAttribute('src');
    video.load();
  };
}
