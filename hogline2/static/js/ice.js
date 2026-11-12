// What the ice did: speed, curl and falls, read from every free glide of the game.
import {api, attempt, esc, link} from './util.js';
import {BACK, FT, HALF_WIDTH, HOG, TEE} from './sheet.js';

const COOL = [78, 160, 235], WARM = [232, 110, 84], NEUTRAL = [52, 62, 67];
const GRADE_ALPHA = {clear: 1, likely: 0.75, faint: 0.3};
const mix = (a, b, f) => a.map((v, i) => Math.round(v + (b[i] - v) * f));
const rgb = c => `rgb(${c[0]},${c[1]},${c[2]})`;
const feet = m => { const inches = Math.round(m / FT * 12); return `${Math.floor(inches / 12)}′${inches % 12}″`; };
const num = (v, digits = 1) => v.toFixed(digits).replace('-', '−');

/** Diverging colour: neutral at zero, `high` hue for positive, `low` for negative, saturating at `full`. */
function diverging(value, full, grade, high = WARM, low = COOL) {
  const f = Math.min(1, Math.abs(value) / full) * GRADE_ALPHA[grade];
  return rgb(mix(NEUTRAL, value >= 0 ? high : low, f));
}

// Each map: which field, how to colour it and how to say one cell in words.
const MAPS = {
  fall: {
    title: 'Falls',
    lead: 'Sideways pull that does not depend on the turn: a slope or a run. Arrows point the way a stone is pushed; longer is stronger.',
    full: 1.5,
  },
  curl: {
    title: 'Curl',
    lead: 'How much harder or softer the ice turns a stone here than on the sheet as a whole.',
    full: 30,
    legend: ['straighter', 'swingier'],
  },
  speed: {
    title: 'Speed',
    lead: 'How much more or less the ice brakes a stone here than on the sheet as a whole.',
    full: 15,
    legend: ['faster (keener)', 'slower (heavier)'],
  },
};

function verdict(evidence, cells) {
  const flagged = cells.filter(c => c && c.grade !== 'faint').length;
  const detail = `Shots left out of the fit were predicted better with this map than without it ${evidence.better_shots} times out of ${evidence.shots}; their error went from ${evidence.uniform_cm.toFixed(2)} to ${evidence.mapped_cm.toFixed(2)} cm.`;
  if (evidence.real) return {level: 'ok', label: 'Real', text: `The game’s shots support this map. ${detail}`};
  if (flagged) return {level: 'warn', label: 'Hints only', text: `The map as a whole does not predict left-out shots better than uniform ice, but ${flagged} zone${flagged === 1 ? ' stands' : 's stand'} out. Treat ${flagged === 1 ? 'it' : 'them'} as something to watch next game. ${detail}`};
  return {level: '', label: 'No evidence', text: `Nothing here that these shots can tell from uniform ice. ${detail}`};
}

/** Geometry of the sideways sheet: house `left` on the left, so `x` up or down depends on the viewpoint. */
function layout(canvas, view) {
  const ratio = window.devicePixelRatio || 1, width = canvas.clientWidth, pad = 6;
  const length = TEE + 2 * BACK, scale = (width - 2 * pad) / length;
  const height = Math.round(2 * HALF_WIDTH * scale + 2 * pad);
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  // from A: A on the left, a thrower at A looks right, so their right hand (−X) is at the bottom
  const flip = view === 'B' ? -1 : 1;
  const px = y => pad + (flip === 1 ? y + BACK : TEE + BACK - y) * scale;
  const py = x => height / 2 - flip * x * scale;
  return {ctx, scale, px, py, flip, width, height};
}

function cellBoxes(grid, g) {
  const edges = knots => knots.map((k, i) => [i ? (knots[i - 1] + k) / 2 : null, i < knots.length - 1 ? (k + knots[i + 1]) / 2 : null]);
  const along = edges(grid.along).map(([lo, hi]) => [lo ?? -BACK, hi ?? TEE + BACK]);
  const across = edges(grid.across).map(([lo, hi]) => [lo ?? -HALF_WIDTH, hi ?? HALF_WIDTH]);
  const boxes = [];
  along.forEach(([y0, y1], i) => across.forEach(([x0, x1], j) => {
    const xa = g.px(y0), xb = g.px(y1), ya = g.py(x0), yb = g.py(x1);
    boxes.push({i, j, x: Math.min(xa, xb), y: Math.min(ya, yb), w: Math.abs(xb - xa), h: Math.abs(yb - ya), cx: g.px(grid.along[i]), cy: g.py(grid.across[j])});
  }));
  return boxes;
}

function drawMarkings(g) {
  const {ctx} = g;
  ctx.strokeStyle = 'rgba(233,239,239,.28)';
  ctx.lineWidth = 1;
  const line = (y0, x0, y1, x1) => { ctx.beginPath(); ctx.moveTo(g.px(y0), g.py(x0)); ctx.lineTo(g.px(y1), g.py(x1)); ctx.stroke(); };
  line(-BACK, 0, TEE + BACK, 0);
  for (const tee of [0, TEE]) {
    line(tee, -HALF_WIDTH, tee, HALF_WIDTH);
    for (const r of [6 * FT, 4 * FT, 2 * FT]) { ctx.beginPath(); ctx.arc(g.px(tee), g.py(0), r * g.scale, 0, 2 * Math.PI); ctx.stroke(); }
  }
  ctx.strokeStyle = 'rgba(229,72,77,.55)';
  ctx.lineWidth = 2;
  line(HOG, -HALF_WIDTH, HOG, HALF_WIDTH);
  line(TEE - HOG, -HALF_WIDTH, TEE - HOG, HALF_WIDTH);
}

function drawArrow(ctx, x, y, dy, colour) {
  const head = Math.min(6, Math.abs(dy) * 0.6), s = Math.sign(dy);
  ctx.strokeStyle = ctx.fillStyle = colour;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y - dy / 2); ctx.lineTo(x, y + dy / 2 - s * head); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y + dy / 2); ctx.lineTo(x - head * 0.7, y + dy / 2 - s * head); ctx.lineTo(x + head * 0.7, y + dy / 2 - s * head); ctx.closePath(); ctx.fill();
}

function drawMap(canvas, key, data, view, showGlides) {
  const g = layout(canvas, view), {ctx} = g, spec = MAPS[key];
  const boxes = cellBoxes(data.grid, g), cells = data[key].cells;
  ctx.fillStyle = '#12181b';
  ctx.fillRect(0, 0, g.width, g.height);
  boxes.forEach((b, k) => {
    const c = cells[k];
    if (!c) { // too few stones came through to say anything
      ctx.fillStyle = '#12181b'; ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.strokeStyle = 'rgba(141,159,165,.18)'; ctx.lineWidth = 1;
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip(); ctx.beginPath();
      for (let d = -b.h; d < b.w; d += 8) { ctx.moveTo(b.x + d, b.y + b.h); ctx.lineTo(b.x + d + b.h, b.y); }
      ctx.stroke(); ctx.restore();
      return;
    }
    ctx.fillStyle = key === 'fall' ? rgb(mix(NEUTRAL, [120, 132, 138], Math.min(1, Math.abs(c.value) / spec.full) * GRADE_ALPHA[c.grade] * 0.6))
      : diverging(c.value, spec.full, c.grade);
    ctx.fillRect(b.x + 1, b.y + 1, b.w - 2, b.h - 2); // a 2px gap between neighbouring fills
  });
  if (showGlides) {
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(233,239,239,.09)';
    for (const glide of data.glides) {
      ctx.beginPath();
      glide.points.forEach(([x, y], i) => (i ? ctx.lineTo(g.px(y), g.py(x)) : ctx.moveTo(g.px(y), g.py(x))));
      ctx.stroke();
    }
  }
  drawMarkings(g);
  ctx.font = '600 11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  boxes.forEach((b, k) => {
    const c = cells[k];
    if (!c) return;
    if (key === 'fall') {
      const lengthPx = Math.min(b.h * 0.9, Math.abs(c.value) / spec.full * b.h * 0.9);
      if (lengthPx < 3) { ctx.fillStyle = 'rgba(233,239,239,.35)'; ctx.beginPath(); ctx.arc(b.cx, b.cy, 1.5, 0, 2 * Math.PI); ctx.fill(); return; }
      // +X is drawn upward when looking from A
      drawArrow(ctx, b.cx, b.cy, -g.flip * Math.sign(c.value) * lengthPx, `rgba(233,239,239,${c.grade === 'faint' ? 0.4 : 1})`);
    } else if (c.grade !== 'faint') {
      ctx.fillStyle = '#0b1113';
      ctx.fillText(`${c.value > 0 ? '+' : '−'}${Math.abs(c.value).toFixed(0)}%`, Math.max(24, Math.min(g.width - 24, b.cx)), b.cy);
    }
  });
  return boxes;
}

/** Words for a place on the sheet, from the chosen viewpoint. */
function place(data, i, j, view) {
  const y = data.grid.along[i], x = data.grid.across[j];
  const side = x === 0 ? 'centre' : (x > 0) === (view === 'A') ? 'left side' : 'right side';
  const zone = y < 2 ? 'house A' : y < HOG + 1 ? 'in front of house A' : y > TEE - 2 ? 'house B' : y > TEE - HOG - 1 ? 'in front of house B'
    : Math.abs(y - TEE / 2) < 3 ? 'mid-sheet' : y < TEE / 2 ? 'A side of mid-sheet' : 'B side of mid-sheet';
  return `${zone}, ${side}`;
}

function cellWords(key, c, data, i, j, view) {
  const where = place(data, i, j, view);
  if (!c) return `<b>${esc(where)}</b><br>Too few stones came through here to say.`;
  const sure = {clear: 'clear', likely: 'likely', faint: 'not distinguishable from nothing'}[c.grade];
  let what;
  if (key === 'fall') {
    const toward = (c.value > 0) === (view === 'A') ? 'left' : 'right';
    what = `Pushes stones to the <b>${toward}</b>: ${num(Math.abs(c.value), 2)} ± ${num(c.sd, 2)} mm/s²<br><span class="muted">a slope of about ${num(Math.abs(c.value) / 9.81, 2)} mm per metre</span>`;
  } else if (key === 'curl') {
    what = `<b>${num(Math.abs(c.value), 0)}% ${c.value > 0 ? 'more' : 'less'} curl</b> than the sheet average (± ${num(c.sd, 0)})`;
  } else {
    what = `<b>${num(Math.abs(c.value), 0)}% ${c.value > 0 ? 'slower' : 'faster'}</b> than the sheet average (± ${num(c.sd, 0)})`;
  }
  return `<b>${esc(where)}</b><br>${what}<br><span class="muted">${sure} · ${c.passes} stones through here</span>`;
}

function findings(data) {
  const out = [], rows = data.draws, centre = t => rows.find(r => r.toward === t && r.x === 0);
  const a = centre('A'), b = centre('B');
  const gap = a.hog_to_hog_s - b.hog_to_hog_s, gapSd = Math.hypot(a.hog_to_hog_sd, b.hog_to_hog_sd);
  out.push(`<b>Speed.</b> A draw to the tee line takes <b>${a.hog_to_hog_s.toFixed(1)} s</b> hog to hog toward house A and <b>${b.hog_to_hog_s.toFixed(1)} s</b> toward house B (± ${Math.max(a.hog_to_hog_sd, b.hog_to_hog_sd).toFixed(1)}). ${
    Math.abs(gap) > 2 * gapSd && Math.abs(gap) >= 0.3 ? `The ice was quicker going toward house ${gap > 0 ? 'A' : 'B'}.` : 'No real difference between the two directions.'}`);
  const trend = data.speed.trend, perEnd = 100 * trend.value / data.speed.mean;
  if (Math.abs(trend.z) >= 2 && Math.abs(perEnd) >= 0.4) out.push(`<b>Through the game.</b> The ice got ${trend.value < 0 ? 'quicker' : 'slower'}: about ${Math.abs(perEnd).toFixed(1)}% ${trend.value < 0 ? 'less' : 'more'} drag each end.`);
  const curlTrend = data.curl.trend, curlPerEnd = 100 * curlTrend.value / data.curl.mean;
  if (Math.abs(curlTrend.z) >= 2 && Math.abs(curlPerEnd) >= 1) out.push(`<b>Through the game.</b> Stones curled ${curlTrend.value > 0 ? 'more' : 'less'} as it went on: about ${Math.abs(curlPerEnd).toFixed(0)}% per end.`);
  const curls = rows.map(r => r.curl_m), mean = curls.reduce((s, v) => s + v, 0) / curls.length;
  const extreme = rows.reduce((best, r) => (Math.abs(r.curl_m - mean) > Math.abs(best.curl_m - mean) ? r : best));
  const sideOf = r => (r.x === 0 ? 'the centre' : `the ${(r.x > 0) === (r.toward === 'B') ? 'left' : 'right'} side`);
  out.push(`<b>Curl.</b> A tee-line draw finishes about <b>${mean.toFixed(1)} m (${feet(mean)})</b> off the line it was on at the near hog. ${
    Math.abs(extreme.curl_m - mean) > 2.5 * extreme.curl_sd && Math.abs(extreme.curl_m - mean) > 0.15
      ? `It was ${extreme.curl_m > mean ? 'swingiest' : 'straightest'} down ${sideOf(extreme)} throwing toward house ${extreme.toward}: ${extreme.curl_m.toFixed(1)} m.` : 'That held on every line the stones tested.'}`);
  const falls = rows.filter(r => Math.abs(r.fall_x_m) >= 0.05 && Math.abs(r.fall_x_m) > 2.5 * r.fall_sd);
  if (falls.length) {
    for (const r of falls) {
      // a thrower heading toward B has +X on their left
      const toward = (r.fall_x_m > 0) === (r.toward === 'B') ? 'left' : 'right';
      out.push(`<b>Fall.</b> Throwing toward house ${r.toward} down ${sideOf(r)}, a draw is carried about <b>${Math.round(Math.abs(r.fall_x_m) * 100)} cm to the thrower’s ${toward}</b> whichever way it is turning (± ${Math.round(r.fall_sd * 100)}).`);
    }
  } else out.push('<b>Falls.</b> No line where a draw was carried sideways by more than a few centimetres regardless of its turn.');
  return out;
}

function drawTable(data) {
  const row = r => {
    const side = r.x === 0 ? 'Centre' : (r.x > 0) === (r.toward === 'B') ? 'Thrower’s left' : 'Thrower’s right';
    const toward = (r.fall_x_m > 0) === (r.toward === 'B') ? 'left' : 'right';
    const fall = Math.abs(r.fall_x_m) < 0.005 ? '0' : `${Math.round(Math.abs(r.fall_x_m) * 100)} cm ${toward}`;
    return `<tr><td>${side}</td><td>${r.hog_to_hog_s.toFixed(1)} s <span class="muted">± ${r.hog_to_hog_sd.toFixed(1)}</span></td>
      <td>${r.curl_m.toFixed(2)} m <span class="muted">(${feet(r.curl_m)}) ± ${r.curl_sd.toFixed(2)}</span></td>
      <td>${fall} <span class="muted">± ${Math.round(r.fall_sd * 100)}</span></td></tr>`;
  };
  return ['A', 'B'].map(toward => `
    <table class="table compact">
      <thead><tr><th>Toward house ${toward}</th><th>Hog to hog</th><th>Curl</th><th>Fall</th></tr></thead>
      <tbody>${data.draws.filter(r => r.toward === toward).sort((p, q) => (toward === 'B' ? q.x - p.x : p.x - q.x)).map(row).join('')}</tbody>
    </table>`).join('');
}

export function mount(container, {id, params}) {
  let disposed = false, view = params?.from === 'B' ? 'B' : 'A', showGlides = true, data = null;
  const redraws = [];
  container.innerHTML = '<p class="muted">Reading the ice…</p>';

  function render() {
    const thrownFrom = end => data.ends.filter(e => e.delivery_end === end).map(e => e.number).join(', ') || 'none';
    const other = view === 'A' ? 'B' : 'A';
    container.innerHTML = `
      <div class="row between">
        <div>
          <h1>Ice report</h1>
          <p class="muted">${data.shots} free glides from ${data.ends.length} end${data.ends.length === 1 ? '' : 's'}: every stone from the near hog line until it stopped, left the sheet or touched another stone.${
            data.untracked.length ? ` <span class="status warn">End${data.untracked.length === 1 ? '' : 's'} ${data.untracked.join(', ')} not tracked yet.</span>` : ''}</p>
        </div>
        <div class="row">
          <span class="muted">Looking from</span>
          <div class="segmented" id="view">
            <button data-view="A" class="${view === 'A' ? 'on' : ''}">House A</button>
            <button data-view="B" class="${view === 'B' ? 'on' : ''}">House B</button>
          </div>
        </div>
      </div>
      <section class="card">
        <h2>The short version</h2>
        <ul class="findings">${findings(data).map(f => `<li>${f}</li>`).join('')}</ul>
      </section>
      <section>
        <p class="muted">Maps are drawn as you would see the sheet standing behind house ${view}: house ${view} on the left (ends ${thrownFrom(view)} were thrown from it), house ${other} on the right (ends ${thrownFrom(other)}), and your left at the top. A patch of ice is the same patch whichever way a stone crosses it, which is what lets the two directions check each other.
          <label class="check" style="display:inline-flex;margin-left:8px"><input type="checkbox" id="glides" ${showGlides ? 'checked' : ''}> show the stones’ paths</label></p>
        ${Object.entries(MAPS).map(([key, spec]) => {
          const v = verdict(data[key].evidence, data[key].cells);
          return `<div class="icemap">
            <div class="row between"><h2>${spec.title} <span class="pill ${v.level}">${v.label}</span></h2>
              ${spec.legend ? `<span class="legend"><span class="swatch" style="background:${rgb(COOL)}"></span>${spec.legend[0]}<span class="swatch" style="background:${rgb(NEUTRAL)}"></span>average<span class="swatch" style="background:${rgb(WARM)}"></span>${spec.legend[1]}</span>` : ''}</div>
            <p class="muted">${spec.lead}</p>
            <div class="mapwrap"><canvas data-map="${key}"></canvas><div class="tip" hidden></div>
              <span class="end-label left">House ${view}</span><span class="end-label right">House ${other}</span></div>
            <p class="muted small">${v.text}</p>
          </div>`;
        }).join('')}
        <p class="muted small">Bright zones are ones the shots agree on; dim zones could be nothing; hatched zones had fewer than four stones through them.</p>
      </section>
      <section>
        <h2>A draw to the tee line, by line</h2>
        <p class="muted">What the fitted ice does to a stone that stops on the tee line: its hog-to-hog time, how far it finishes from where it was heading at the near hog line, and how much of that was the ice rather than the turn. Sides are the thrower’s.</p>
        <div class="draws">${drawTable(data)}</div>
      </section>
      <section>
        <details><summary>How this is worked out, and what it cannot see</summary>
          <p>Curl pushes a stone toward the side it is turning to; a fall pushes it the same way on the ice whatever the turn and whichever house it was thrown from. The game has ${Object.entries(data.counts).map(([k, n]) => `${n} ${k.endsWith('ccw') ? 'counter-clockwise' : 'clockwise'} toward ${k[0]}`).join(', ')}, and the two are separated by fitting all of them at once. Each shot keeps its own amount of curl and drag, so one heavily swept or lazily turned stone does not paint a patch.</p>
          <p>A map counts as real only if it predicts shots that were left out of the fit better than uniform ice does. Maps are also pulled toward “no difference”, so a real patch is, if anything, stronger than drawn.</p>
          <p><b>Sweeping is not seen.</b> It makes ice faster and straighter where it happens, and draws are swept most in the second half of their travel, so a zone that reads keen and straight may be brooms rather than ice. Falls are not affected in the same way: sweeping does not push stones sideways by turn-independent amounts.</p>
          <p>Where one turn was thrown much more than the other down a line, curl and fall can borrow from each other. Camera calibration errors that are fixed on the sheet are fitted separately and removed, since they move fast and slow stones alike while a real pull moves slow stones far more.</p>
          <p><a href="${link(id, 'report')}">Shot report</a> · <a href="${link(id, 'tracking')}">Tracking</a></p>
        </details>
      </section>`;
    redraws.length = 0;
    container.querySelectorAll('canvas[data-map]').forEach(canvas => {
      const key = canvas.dataset.map, tip = canvas.parentElement.querySelector('.tip');
      let boxes = [];
      const paint = () => { boxes = drawMap(canvas, key, data, view, showGlides); };
      paint();
      redraws.push(paint);
      canvas.onmousemove = event => {
        const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
        const k = boxes.findIndex(b => x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h);
        if (k < 0) { tip.hidden = true; return; }
        tip.innerHTML = cellWords(key, data[key].cells[k], data, boxes[k].i, boxes[k].j, view);
        tip.hidden = false;
        tip.style.left = `${Math.min(rect.width - 250, Math.max(0, x + 12))}px`;
        tip.style.top = `${y + 14}px`;
      };
      canvas.onmouseleave = () => { tip.hidden = true; };
    });
    container.querySelectorAll('#view button').forEach(button => { button.onclick = () => { view = button.dataset.view; render(); }; });
    container.querySelector('#glides').onchange = event => { showGlides = event.target.checked; redraws.forEach(paint => paint()); };
  }

  const onResize = () => redraws.forEach(paint => paint());
  window.addEventListener('resize', onResize);
  attempt(async () => {
    data = await api(`/recordings/${encodeURIComponent(id)}/ice`);
    if (disposed) return;
    if (!data.ready) {
      container.innerHTML = `<div class="empty"><h2>Not enough shots for an ice report yet</h2>
        <p>${data.shots} free glide${data.shots === 1 ? '' : 's'} so far. The ice can only be read once stones have gone both ways down the sheet with both turns: track at least two or three ends.</p>
        <p><a href="${link(id, 'tracking')}">Track the stones</a></p></div>`;
      return;
    }
    render();
  });
  return () => { disposed = true; window.removeEventListener('resize', onResize); };
}
