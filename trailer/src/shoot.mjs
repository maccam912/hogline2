// Frame-by-frame capture of the Hogline 2 trailer scenes. usage: node shoot.mjs <scene...> [--still N]
import {chromium} from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = 'http://127.0.0.1:8772', REC = 'eDJdwj0JqBY';
const WIN = {x: 120, y: 18, w: 1040, h: 585};          // where the app "window" sits in the 1280x720 frame
const FULL = [0, 0, 1280];
const cl = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const ease = x => { x = cl(x); return x * x * (3 - 2 * x); };
const mix = (a, b, k) => a.map((v, i) => v + (b[i] - v) * k);
const seg = (f, a, b) => ease((f - a) / (b - a));

const args = process.argv.slice(2);
const stillAt = args.includes('--still') ? Number(args[args.indexOf('--still') + 1]) : null;
const wanted = args.filter(a => !a.startsWith('--') && !/^\d+$/.test(a));

const browser = await chromium.launch({channel: 'chrome', headless: true});
const context = await browser.newContext({viewport: {width: 1280, height: 720}, deviceScaleFactor: 1.5});
await context.addInitScript(() => {
  const real = window.setTimeout;
  window.setTimeout = (fn, ms, ...rest) => real(fn, ms === 2500 ? 600000 : ms, ...rest);   // capture is slower than real time; toasts are hidden by the script
});

function overlay() {
  if (document.querySelector('#__ov')) return;
  const WIN = {x: 120, y: 18, w: 1040, h: 585};
  const style = document.createElement('style');
  style.textContent = `
    html { overflow: hidden !important; background: radial-gradient(ellipse at 50% 0%, #17232a 0%, #080c0e 70%) #080c0e !important; }
    body { transform-origin: 0 0; width: 1280px; height: 720px !important; background: #0e1315; }
    #__ov { position: fixed; inset: 0; pointer-events: none; z-index: 99999; font-family: "Avenir Next", "Helvetica Neue", sans-serif; color: #fff; }
    #__frame { position: absolute; left: ${WIN.x}px; top: ${WIN.y}px; width: ${WIN.w}px; height: ${WIN.h}px; border-radius: 10px;
      box-shadow: 0 0 0 1px rgba(255,255,255,.10), 0 28px 70px rgba(0,0,0,.65); }
    #__cap { position: absolute; left: ${WIN.x}px; right: ${WIN.x}px; top: ${WIN.y + WIN.h + 20}px; display: flex; align-items: center; gap: 22px; }
    #__n { font: 700 54px "Avenir Next"; color: #6ec1ff; letter-spacing: -.02em; line-height: 1; min-width: 70px; }
    #__h { font-size: 31px; font-weight: 700; line-height: 1.15; }
    #__s { font-size: 18px; font-weight: 500; color: #9fb3ba; margin-top: 3px; }
    #__cur { position: absolute; width: 26px; height: 26px; margin: -3px 0 0 -5px; filter: drop-shadow(0 2px 4px rgba(0,0,0,.6)); }
    #__rip { position: absolute; width: 60px; height: 60px; margin: -30px 0 0 -30px; border-radius: 50%; border: 3px solid #6ec1ff; opacity: 0; }
    #__blk { position: absolute; inset: 0; background: #000; opacity: 0; }`;
  document.head.append(style);
  const ov = document.createElement('div');
  ov.id = '__ov';
  ov.innerHTML = `<div id="__frame"></div>
    <div id="__cap"><div id="__n"></div><div><div id="__h"></div><div id="__s"></div></div></div>
    <div id="__rip"></div>
    <svg id="__cur" viewBox="0 0 24 24"><path d="M5 3l14 8.5-6.3 1.4 3.6 7.2-2.6 1.3-3.6-7.2L5 19z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>
    <div id="__blk"></div>`;
  document.documentElement.append(ov);
  window.__cam = [0, 0, 1280];
  window.__ov = st => {
    const [cx, cy, cw] = st.cam, s = WIN.w / cw, ch = cw * WIN.h / WIN.w;
    window.__cam = st.cam;
    const b = document.body.style;
    b.transform = `translate(${WIN.x - cx * s}px, ${WIN.y - cy * s}px) scale(${s})`;
    b.clipPath = `inset(${cy}px ${1280 - cx - cw}px ${720 - cy - ch}px ${cx}px round ${10 / s}px)`;
    const q = id => document.querySelector(id);
    q('#__cur').style.display = st.cursor ? '' : 'none';
    if (st.cursor) { q('#__cur').style.left = st.cursor[0] + 'px'; q('#__cur').style.top = st.cursor[1] + 'px'; }
    const r = q('#__rip').style, p = st.press ?? -1;
    r.opacity = p >= 0 && p < 1 ? 1 - p : 0;
    if (st.cursor) { r.left = st.cursor[0] + 'px'; r.top = st.cursor[1] + 'px'; }
    r.transform = `scale(${.3 + p * .9})`;
    q('#__n').textContent = st.cap?.n ?? ''; q('#__h').textContent = st.cap?.h ?? ''; q('#__s').textContent = st.cap?.s ?? '';
    q('#__cap').style.opacity = st.cap?.o ?? 1;
    q('#__cap').style.transform = `translateY(${(1 - (st.cap?.o ?? 1)) * 10}px)`;
    q('#__blk').style.opacity = st.black ?? 0;
  };
  // rect of an element in app (untransformed) coordinates
  window.__rect = el => {
    const r = el.getBoundingClientRect(), [cx, cy, cw] = window.__cam, s = WIN.w / cw;
    return {x: cx + (r.left - WIN.x) / s, y: cy + (r.top - WIN.y) / s, w: r.width / s, h: r.height / s};
  };
  window.__seek = async t => {
    const v = document.querySelector('video');
    if (Math.abs(v.currentTime - t) > 1e-3) await new Promise(res => { v.addEventListener('seeked', res, {once: true}); v.currentTime = t; });
    await new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
  };
  window.__ov({cam: [0, 0, 1280]});
}

/** A scene being captured: holds camera, cursor and caption state between frames. */
class Take {
  constructor(page, name, frames) {
    Object.assign(this, {page, name, frames, f: 0, cam: FULL, cur: null, press: -1, cap: null, black: 0});
    this.dir = path.join(HERE, 'frames', name);
    if (stillAt === null) { fs.rmSync(this.dir, {recursive: true, force: true}); fs.mkdirSync(this.dir, {recursive: true}); }
  }
  toScreen([x, y]) { const s = WIN.w / this.cam[2]; return [WIN.x + (x - this.cam[0]) * s, WIN.y + (y - this.cam[1]) * s]; }
  async rect(selector, nth = 0) {
    return this.page.evaluate(([sel, n]) => { const el = document.querySelectorAll(sel)[n]; return el ? window.__rect(el) : null; }, [selector, nth]);
  }
  async rectOf(fn, arg) { return this.page.evaluate(new Function('a', `const el = (${fn})(a); return el ? window.__rect(el) : null;`), arg); }
  async snap() {
    const edge = Math.max(cl(1 - this.f / 5), cl((this.f - (this.frames - 5)) / 5)) * 0;   // hard cuts between scenes
    await this.page.evaluate(st => window.__ov(st), {cam: this.cam, cursor: this.cur && this.toScreen(this.cur), press: this.press, cap: this.cap, black: Math.max(this.black, edge)});
    if (stillAt === null) await this.page.screenshot({path: path.join(this.dir, String(this.f).padStart(5, '0') + '.jpg'), type: 'jpeg', quality: 94});
    else if (this.f === stillAt) await this.page.screenshot({path: path.join(HERE, 'look', `${this.name}-${stillAt}.jpg`), type: 'jpeg', quality: 85});
    this.f++;
    if (this.f % 24 === 0) process.stdout.write(`\r${this.name} ${this.f}/${this.frames}   `);
  }
  /** Run frames up to `until`; `each(f)` sets state for that frame. */
  async run(until, each) { while (this.f < until) { await each?.(this.f); await this.snap(); } }
  async click() { const [x, y] = this.toScreen(this.cur); await this.page.mouse.click(x, y); }
  async move() { const [x, y] = this.toScreen(this.cur); await this.page.mouse.move(x, y); }
}

const mid = r => [r.x + r.w / 2, r.y + r.h / 2];
const capIn = (f, a = 2) => cl((f - a) / 10);
/** 16:9 camera that contains rect r with a margin, clamped to the page. */
function camOn(r, margin = 30, minW = 420) {
  let w = Math.max(minW, r.w + 2 * margin, (r.h + 2 * margin) * WIN.w / WIN.h);
  w = Math.min(w, 1280);
  const h = w * WIN.h / WIN.w;
  return [cl(r.x + r.w / 2 - w / 2, 0, 1280 - w), cl(r.y + r.h / 2 - h / 2, 0, 720 - h), w];
}
async function open(page, hash, ready) {
  await page.goto(APP + '/#/');
  await page.goto(`${APP}/${hash}`);
  await page.waitForSelector(ready);
  await page.waitForTimeout(1200);
  await page.evaluate(overlay);
}

const scenes = {
  // ---------------------------------------------------------------- title cards (local page)
  async cards(page, which) {
    await page.goto('file://' + path.join(HERE, 'cards.html'));
    await page.waitForFunction(() => document.querySelector('video').readyState >= 2);
    for (const [name, n] of which) {
      const dir = path.join(HERE, 'frames', name);
      if (stillAt === null) { fs.rmSync(dir, {recursive: true, force: true}); fs.mkdirSync(dir, {recursive: true}); }
      for (let f = 0; f < n; f++) {
        if (stillAt !== null && f !== stillAt) continue;
        await page.evaluate(([s, i]) => window.render(s, i), [name, f]);
        await page.screenshot({path: stillAt === null ? path.join(dir, String(f).padStart(5, '0') + '.jpg') : path.join(HERE, 'look', `${name}-${f}.jpg`), type: 'jpeg', quality: 94});
        if (f % 24 === 0) process.stdout.write(`\r${name} ${f}/${n}   `);
      }
    }
  },

  // ---------------------------------------------------------------- 1a: paste the link
  async s1a(page) {
    await open(page, '#/', '#url');
    const t = new Take(page, 's1a', 144);
    const url = await t.rect('#url'), btn = await t.rect('#add button[type=submit]'), form = await t.rect('#add');
    const cap = o => ({n: '1', h: 'Paste the link to the club’s stream', s: 'Any finished YouTube broadcast of your game.', o});
    const start = [820, 470], zoom = camOn({x: url.x - 16, y: url.y - 120, w: 600, h: 300}, 0), zoom2 = camOn({x: btn.x + btn.w - 700, y: url.y - 110, w: 700, h: 330}, 16);
    const link = `https://www.youtube.com/watch?v=${REC}`;
    await t.run(26, f => { t.cap = cap(capIn(f)); t.cur = mix(start, [url.x + 190, url.y + url.h / 2], seg(f, 4, 24)); t.cam = mix(FULL, zoom, seg(f, 0, 60)); });
    await t.click(); t.press = 0;
    let typed = 0;
    await t.run(96, async f => {
      t.press = f < 36 ? (f - 26) / 10 : -1; t.cam = mix(FULL, zoom, seg(f, 0, 60)); t.cur = [url.x + 190 + 40 * seg(f, 28, 40), url.y + url.h / 2 + 34 * seg(f, 28, 40)];
      const want = Math.round(cl((f - 34) / 50) * link.length);
      if (want > typed) { await page.keyboard.type(link.slice(typed, want)); typed = want; }
    });
    await t.run(144, async f => { t.cur = mix([url.x + 230, url.y + url.h / 2 + 34], mid(btn), seg(f, 98, 122)); t.cam = mix(zoom, zoom2, seg(f, 96, 126)); await t.move(); });
  },

  // ---------------------------------------------------------------- 1b: ends found
  async s1b(page) {
    await open(page, `#/r/${REC}/ends`, '.end.card');
    await page.waitForFunction(() => document.querySelector('#player').readyState >= 1);
    const t = new Take(page, 's1b', 144);
    const cap = o => ({n: '1', h: 'It finds every end for you', s: 'A vision model proposes the cuts. You check them, then export a clip per end.', o: 1});
    const go = async i => mid(await t.rectOf(`i => document.querySelectorAll('.end.card')[i].querySelector('.jump[data-kind=start]')`, i));
    const g1 = await go(1), g2 = await go(2);
    t.cur = [700, 520];
    const jump = async at => { await t.click(); t.press = 0; await page.waitForTimeout(150); await page.evaluate(() => window.__seek(document.querySelector('video').currentTime + 0)); await page.waitForTimeout(700); };
    await t.run(34, f => { t.cap = cap(); t.cur = mix([700, 520], g1, seg(f, 6, 30)); });
    await jump();
    await t.run(84, f => { t.press = f < 46 ? (f - 34) / 12 : -1; t.cur = mix(g1, g2, seg(f, 58, 82)); });
    await jump();
    const vid = await t.rect('#player');
    await t.run(144, f => { t.press = f < 96 ? (f - 84) / 12 : -1; t.cam = mix(FULL, camOn({x: vid.x, y: vid.y, w: 1040, h: vid.h}, 20), seg(f, 96, 144)); });
  },

  // ---------------------------------------------------------------- 2a: tracking table
  async s2a(page) {
    await open(page, `#/r/${REC}/tracking`, 'table');
    const t = new Take(page, 's2a', 72);
    const table = await t.rect('table');
    const to = camOn({x: table.x, y: table.y - 100, w: table.w, h: table.h + 100}, 16);
    await t.run(72, f => { t.cap = {n: '2', h: 'It tracks every stone', s: 'Four cameras, calibrated from the painted rings. Under a minute per end.', o: capIn(f)}; t.cam = mix(FULL, to, seg(f, 0, 72)); });
  },

  // ---------------------------------------------------------------- 2b: stones followed on all cameras
  async s2b(page) {
    await open(page, `#/r/${REC}/review?end=1&shot=yellow_7`, '#facts h2');
    await page.waitForFunction(() => document.querySelector('video').readyState >= 2);
    const t = new Take(page, 's2b', 216);
    const cap = {n: '2', h: 'It tracks every stone', s: 'Every stone is named and followed down the sheet, on every camera.', o: 1};
    const whole = mid(await t.rect('#views button[data-view=all]'));
    const t0 = 1103.2;
    await page.evaluate(s => window.__seek(s), t0);
    t.cap = cap; t.cur = [640, 420];
    await t.run(20, f => { t.cur = mix([640, 420], whole, seg(f, 2, 18)); });
    await t.click(); t.press = 0;
    await page.waitForTimeout(300);
    const scr = await t.rect('#screen'), strip = await t.rect('#strip');
    const to = camOn({x: scr.x, y: scr.y, w: scr.w, h: strip.y + strip.h - scr.y}, 10);
    await t.run(216, async f => {
      t.press = f < 32 ? (f - 20) / 12 : -1;
      if (f > 40) t.cur = null;
      t.cam = mix(FULL, to, seg(f, 30, 90));
      await page.evaluate(s => window.__seek(s), t0 + (f - 20) / 10);       // 3x speed
    });
  },

  // ---------------------------------------------------------------- 3: click the broom
  async s3(page) {
    const api = `${APP}/api/recordings/${REC}/shots/1/red_1`;
    const marks = JSON.parse(fs.readFileSync('/Users/maccam912/dev/hogline2/data/recordings/eDJdwj0JqBY/tracking/broom-marks.json'));
    const original = marks['1:red_1'];
    const body = m => ({seconds: m.seconds ?? null, px: m.px ?? null, call: m.call ?? null, no_broom: !!m.no_broom});
    const put = b => fetch(api, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b)}).then(r => { if (!r.ok) throw Error('PUT ' + r.status); });
    try {
      await put({...body(original), px: null, seconds: null});            // the "before": this shot unmarked
      await open(page, `#/r/${REC}/review?end=1&shot=red_1`, '#facts h2');
      await page.waitForFunction(() => document.querySelector('video').readyState >= 2);
      await page.evaluate(s => window.__seek(s), original.seconds);
      const t = new Take(page, 's3', 288);
      const cap = {n: '3', h: 'Click the skip’s broom', s: 'One click per shot. It saves as you go.', o: 1};
      // where the original mark sits on screen: perspective_a is drawn through the house crop
      const spot = await page.evaluate(px => {
        const o = document.querySelector('#overlay'), v = document.querySelector('#video');
        const vr = window.__rect(v);
        return [vr.x + px[0] / 1920 * vr.w, vr.y + px[1] / 1080 * vr.h];
      }, original.px);
      const stage = await t.rect('.stage'), facts = await t.rect('#facts');
      const wide = camOn({x: stage.x, y: stage.y, w: 1280 - stage.x, h: stage.h - 60}, 0);
      t.cur = [760, 560];
      await t.run(50, f => { t.cap = {...cap, o: capIn(f)}; t.cur = mix([760, 560], spot, seg(f, 10, 46)); });
      await t.click(); t.press = 0;
      await page.waitForFunction(() => !document.querySelector('#toast').hidden);
      await page.waitForTimeout(300);
      await t.run(100, f => { t.press = f < 64 ? (f - 50) / 14 : -1; if (f > 76) t.cur = mix(spot, [spot[0] + 150, spot[1] + 140], seg(f, 76, 100)); });
      const dl = await t.rect('#facts dl');
      const tight = camOn({x: dl.x - 330, y: dl.y - 60, w: dl.w + 330, h: dl.h + 80}, 10);
      await t.run(288, async f => {
        if (f === 124) await page.evaluate(() => { document.querySelector('#toast').hidden = true; });
        if (f > 120) t.cur = null;
        t.cam = f < 200 ? mix(FULL, wide, seg(f, 118, 160)) : mix(wide, tight, seg(f, 200, 262));
        await page.evaluate(s => window.__seek(s), original.seconds + (f - 100) * 0.125);   // ~4x speed: release to the house
      });
    } finally {
      await put(body(original));                                            // put the mark back exactly as it was
    }
  },

  // ---------------------------------------------------------------- 4: the report
  async s4(page) {
    await open(page, `#/r/${REC}/report`, 'table.table');
    await page.waitForTimeout(6500);                                        // let any start-up toast clear
    const t = new Take(page, 's4', 288);
    const cap = {n: '4', h: 'Line and weight, for every player', s: 'How far off the broom, split by turn. Hog-to-hog time, split by call.', o: 1};
    const row = await t.rectOf(`() => [...document.querySelectorAll('table.table tbody tr')].find(tr => tr.querySelector('b')?.textContent === 'Matt')`);
    const tight = camOn({x: row.x, y: row.y - 8, w: 760, h: row.h + 16}, 14, 640);
    const scroller = `(() => { let best = document.scrollingElement; for (const el of document.querySelectorAll('main, main *, body')) if (el.scrollHeight - el.clientHeight > best.scrollHeight - best.clientHeight && getComputedStyle(el).overflowY !== 'visible') best = el; return best; })()`;
    const csvTop = await page.evaluate(`(() => { const s = ${scroller}; return document.querySelector('#csv').getBoundingClientRect().top - s.getBoundingClientRect().top + s.scrollTop; })()`);
    await t.run(110, f => { t.cap = {...cap, o: capIn(f)}; t.cam = mix(FULL, tight, seg(f, 14, 70)); });
    await t.run(288, async f => {
      t.cam = mix(tight, FULL, seg(f, 110, 150));
      const y = (csvTop - 60) * seg(f, 120, 175) + Math.max(0, f - 175) * 3.2;
      await page.evaluate(`${scroller}.scrollTop = ${y}`);
      if (f >= 150) {
        const csv = await t.rect('#csv');
        const target = mid(csv);
        t.cur = f < 190 ? mix([900, 520], target, seg(f, 150, 186)) : target;
        if (f === 190) await t.move();
      }
    });
  },
};

const page = await context.newPage();
page.on('pageerror', e => console.log('\npage error:', e.message));
for (const name of wanted) {
  const t0 = Date.now();
  if (name === 'cards') await scenes.cards(page, [['open', 288], ['title', 144], ['why', 288], ['end', 288]]);
  else if (['open', 'title', 'why', 'end'].includes(name)) await scenes.cards(page, [[name, {open: 288, title: 144, why: 288, end: 288}[name]]]);
  else await scenes[name](page);
  console.log(`\n${name} done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
await browser.close();
