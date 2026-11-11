// Small helpers shared by every screen.

export const esc = value => String(value ?? '').replace(/[&<>"']/g,
  c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

/** Seconds as 1:02:03.4 (hours only when needed). */
export function clock(seconds, decimals = 1) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '–';
  const whole = Math.floor(seconds / 60), s = seconds - whole * 60;
  const tail = `${String(whole % 60).padStart(whole >= 60 ? 2 : 1, '0')}:${s.toFixed(decimals).padStart(decimals ? 3 + decimals : 2, '0')}`;
  return whole >= 60 ? `${Math.floor(whole / 60)}:${tail}` : tail;
}

/** "1:02:03.4", "62:03" or plain seconds, back to seconds. NaN when it is none of those. */
export function parseClock(text) {
  const parts = String(text).trim().split(':');
  if (!parts.length || parts.length > 3 || parts.some(p => p === '' || !/^\d+(\.\d+)?$/.test(p))) return NaN;
  return parts.reduce((total, part) => total * 60 + Number(part), 0);
}

export function duration(seconds) {
  const minutes = Math.round(seconds / 60);
  return minutes >= 60 ? `${Math.floor(minutes / 60)} h ${minutes % 60} min` : `${minutes} min`;
}

export async function api(path, method = 'GET', body) {
  const response = await fetch('/api' + path, {
    method,
    headers: body === undefined ? {} : {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = value.detail;
    throw Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Request failed (${response.status})`);
  }
  return value;
}

let toastTimer = null;
export function toast(text, kind = 'info') {
  const el = document.querySelector('#toast');
  el.textContent = text;
  el.className = kind;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === 'error' ? 6000 : 2500);
}

/** Run an async action from a click handler; failures become an error toast. */
export async function attempt(action) {
  try { return await action(); } catch (error) { toast(error.message, 'error'); return undefined; }
}

/** Call `tick` now and every `ms` until the returned function is called. Overlapping calls are skipped. */
export function poll(tick, ms = 3000) {
  let stopped = false, busy = false;
  const run = async () => {
    if (stopped || busy) return;
    busy = true;
    try { await tick(); } catch (error) { toast(error.message, 'error'); } finally { busy = false; }
  };
  run();
  const timer = setInterval(run, ms);
  return () => { stopped = true; clearInterval(timer); };
}

/** Address of a recording's screen: #/r/<id>/<tab>?end=&shot= */
export function link(id, tab, params = {}) {
  const query = new URLSearchParams(params).toString();
  return `#/r/${encodeURIComponent(id)}/${tab}${query ? '?' + query : ''}`;
}

/** Update the address without re-mounting the screen (used while stepping through shots). */
export function replaceParams(id, tab, params) {
  history.replaceState(null, '', link(id, tab, params));
}
