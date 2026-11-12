// App shell: the top bar and a tiny hash router.
//   #/                         all recordings
//   #/r/<id>/<tab>[?end=&shot=]   one recording: ends | tracking | review | report | ice
import {api, esc, link, toast} from './util.js';
import * as home from './home.js';
import * as ends from './ends.js';
import * as tracking from './tracking.js';
import * as review from './review.js';
import * as report from './report.js';
import * as ice from './ice.js';

const TABS = [
  ['ends', 'Ends', ends],
  ['tracking', 'Tracking', tracking],
  ['review', 'Review shots', review],
  ['report', 'Report', report],
  ['ice', 'Ice', ice],
];
const view = document.querySelector('#view');
const picker = document.querySelector('#recording-picker');
const tabs = document.querySelector('#tabs');
let cleanup = null, routeToken = 0;

function parse() {
  const [path, query = ''] = location.hash.replace(/^#/, '').split('?');
  const parts = path.split('/').filter(Boolean);
  if (parts[0] !== 'r' || !parts[1]) return {};
  return {id: decodeURIComponent(parts[1]), tab: parts[2], params: Object.fromEntries(new URLSearchParams(query))};
}

/** The first screen that still has something to do. */
async function firstTab(id, record) {
  if (!record.ends?.length) return 'ends';
  const state = await api(`/recordings/${encodeURIComponent(id)}/tracking`);
  return state.ends.some(e => e.run) ? 'review' : 'tracking';
}

async function route() {
  const token = ++routeToken;
  cleanup?.();
  cleanup = null;
  const {id, tab, params} = parse();
  try {
    const recordings = await api('/recordings');
    if (token !== routeToken) return;
    picker.hidden = tabs.hidden = !id;
    if (!id) {
      document.title = 'Hogline 2';
      view.className = 'page';
      cleanup = home.mount(view, {recordings});
      return;
    }
    const record = recordings.find(r => r.id === id);
    if (!record) throw Error('That recording is not in the catalog.');
    if (!TABS.some(t => t[0] === tab)) {
      location.replace(link(id, await firstTab(id, record)));
      return;
    }
    picker.innerHTML = recordings.map(r => `<option value="${esc(r.id)}"${r.id === id ? ' selected' : ''}>${esc(r.title)}</option>`).join('');
    tabs.innerHTML = TABS.map(([key, label]) => `<a href="${link(id, key)}"${key === tab ? ' aria-current="page"' : ''}>${label}</a>`).join('');
    document.title = `${record.title} · Hogline 2`;
    view.className = tab === 'review' ? 'workspace' : 'page';
    view.innerHTML = '';
    cleanup = TABS.find(t => t[0] === tab)[2].mount(view, {id, record, params});
  } catch (error) {
    if (token !== routeToken) return;
    view.className = 'page';
    view.innerHTML = `<div class="empty"><h2>Something went wrong</h2><p>${esc(error.message)}</p><p><a href="#/">Back to recordings</a></p></div>`;
    toast(error.message, 'error');
  }
}

picker.onchange = () => { location.hash = link(picker.value, parse().tab || 'review'); };
window.addEventListener('hashchange', route);
route();
