// Who threw what: team lineups, the per-player summary and every shot in a table.
import {api, attempt, esc, link, toast} from './util.js';

const POSITIONS = ['Lead', 'Second', 'Third', 'Fourth'];
const cm = value => (value === null || value === undefined ? '–' : `${value.toFixed(0)} cm`);
const signed = value => (value === null || value === undefined ? '–' : `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(0)} cm`);

function lineupForm(lineups) {
  return ['red', 'yellow'].map(colour => `
    <fieldset>
      <legend><span class="dot ${colour}"></span>${colour === 'red' ? 'Red' : 'Yellow'}</legend>
      ${POSITIONS.map((position, i) => `<label>${position} <span class="muted">stones ${2 * i + 1}–${2 * i + 2}</span>
        <input data-colour="${colour}" data-i="${i}" maxlength="100" value="${esc(lineups[colour][i])}" placeholder="Name"></label>`).join('')}
    </fieldset>`).join('');
}

function playerTable(report) {
  if (!report.length) return '<p class="muted">No tracked shots yet.</p>';
  const rows = report.map(p => {
    const turns = Object.entries(p.line);
    const weight = Object.entries(p.weight).map(([call, w]) =>
      `<span class="chip">${esc(call)} <b>${w.mean_hog_to_hog_s.toFixed(1)} s</b>${w.spread_s === null ? '' : ` ± ${w.spread_s.toFixed(1)}`} <span class="muted">(${w.shots})</span></span>`).join(' ');
    const lines = turns.length ? turns.map(([turn, t]) =>
      `<div>${esc(turn)}: typically <b>${cm(t.mean_abs_cm)}</b> off, average ${signed(t.mean_narrow_cm)} <span class="muted">(${t.shots})</span></div>`).join('')
      : '<span class="muted">No brooms marked</span>';
    return `<tr>
      <td><span class="dot ${p.colour}"></span><b>${esc(p.player)}</b></td>
      <td>${p.marked} of ${p.shots}</td>
      <td>${lines}</td>
      <td>${weight || '–'}</td></tr>`;
  }).join('');
  return `<table class="table">
    <thead><tr><th>Player</th><th>Brooms marked</th><th>Line, by turn</th><th>Weight, hog to hog by call</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function shotTable(id, shots) {
  const cell = (value, digits = 2) => (value === null || value === undefined ? '–' : value.toFixed(digits));
  return `<table class="table compact">
    <thead><tr><th>End</th><th>#</th><th>Stone</th><th>Player</th><th>Call</th><th>Hog to hog</th><th>Tee to hog</th><th>Turn</th><th>Curl</th><th>Line vs broom</th><th></th></tr></thead>
    <tbody>${shots.map(s => {
      const aim = s.aim || {};
      const line = aim.miss_right_cm === undefined ? (s.mark?.no_broom ? 'no broom' : '–')
        : `${Math.abs(aim.miss_right_cm).toFixed(0)} cm ${aim.miss_right_cm >= 0 ? 'right' : 'left'}${aim.miss_narrow_cm === undefined ? '' : aim.miss_narrow_cm >= 0 ? ', narrow' : ', wide'}`;
      return `<tr>
        <td>${s.end}</td><td>${s.throw}</td><td><span class="dot ${s.colour}"></span>${esc(s.label)}</td>
        <td>${esc(s.player)}</td><td>${esc(s.mark?.call || '–')}</td>
        <td>${cell(s.hog_to_hog_s)}</td><td>${cell(s.tee_to_hog_s)}</td>
        <td>${esc(aim.turn || '–')}</td><td>${aim.curl_right_m === null || aim.curl_right_m === undefined ? '–' : cm(Math.abs(aim.curl_right_m) * 100)}</td>
        <td>${esc(line)}</td>
        <td><a href="${link(id, 'review', {end: s.end, shot: s.stone})}">Review</a></td></tr>`;
    }).join('')}</tbody></table>`;
}

function csv(shots) {
  const header = ['end', 'throw', 'stone', 'player', 'call', 'hog_to_hog_s', 'tee_to_hog_s', 'turn', 'curl_right_m', 'miss_right_cm', 'miss_narrow_cm', 'broom'];
  const rows = shots.map(s => [s.end, s.throw, s.label, s.player, s.mark?.call, s.hog_to_hog_s, s.tee_to_hog_s, s.aim?.turn,
    s.aim?.curl_right_m, s.aim?.miss_right_cm, s.aim?.miss_narrow_cm, s.mark?.px ? 'marked' : s.mark?.no_broom ? 'not visible' : '']);
  const quote = value => (value === null || value === undefined ? '' : /[",\n]/.test(String(value)) ? `"${String(value).replace(/"/g, '""')}"` : String(value));
  return [header, ...rows].map(row => row.map(quote).join(',')).join('\n') + '\n';
}

export function mount(container, {id, record}) {
  const base = `/recordings/${encodeURIComponent(id)}`;
  let disposed = false;
  container.innerHTML = '<p class="muted">Loading…</p>';

  async function load() {
    const data = await api(`${base}/shots`);
    if (disposed) return;
    container.innerHTML = `
      <h1>Report</h1>
      <section>
        <h2>Players</h2>
        <p class="muted">Line is the stone’s heading just after release, carried to the broom’s depth. Narrow (+) means it started toward the side it then curled to; wide (−) is the other side. Weight is the hog-to-hog time, grouped by the call you labelled.</p>
        ${playerTable(data.report)}
      </section>
      <section>
        <h2>Lineups</h2>
        <p class="muted">Names are matched to stones by throwing order, so each player has two stones per end.</p>
        <form id="lineups" class="lineups">${lineupForm(data.lineups)}<div><button type="submit">Save lineups</button></div></form>
      </section>
      <section>
        <div class="row between"><h2>Every shot</h2><button id="csv" class="quiet" ${data.shots.length ? '' : 'disabled'}>Download CSV</button></div>
        ${data.shots.length ? shotTable(id, data.shots) : `<p class="muted">Nothing tracked yet. <a href="${link(id, 'tracking')}">Track the stones</a> first.</p>`}
      </section>`;
    container.querySelector('#lineups').onsubmit = event => {
      event.preventDefault();
      attempt(async () => {
        const value = {red: [], yellow: []};
        container.querySelectorAll('#lineups input').forEach(input => { value[input.dataset.colour][Number(input.dataset.i)] = input.value; });
        await api(`${base}/lineups`, 'PUT', value);
        toast('Lineups saved');
        await load();
      });
    };
    container.querySelector('#csv').onclick = () => {
      const url = URL.createObjectURL(new Blob([csv(data.shots)], {type: 'text/csv'}));
      const a = Object.assign(document.createElement('a'), {href: url, download: `${record.title.replace(/[^\w-]+/g, '_')}-shots.csv`});
      a.click();
      URL.revokeObjectURL(url);
    };
  }
  attempt(load);
  return () => { disposed = true; };
}
