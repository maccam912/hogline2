// Track the stones: one button for the whole game, or one end at a time.
import {api, attempt, clock, esc, link, poll} from './util.js';

const ACTIVE = ['queued', 'running', 'stopping'];

function checkNote(run) {
  const notes = [];
  if (run.stones !== 16) notes.push(`${run.stones} of 16 deliveries found`);
  const check = run.check;
  if (check?.tracked_not_detected?.length) notes.push(`not seen at the end of play: ${check.tracked_not_detected.join(', ')}`);
  if (check?.detected_not_tracked?.length) notes.push(`${check.detected_not_tracked.length} unnamed stone${check.detected_not_tracked.length === 1 ? '' : 's'} in the house`);
  if (!run.whole_end) notes.push('covers only part of the end');
  return notes;
}

export function mount(container, {id}) {
  const base = `/recordings/${encodeURIComponent(id)}/tracking`;
  container.innerHTML = `
    <div class="row between">
      <div>
        <h1>Tracking</h1>
        <p class="muted">Finds every stone in every camera and follows it down the sheet. The first run on a recording also calibrates the cameras from the painted lines, which takes a few extra minutes.</p>
      </div>
      <div class="row">
        <button id="all">Track all ends</button>
        <button id="stop" class="quiet" hidden>Stop</button>
      </div>
    </div>
    <div id="job"></div>
    <table class="table" id="ends"></table>
    <label class="inline">End 1 is thrown from
      <select id="first"><option value="A">house A</option><option value="B">house B</option></select>
      <span class="muted">Ends alternate from there. If an end finds no deliveries, this is probably the wrong way round.</span>
    </label>`;
  const $ = selector => container.querySelector(selector);

  function draw(state) {
    const job = state.job, active = job && ACTIVE.includes(job.state);
    const untracked = state.ends.filter(e => !e.run).map(e => e.number);
    const inJob = new Map((active ? job.ends : []).map(e => [e.end_number, e]));
    $('#all').disabled = active || !state.ends.length;
    $('#all').textContent = untracked.length && untracked.length < state.ends.length ? `Track the ${untracked.length} untracked end${untracked.length === 1 ? '' : 's'}` : 'Track all ends';
    $('#all').dataset.ends = JSON.stringify(untracked.length && untracked.length < state.ends.length ? untracked : state.ends.map(e => e.number));
    $('#stop').hidden = !active;
    $('#first').value = state.first_delivery_end;
    $('#first').disabled = active;
    $('#job').innerHTML = active
      ? `<div class="banner"><b>${esc(job.stage || 'Starting')}</b><progress max="1" value="${job.progress || 0}"></progress></div>`
      : job && ['failed', 'interrupted'].includes(job.state)
        ? `<div class="banner warn"><b>${esc(job.error || job.stage)}</b>${job.ends.filter(e => e.error).map(e => `<span>End ${e.end_number}: ${esc(e.error)}</span>`).join('')}</div>`
        : '';
    $('#ends').innerHTML = state.ends.length ? `
      <thead><tr><th>End</th><th>In the video</th><th>Thrown from</th><th>Status</th><th></th></tr></thead>
      <tbody>${state.ends.map(end => {
        const queued = inJob.get(end.number);
        const notes = end.run ? checkNote(end.run) : [];
        const status = queued && queued.state !== 'complete'
          ? `<span class="pill busy">${queued.state === 'running' ? esc(queued.stage || 'Tracking') + ` · ${Math.round((queued.progress || 0) * 100)}%` : 'Waiting'}</span>`
          : end.run
            ? `<span class="pill ok">Tracked</span>${notes.length ? ` <span class="status warn">${esc(notes.join(' · '))}</span>` : ''}`
            : '<span class="pill">Not tracked</span>';
        return `<tr>
          <td><b>End ${end.number}</b></td>
          <td>${clock(end.start_seconds, 0)} – ${clock(end.end_seconds, 0)}</td>
          <td>House ${end.delivery_end}</td>
          <td>${status}</td>
          <td class="actions">
            ${end.run ? `<a href="${link(id, 'review', {end: end.number})}">Review shots</a>` : ''}
            <button class="quiet one" data-end="${end.number}" ${active ? 'disabled' : ''}>${end.run ? 'Track again' : 'Track'}</button>
          </td></tr>`;
      }).join('')}</tbody>` : `<tbody><tr><td class="muted">Split the recording into ends first, on the <a href="${link(id, 'ends')}">Ends</a> screen.</td></tr></tbody>`;
    container.querySelectorAll('.one').forEach(button => {
      button.onclick = () => start([Number(button.dataset.end)]);
    });
  }

  const refresh = async () => draw(await api(base));
  const start = ends => attempt(async () => { await api(base, 'POST', {ends}); await refresh(); });
  $('#all').onclick = () => start(JSON.parse($('#all').dataset.ends));
  $('#stop').onclick = () => attempt(async () => { await api(`${base}/stop`, 'POST'); await refresh(); });
  $('#first').onchange = () => attempt(async () => draw(await api(`${base}/settings`, 'PUT', {first_delivery_end: $('#first').value})));
  return poll(refresh, 2000);
}
