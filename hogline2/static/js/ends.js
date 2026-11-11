// Split the recording into ends: watch the video, adjust the cuts, export a clip per end.
import {api, attempt, clock, esc, parseClock, poll, toast} from './util.js';

const BUSY = ['queued', 'running'];

export function mount(container, {id}) {
  const base = `/recordings/${encodeURIComponent(id)}`;
  let record = null, ends = [], dirty = false, drawnKey = '';

  container.innerHTML = `
    <div id="job"></div>
    <div class="split">
      <section>
        <video id="player" controls preload="metadata"></video>
        <p class="muted">Playhead <b id="playhead">0:00.0</b> · scrub to a cut, then press <b>Set</b> on the start or finish it belongs to.</p>
      </section>
      <section>
        <div class="row between">
          <h2>Ends</h2>
          <span id="origin" class="muted"></span>
        </div>
        <div id="ends" class="ends"></div>
        <div class="row">
          <button id="add" class="quiet">Add end</button>
          <span class="grow"></span>
          <button id="save" disabled>Save cuts</button>
          <button id="export" class="quiet">Export clips</button>
        </div>
        <p class="muted">An end runs from the start of its first delivery until the stones are cleared. Clips are cut without re-encoding, so they start and finish on the nearest keyframes outside your cuts.</p>
        <details><summary>More</summary>
          <div class="row">
            <button id="analyze" class="quiet">Detect ends again</button>
            <a id="manifest" target="_blank">Manifest (JSON)</a>
          </div>
        </details>
      </section>
    </div>`;
  const $ = selector => container.querySelector(selector);
  const player = $('#player');
  player.ontimeupdate = () => { $('#playhead').textContent = clock(player.currentTime); };
  $('#manifest').href = `/api${base}/manifest`;

  function collect() {
    container.querySelectorAll('.end').forEach((row, i) => {
      ends[i].start_seconds = parseClock(row.querySelector('.start').value);
      ends[i].end_seconds = parseClock(row.querySelector('.finish').value);
    });
  }

  function drawJob() {
    const busy = BUSY.includes(record.state);
    const problem = ['failed', 'interrupted'].includes(record.state);
    $('#job').innerHTML = busy
      ? `<div class="banner"><b>${esc(record.stage)}</b><progress max="1" value="${record.progress || 0}"></progress></div>`
      : problem
        ? `<div class="banner warn"><b>${esc(record.error || record.stage)}</b><button id="retry">Retry</button></div>`
        : (record.warnings || []).length
          ? `<details class="banner"><summary>${record.warnings.length} note${record.warnings.length === 1 ? '' : 's'} from end detection</summary>${record.warnings.map(w => `<p>${esc(w)}</p>`).join('')}</details>`
          : '';
    $('#retry')?.addEventListener('click', () => action('retry'));
    for (const button of container.querySelectorAll('#add, #export, #analyze, .end button, .end input')) button.disabled = busy;
    $('#save').disabled = busy || !dirty;
  }

  function drawEnds() {
    const cut = (kind, label, value) => `
      <label>${label}
        <span class="cut">
          <input class="${kind}" value="${clock(value)}" inputmode="decimal" aria-label="End ${label.toLowerCase()} time">
          <button class="quiet set" data-kind="${kind}" title="Use the video playhead">Set</button>
          <button class="quiet jump" data-kind="${kind}" title="Move the video here">Go</button>
        </span>
      </label>`;
    $('#ends').innerHTML = ends.length ? ends.map((end, i) => {
      const clip = record.clips.find(c => c.number === end.number && c.revision === record.revision);
      return `<article class="end card" data-i="${i}">
        <div class="row between">
          <h3>End ${end.number}</h3>
          <span class="muted">${clock(end.end_seconds - end.start_seconds, 0)} long</span>
          <button class="quiet remove" title="Remove this end">Remove</button>
        </div>
        <div class="row">${cut('start', 'Start', end.start_seconds)}${cut('finish', 'Finish', end.end_seconds)}</div>
        ${(end.warnings || []).map(w => `<p class="status warn">${esc(w)}</p>`).join('')}
        <div class="row">
          ${clip ? `<a href="/api${base}/clips/${end.number}" target="_blank">Open clip</a><a href="/api${base}/clips/${end.number}" download="end-${end.number}.mp4">Download</a>` : '<span class="muted">No clip for these cuts yet</span>'}
          ${end.evidence ? `<details class="grow"><summary>Why these cuts</summary><p class="muted">${esc(end.evidence)}</p></details>` : ''}
        </div>
      </article>`;
    }).join('') : '<p class="muted">No ends yet. Detect them, or add them by hand.</p>';
    container.querySelectorAll('.end').forEach(row => {
      const i = Number(row.dataset.i);
      row.querySelectorAll('input').forEach(input => { input.oninput = () => { dirty = true; drawJob(); }; });
      row.querySelectorAll('.set').forEach(button => {
        button.onclick = () => {
          row.querySelector('.' + button.dataset.kind).value = clock(player.currentTime);
          dirty = true; drawJob();
        };
      });
      row.querySelectorAll('.jump').forEach(button => {
        button.onclick = () => {
          const seconds = parseClock(row.querySelector('.' + button.dataset.kind).value);
          if (Number.isFinite(seconds)) player.currentTime = seconds;
        };
      });
      row.querySelector('.remove').onclick = () => {
        collect();
        ends.splice(i, 1);
        ends.forEach((end, j) => { end.number = j + 1; });
        dirty = true; drawEnds(); drawJob();
      };
    });
  }

  async function save() {
    collect();
    if (ends.some(e => !Number.isFinite(e.start_seconds) || !Number.isFinite(e.end_seconds))) {
      throw Error('Times look like 12:34.5 (or 1:02:03.5 past the hour).');
    }
    record = await api(`${base}/ends`, 'PUT', {revision: record.revision, ends});
    dirty = false; drawnKey = '';
    toast('Cuts saved');
    await refresh();
  }

  async function action(name) {
    await attempt(async () => {
      await api(`${base}/${name}`, 'POST');
      drawnKey = '';
      await refresh();
    });
  }

  async function refresh() {
    const latest = await api(base);
    if (!player.src && latest.source_path) player.src = `/api${base}/source`;
    if (dirty) { record = {...record, state: latest.state, stage: latest.stage, progress: latest.progress, error: latest.error}; drawJob(); return; }
    record = latest;
    const key = [latest.revision, latest.state, latest.clips.length].join(':');
    if (key !== drawnKey) {
      drawnKey = key;
      ends = structuredClone(latest.ends);
      $('#origin').textContent = !ends.length ? '' : latest.boundary_origin === 'user_edited' ? 'Edited by hand' : 'Detected automatically — check them';
      drawEnds();
    }
    drawJob();
  }

  $('#add').onclick = () => {
    collect();
    const last = ends.at(-1)?.end_seconds || 0, total = record.media?.duration || 0;
    if (last >= total) { toast('No video is left after the last end', 'error'); return; }
    ends.push({number: ends.length + 1, start_seconds: last, end_seconds: total, evidence: '', warnings: []});
    dirty = true; drawEnds(); drawJob();
  };
  $('#save').onclick = () => attempt(save);
  $('#export').onclick = () => attempt(async () => { if (dirty) await save(); await action('export'); });
  $('#analyze').onclick = () => {
    if (confirm('Detect the ends again? This replaces the current cuts.')) { dirty = false; action('analyze'); }
  };
  const warnUnsaved = event => { if (dirty) event.preventDefault(); };
  window.addEventListener('beforeunload', warnUnsaved);
  const stop = poll(refresh, 3000);
  return () => { stop(); window.removeEventListener('beforeunload', warnUnsaved); };
}
