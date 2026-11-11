// All recordings, and the form that adds one.
import {api, attempt, duration, esc, link, poll} from './util.js';

const BUSY = ['queued', 'running'];

function card(r) {
  const busy = BUSY.includes(r.state);
  const facts = [
    r.media ? duration(r.media.duration) : null,
    r.ends?.length ? `${r.ends.length} ends` : null,
  ].filter(Boolean).join(' · ');
  const problem = r.state === 'failed' || r.state === 'interrupted';
  return `<a class="card recording" href="${link(r.id, '')}">
    <h3>${esc(r.title)}</h3>
    <p class="muted">${esc(facts || 'Not downloaded yet')}</p>
    ${busy ? `<p class="status">${esc(r.stage)}</p><progress max="1" value="${r.progress || 0}"></progress>` : ''}
    ${problem ? `<p class="status warn">${esc(r.error || r.stage)}</p>` : ''}
  </a>`;
}

export function mount(container, {recordings}) {
  container.innerHTML = `
    <section class="add">
      <h1>Recordings</h1>
      <form id="add" class="row">
        <label class="grow">YouTube link
          <input id="url" type="url" required placeholder="https://www.youtube.com/watch?v=…">
        </label>
        <label>Ends played <span class="muted">(optional)</span>
          <input id="expected" type="number" min="1" max="99" placeholder="any">
        </label>
        <button type="submit">Add recording</button>
      </form>
      <p id="vision" class="muted"></p>
    </section>
    <div id="list" class="cards"></div>`;
  const list = container.querySelector('#list');
  const draw = rows => {
    list.innerHTML = rows.length ? rows.map(card).join('')
      : '<div class="empty"><h2>No recordings yet</h2><p>Paste the link to a finished YouTube stream above. It is downloaded and split into ends; then you can track the stones and review every shot.</p></div>';
  };
  draw(recordings);
  api('/status').then(s => {
    if (!s.vision_configured) container.querySelector('#vision').textContent = 'Automatic end detection needs OPENAI_API_KEY in .env. Without it you can still set the cuts by hand.';
  }).catch(() => {});
  container.querySelector('#add').onsubmit = event => {
    event.preventDefault();
    attempt(async () => {
      const expected = container.querySelector('#expected').value;
      const record = await api('/recordings', 'POST', {
        url: container.querySelector('#url').value,
        expected_ends: expected ? Number(expected) : null,
      });
      location.hash = link(record.id, 'ends');
    });
  };
  return poll(async () => draw(await api('/recordings')), 4000);
}
