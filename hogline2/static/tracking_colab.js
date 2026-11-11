// The Colab workflow deliberately uses files; a local Hogline server needs no public tunnel.
export function mountColab({base, say, fail, poll, openRun}) {
  const $ = s => document.querySelector(s);
  const execution = $('#execution');
  const preference = 'hogline2.tracking.execution';
  let signature = '', importing = false;
  try {
    const saved = localStorage.getItem(preference);
    if (saved === 'colab' || saved === 'local' || saved === 'stonetrack') execution.value = saved;
  } catch {
    // The selector still works if the browser blocks storage.
  }
  function showExecution() {
    const cloud = execution.value === 'colab';
    $('#colab-model-label').hidden = !cloud;
    $('#colab-explanation').hidden = !cloud;
    const stonetrack = execution.value === 'stonetrack';
    const note = $('#stonetrack-explanation');
    if (note) note.hidden = !stonetrack;
    $('#fps').disabled = stonetrack;
    $('#run').textContent = cloud ? 'Prepare Colab run' : stonetrack ? 'Track with stonetrack' : 'Track on this Mac';
  }
  execution.onchange = () => {
    showExecution();
    try {
      localStorage.setItem(preference, execution.value);
    } catch {
      // Keep the choice for the current page.
    }
  };
  showExecution();
  return function render(runs) {
    const pending = runs.filter(r => r.execution === 'colab' && r.state === 'awaiting_colab');
    const nextSignature = pending.map(r => r.run_id).join(',');
    if (importing || nextSignature === signature) return;
    signature = nextSignature;
    $('#colab-runs').innerHTML = pending.map(r => {
      const prefix = '/api' + base + '/runs/' + r.run_id + '/colab/';
      return `<article class="colab-run" data-colab-run="${r.run_id}">
        <h3>Colab · end ${r.end_number} · ${r.duration.toFixed(0)} seconds</h3>
        <ol>
          <li>Download <a href="${prefix}notebook">the notebook</a> and <a href="${prefix}bundle">the video bundle</a>.</li>
          <li><a href="https://colab.research.google.com/" target="_blank" rel="noopener">Open Colab</a>, choose <strong>File → Upload notebook</strong> and select <strong>hogline-colab.ipynb</strong>. Select a GPU under <strong>Runtime → Change runtime type</strong>, then run the cells and upload <strong>colab-input.zip</strong> when prompted.</li>
          <li>After Colab downloads <strong>colab-results.zip</strong>, import it here: <input aria-label="Colab result ZIP for end ${r.end_number}" type="file" accept=".zip,application/zip" data-results></li>
        </ol>
        <p data-import-status role="status"></p>
        <button class="secondary" data-cancel>Cancel this Colab run</button>
        <span class="muted">Cancelling here does not stop the Colab runtime.</span>
      </article>`;
    }).join('');
    $('#colab-runs').querySelectorAll('[data-colab-run]').forEach(card => {
      const run = card.dataset.colabRun;
      card.querySelector('[data-cancel]').onclick = async () => {
        try {
          const response = await fetch('/api' + base + '/runs/' + run + '/stop', {method: 'POST'});
          const result = await response.json();
          if (!response.ok) throw Error(result.detail);
          await poll();
        } catch (e) { fail(e); }
      };
      card.querySelector('[data-results]').onchange = async e => {
        const file = e.target.files[0];
        if (!file) return;
        e.target.disabled = true;
        importing = true;
        const status = card.querySelector('[data-import-status]');
        status.textContent = 'Uploading and checking Colab results…';
        try {
          const response = await fetch('/api' + base + '/runs/' + run + '/colab/results', {
            method: 'POST', headers: {'Content-Type': 'application/zip'}, body: file,
          });
          const result = await response.json();
          if (!response.ok) throw Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
          importing = false;
          say('Colab results imported. Review the masks and identities below.');
          await poll();
          await openRun(run);
        } catch (error) {
          importing = false;
          status.textContent = error.message;
          e.target.disabled = false;
          e.target.value = '';
        }
      };
    });
  };
}
