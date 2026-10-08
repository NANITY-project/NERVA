// Nebula Studio UI. DOM + network only; the scene graph lives in viewport.js and pure logic in logic.js.
// Dependencies are injected (boot(env)) so the whole UI can be driven headlessly in tests.
// SECURITY: character names, personas, speech and log text come from users and from a language model.
// They are only ever written with textContent / value, never innerHTML.
import { structureKey, fmtTime, completions, applyCompletion } from './logic.js';

export function h(doc, tag, attrs = {}, ...kids) {
  const el = doc.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'value') el.value = v;
    else if (k === 'checked') el.checked = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) el.append(kid.nodeType ? kid : doc.createTextNode(String(kid)));
  return el;
}

export async function boot(env) {
  const { document: doc, window: win, fetch: _fetch, EventSource: ES, createViewport } = env;   // env.viewportError: why the 3D module could not load
  const $ = id => doc.getElementById(id);
  const el = (tag, attrs, ...kids) => h(doc, tag, attrs, ...kids);
  const st = { snap: null, view: null, selected: null, log: [], takes: [], config: null, listKey: '', inspKey: '',
               play: null, playing: false, recording: false, selSeen: false, onModalClose: null, modelVer: {}, modelMode: {}, modelBusy: new Set(), neonSel: '', history: [], histIdx: -1, labels: new Map(), connected: false };
  let viewport = null, es = null, raf = 0, alive = true;

  // ------------------------------------------------------------------ network
  async function api(path, body, { raw, ctype } = {}) {
    try {
      const r = await _fetch(path, raw
        ? { method: 'POST', headers: { 'Content-Type': ctype || 'application/octet-stream' }, body: raw }
        : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
      const data = await r.json().catch(() => ({}));
      if (!r.ok || data.ok === false) { toast(data.error || `request failed (${r.status})`, 'error'); return null; }
      return data;
    } catch (e) { toast(`server unreachable: ${e.message}`, 'error'); return null; }
  }
  let toastTimer = 0;
  function toast(msg, kind = '') {
    const t = $('toast'); t.textContent = msg; t.className = 'toast ' + kind; t.hidden = false;
    win.clearTimeout(toastTimer); toastTimer = win.setTimeout(() => { t.hidden = true; }, 4200);
  }

  // ------------------------------------------------------------------ state in
  function setConn(ok) {
    st.connected = ok;
    $('conn').className = 'pill' + (ok ? ' ok' : '');
    $('conn-text').textContent = ok ? 'live' : 'reconnecting';
  }
  function applyConfig(c) {
    st.config = c;
    const m = $('mode'); m.textContent = c.mode === 'model' ? 'model' : 'mock agent';
    m.className = 'pill' + (c.mode === 'model' ? ' model' : '');
    m.title = c.mode === 'model' ? c.llm_url : 'No model attached: a built-in rule-based agent stands in. Open "Model" to connect yours.';
  }
  function applyTakes(takes) {
    st.takes = takes;
    const sel = $('takes'), prev = sel.value;
    sel.replaceChildren(...(takes.length ? takes.map(t => el('option', { value: String(t.id), text: `${t.name} · ${t.duration.toFixed(1)}s` })) : [el('option', { value: '', text: 'no takes' })]));
    if (prev && takes.some(t => String(t.id) === prev)) sel.value = prev;
    for (const id of ['btn-play', 'btn-export']) $(id).disabled = takes.length === 0;
  }
  function applyMeta(meta) {
    if (!meta || !meta.llm) return;
    const m = $('mode'); if (m.textContent !== meta.llm) m.textContent = meta.llm;
    m.className = 'pill' + (meta.mode && meta.mode !== 'mock' ? ' model' : '');
    m.title = meta.mode === 'neon' ? 'Driven by a local NEON model' : meta.mode === 'model' ? 'Driven by an external endpoint' : 'No model attached: a built-in rule-based agent stands in. Open "Model" to connect one.';
  }
  function onSnapshot(s) {
    st.snap = s; applyMeta(s.meta);
    $('empty-state').hidden = s.chars.length > 0 || s.props.length > 0 || !!st.play;
    if (!st.play) syncModels(s);
    if (!st.play) { st.view = s; viewport?.push(s); }
    $('simtime').textContent = fmtTime(s.t);
    if (s.recording !== st.recording) {
      st.recording = s.recording;
      $('btn-rec').classList.toggle('on', s.recording); $('rec-indicator').hidden = !s.recording;
      if (!s.recording) refreshTakes();
    }
    if (st.selected) {                      // a just-created item may not be in the stream yet: only drop it once it was seen
      if (exists(st.selected, s)) st.selSeen = true;
      else if (st.selSeen) select(null);
    }
    const key = structureKey(s);
    if (key !== st.listKey) { st.listKey = key; renderLists(); renderDirectorTargets(); renderInspector(false); }
    else if (st.selected?.type === 'char') updateInspectorStatus();
  }
  const exists = (sel, s) => sel.type === 'char' ? s.chars.some(c => c.id === sel.id) : s.props.some(p => p.name === sel.id);
  const charOf = id => st.snap?.chars.find(c => c.id === id);
  const propOf = id => st.snap?.props.find(p => p.name === id);

  async function refreshTakes() { const r = await _fetch('/api/takes').then(x => x.json()).catch(() => null); if (r) applyTakes(r); }

  // ------------------------------------------------------------------ log
  function addLog(e) {
    st.log.push(e); if (st.log.length > 300) st.log.shift();
    const box = $('log'), stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
    box.append(el('div', { class: `e ${e.kind}` }, el('span', { class: 't', text: e.t.toFixed(1) }), e.text));
    while (box.childNodes.length > 300) box.firstChild.remove();
    if (stick) box.scrollTop = box.scrollHeight;
  }

  // ------------------------------------------------------------------ lists / selection
  function select(sel) {
    st.selected = sel; st.selSeen = !!sel && !!st.snap && exists(sel, st.snap);
    viewport?.select(sel); st.inspKey = ''; renderLists(); renderInspector(true);
  }
  function renderLists() {
    const s = st.snap; if (!s) return;
    $('chars-list').replaceChildren(...s.chars.map(c => el('li', {
      class: 'item' + (st.selected?.type === 'char' && st.selected.id === c.id ? ' sel' : ''), 'data-char': c.id, onclick: () => select({ type: 'char', id: c.id }),
    }, el('span', { class: 'sw', style: `background:${/^#[0-9a-f]{6}$/i.test(c.color) ? c.color : '#a60000'}` }), el('span', { class: 'nm', text: c.name }),
       c.agent ? el('span', { class: 'tag', text: 'auto' }) : null, el('span', { class: 'st', text: c.busy || '' }))));
    $('props-list').replaceChildren(...s.props.map(p => el('li', {
      class: 'item' + (st.selected?.type === 'prop' && st.selected.id === p.name ? ' sel' : ''), 'data-prop': p.name, onclick: () => select({ type: 'prop', id: p.name }),
    }, el('span', { class: 'nm', text: p.name }), p.held ? el('span', { class: 'tag', text: 'held' }) : null,
       ...p.tags.slice(0, 2).map(t => el('span', { class: 'tag', text: t })))));
  }
  function renderDirectorTargets() {
    const sel = $('director-target'), prev = sel.value;
    sel.replaceChildren(el('option', { value: 'all', text: 'Everyone' }), ...(st.snap?.chars || []).map(c => el('option', { value: c.id, text: c.name })));
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
  }

  // ------------------------------------------------------------------ inspector
  function updateInspectorStatus() {
    const c = charOf(st.selected.id), s = $('insp-status'); if (!c || !s) return;
    s.textContent = c.busy ? `${c.busy}${c.queued ? ` (+${c.queued} queued)` : ''}` : 'idle';
    const b = $('insp-body');
    if (b) {
      const mode = st.modelMode[c.id];
      const t = c.model ? `${c.model_name || 'model'} · ${mode || 'loading…'}` : 'blacked-out entity (default)';
      if (b.textContent !== t) b.textContent = t;
      $('btn-model-clear').disabled = !c.model;
    }
  }

  // ------------------------------------------------------------------ character models (.glb)
  async function uploadModel(cid, file) {
    if (!file) return;
    if (file.size > 30 * 1024 * 1024) { toast('That model is larger than 30 MB', 'error'); return; }
    const r = await api(`/api/character/${cid}/model?name=${encodeURIComponent(file.name)}`, null, { raw: await file.arrayBuffer(), ctype: 'model/gltf-binary' });
    // no success toast: the Body label shows loading → rigged/static, and a failure toast must not be overwritten by a late success one
    return !!r;
  }
  // keep each character's body in step with the server (version number changes => refetch)
  function syncModels(s) {
    const ids = new Set(s.chars.map(c => c.id));
    for (const id of Object.keys(st.modelVer)) if (!ids.has(id)) { delete st.modelVer[id]; delete st.modelMode[id]; }
    for (const c of s.chars) {
      const want = c.model || 0, have = st.modelVer[c.id] || 0;
      if (want === have || st.modelBusy.has(c.id)) continue;
      st.modelBusy.add(c.id);
      (async () => {
        try {
          if (!want) { viewport?.clearCharacterModel?.(c.id); delete st.modelMode[c.id]; return; }
          const buf = await _fetch(`/api/model/${c.id}`).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); });
          if (!viewport?.setCharacterModel) { st.modelMode[c.id] = 'no viewport'; return; }
          const r = await viewport.setCharacterModel(c.id, buf);
          st.modelMode[c.id] = r.mode; if (r.note) toast(`${c.name}: ${r.note}`);
        } catch (e) {
          st.modelMode[c.id] = 'failed'; toast(`${c.name}: couldn't use that model (${e.message || e}) — showing the default body`, 'error');
          viewport?.clearCharacterModel?.(c.id);
        } finally { st.modelVer[c.id] = want; st.modelBusy.delete(c.id); }
      })();
    }
  }
  function renderInspector(force) {
    const box = $('inspector'), sel = st.selected;
    const key = sel ? `${sel.type}:${sel.id}` : '';
    if (!force && key === st.inspKey) return;
    if (!sel) { st.inspKey = ''; box.replaceChildren(el('p', { class: 'empty', text: 'Select a character or prop.' })); return; }
    const ent = sel.type === 'char' ? charOf(sel.id) : propOf(sel.id);
    if (!ent) return;                       // not in a snapshot yet: leave inspKey unset so the next snapshot renders it
    st.inspKey = key;
    return sel.type === 'char' ? renderCharInspector(box, ent) : renderPropInspector(box, ent);
  }
  function renderCharInspector(box, c) {
    if (!c) return;
    const post = body => api(`/api/character/${c.id}`, body);
    const cmd = el('input', { type: 'text', id: 'insp-cmd', placeholder: 'GRAB right cup', autocomplete: 'off', spellcheck: 'false' });
    const hint = el('div', { class: 'cmd-hint', id: 'cmd-hint' });
    const send = async () => {
      const line = cmd.value.trim(); if (!line) return;
      st.history.unshift(line); st.history.length = Math.min(st.history.length, 50); st.histIdx = -1;
      if (await api('/api/command', { character: c.id, line })) { cmd.value = ''; hint.textContent = ''; }
    };
    cmd.addEventListener('keydown', ev => {
      if (ev.key === 'Enter') { ev.preventDefault(); send(); }
      else if (ev.key === 'Tab') {
        ev.preventDefault(); const opts = completions(cmd.value, st.snap);
        if (opts.length === 1) { cmd.value = applyCompletion(cmd.value, opts[0]); hint.textContent = ''; }
        else hint.textContent = opts.join('   ');
      } else if (ev.key === 'ArrowUp' && st.history.length) { ev.preventDefault(); st.histIdx = Math.min(st.histIdx + 1, st.history.length - 1); cmd.value = st.history[st.histIdx]; }
      else if (ev.key === 'ArrowDown') { ev.preventDefault(); st.histIdx = Math.max(st.histIdx - 1, -1); cmd.value = st.histIdx < 0 ? '' : st.history[st.histIdx]; }
    });
    const quick = (label, line) => el('button', { class: 'btn-ghost btn-sm', 'data-quick': line, text: label, onclick: () => api('/api/command', { character: c.id, line }) });
    box.replaceChildren(
      el('div', { class: 'name', text: c.name }),
      el('div', { class: 'mono', id: 'insp-status', style: 'font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;color:var(--red-hi)', text: 'idle' }),
      el('div', { class: 'field' }, el('h2', { text: 'Persona' }),
        el('textarea', { id: 'insp-persona', value: c.persona, placeholder: 'Who is this character? How do they talk and behave?', maxlength: '600',
          onchange: ev => post({ persona: ev.target.value }) })),
      el('label', { class: 'check' }, el('input', { type: 'checkbox', id: 'insp-agent', checked: c.agent, onchange: ev => post({ agent: ev.target.checked }) }),
        'Act on their own (autonomous)'),
      el('div', { class: 'field' }, el('h2', { text: 'Body' }), el('div', { class: 'cmd-hint', id: 'insp-body' }),
        el('div', { class: 'row' },
          el('button', { class: 'btn-ghost btn-sm', id: 'btn-model', text: 'Choose .glb', onclick: () => $('file-model').click() }),
          el('button', { class: 'btn-ghost btn-sm', id: 'btn-model-clear', text: 'Blackout entity', onclick: () => api(`/api/character/${c.id}`, { clear_model: true }) }),
          el('input', { type: 'file', id: 'file-model', accept: '.glb,.gltf,model/gltf-binary', hidden: true, onchange: ev => { uploadModel(c.id, ev.target.files[0]); ev.target.value = ''; } }))),
      el('div', { class: 'field' }, el('h2', { text: 'Command' }), cmd, hint),
      el('div', { class: 'row' }, quick('Wave', 'WAVE right'), quick('Relax L', 'RESET left'), quick('Relax R', 'RESET right'), quick('Drop', 'DROP right'),
        el('button', { class: 'btn-ghost btn-sm', id: 'btn-stop', text: 'Stop', onclick: () => api('/api/stop', { character: c.id }) })),
      el('div', { class: 'row' }, el('button', { class: 'btn-ghost btn-sm', id: 'btn-del-char', text: 'Remove', onclick: async () => { if (await post({ delete: true })) select(null); } })),
    );
    updateInspectorStatus();
  }
  function renderPropInspector(box, p) {
    if (!p) return;
    const post = body => api(`/api/prop/${p.name}`, body);
    const num = (i) => el('input', { type: 'number', step: '0.05', value: String(+p.pos[i].toFixed(3)), 'data-axis': 'xyz'[i],
      onchange: ev => { const pos = [...propOf(p.name).pos]; pos[i] = parseFloat(ev.target.value); if (pos.every(Number.isFinite)) post({ pos }); } });
    box.replaceChildren(
      el('div', { class: 'name', text: p.name }),
      el('div', { class: 'mono', style: 'font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;color:var(--ivory-dim)', text: `${p.kind} · ${p.size.map(v => v.toFixed(2)).join(' × ')} m` }),
      el('div', { class: 'field' }, el('h2', { text: 'Position (x y z)' }), el('div', { class: 'row' }, num(0), num(1), num(2))),
      el('div', { class: 'field' }, el('h2', { text: 'Tags' }),
        el('input', { type: 'text', id: 'insp-tags', value: p.tags.join(', '), placeholder: 'fixed, surface, fragile',
          onchange: ev => post({ tags: ev.target.value.split(',').map(s => s.trim()).filter(Boolean) }) })),
      el('p', { class: 'cmd-hint', text: 'Tag "fixed" to stop characters grabbing it.' }),
      el('div', { class: 'row' }, el('button', { class: 'btn-ghost btn-sm', id: 'btn-del-prop', text: 'Delete', onclick: async () => { if (await post({ delete: true })) select(null); } })),
    );
  }

  // ------------------------------------------------------------------ modal forms
  function closeModal() { $('modal').hidden = true; $('modal-card').classList.remove('wide'); const f = st.onModalClose; st.onModalClose = null; if (f) f(); }
  function modal(title, fields, okLabel, onOk, extra) {
    if (!$('modal').hidden) closeModal();
    const card = $('modal-card'), inputs = {};
    const fieldEls = fields.map(f => {
      const inp = f.area ? el('textarea', { placeholder: f.placeholder, value: f.value || '' }) : el('input', { type: f.type || 'text', placeholder: f.placeholder, value: f.value || '', autocomplete: 'off' });
      if (f.type === 'checkbox') inp.checked = !!f.value;
      inputs[f.key] = inp;
      return f.type === 'checkbox' ? el('label', { class: 'check' }, inp, f.label) : el('div', { class: 'field' }, el('h2', { text: f.label }), inp);
    });
    const close = closeModal;
    card.replaceChildren(el('h3', { text: title }), ...fieldEls, extra || null,
      el('div', { class: 'btns' }, el('button', { class: 'btn-ghost btn-sm', id: 'modal-cancel', text: 'Cancel', onclick: close }),
        el('button', { class: 'btn-submit', id: 'modal-ok', text: okLabel, onclick: async () => {
          const vals = Object.fromEntries(Object.entries(inputs).map(([k, i]) => [k, i.type === 'checkbox' ? i.checked : i.value]));
          if (await onOk(vals) !== false) close();
        } })));
    $('modal').hidden = false; (Object.values(inputs)[0])?.focus?.();
  }
  function addCharacter() {
    modal('Add character', [{ key: 'name', label: 'Name', placeholder: 'Mara' }, { key: 'persona', label: 'Persona', area: true, placeholder: 'A retired sea captain. Gruff, but kind to strangers.' }],
      'Add', async v => { if (!v.name.trim()) { toast('Give them a name', 'error'); return false; }
        const n = st.snap?.chars.length || 0;
        const r = await api('/api/character', { name: v.name.trim(), persona: v.persona, x: -2 + 0.9 * n, z: -0.5 });
        if (r) select({ type: 'char', id: r.id }); return !!r; });
  }
  function addProp() {
    modal('Add prop', [{ key: 'name', label: 'Name', placeholder: 'lantern' }, { key: 'size', label: 'Size in metres (w h d)', value: '0.2 0.2 0.2' },
      { key: 'tags', label: 'Tags (comma separated)', placeholder: 'fixed, surface' }],
      'Add', async v => {
        const size = v.size.trim().split(/\s+/).map(Number);
        if (!v.name.trim() || size.length !== 3 || !size.every(x => Number.isFinite(x) && x > 0)) { toast('Name and three positive sizes are required', 'error'); return false; }
        const r = await api('/api/prop', { name: v.name.trim(), size, pos: [0, size[1] / 2 + 0.6, 1.0], tags: v.tags.split(',').map(s => s.trim()).filter(Boolean) });
        if (r) select({ type: 'prop', id: r.name }); return !!r; });
  }
  function openModelPanel() {
    if (!$('modal').hidden) closeModal();
    const card = $('modal-card'); card.classList.add('wide');
    const fmtSize = b => b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(0)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`;
    const stateLine = el('div', { class: 'neon-state', id: 'neon-state' }), binBox = el('div', { class: 'field', id: 'neon-bin' });
    const modelsBox = el('div', { class: 'models', id: 'neon-models' }), result = el('div', { class: 'cmd-hint', id: 'neon-result' }), procLog = el('pre', { class: 'logbox', id: 'neon-log', hidden: true });
    const threads = el('input', { type: 'number', id: 'neon-threads', min: '0', max: '256', value: '0', style: 'width:80px' });
    const dirIn = el('input', { type: 'text', id: 'neon-dir', placeholder: '/path/to/folder/with/models', autocomplete: 'off' });
    const sel = () => { const r = card.querySelector('input[name=neon-model]:checked'); return r ? r.value : ''; };
    const post = async (path, body) => { const r = await api(path, body); refresh(); return r; };
    const btnLaunch = el('button', { class: 'btn-submit', id: 'neon-launch', text: 'Launch', onclick: async () => {
      const m = sel(); if (!m) { toast('Pick a model first', 'error'); return; }
      st.neonSel = m; result.textContent = ''; await post('/api/neon/launch', { model: m, threads: parseInt(threads.value, 10) || 0 }); } });
    const btnStop = el('button', { class: 'btn-ghost btn-sm', id: 'neon-stop', text: 'Stop', onclick: () => post('/api/neon/stop', {}) });
    const btnCheck = el('button', { class: 'btn-ghost btn-sm', id: 'neon-check', text: 'Check model', onclick: async () => {
      const m = sel(); if (!m) { toast('Pick a model first', 'error'); return; }
      result.className = 'cmd-hint'; result.textContent = 'checking…'; const r = await api('/api/neon/probe', { model: m });
      if (r) { result.className = 'cmd-hint ' + (r.result.ok ? 'ok-text' : 'bad-text'); result.textContent = r.result.message; } else result.textContent = ''; } });
    const ext = {
      url: el('input', { type: 'text', id: 'ext-url', placeholder: 'http://127.0.0.1:8080', autocomplete: 'off' }), model: el('input', { type: 'text', id: 'ext-model', placeholder: 'model name' }),
      key: el('input', { type: 'password', id: 'ext-key', placeholder: 'API key (optional)' }), grammar: el('input', { type: 'checkbox', id: 'ext-grammar' }),
    };
    const c0 = st.config || {}; ext.url.value = c0.llm_url || ''; ext.model.value = c0.model || ''; ext.grammar.checked = !!c0.use_grammar;
    const saveExt = async () => {
      const body = { llm_url: ext.url.value, model: ext.model.value, use_grammar: ext.grammar.checked }; if (ext.key.value) body.api_key = ext.key.value;
      const r = await api('/api/config', body); if (r) { applyConfig(r.config); toast(r.config.llm_url ? 'External endpoint saved' : 'External endpoint cleared'); } };
    card.replaceChildren(
      el('h3', { text: 'Model' }),
      el('p', { class: 'small', text: 'Characters think with a language model. Studio looks for the NEON runtime on this computer and lists the models it can run. If a model is running here it is used; otherwise an external endpoint; otherwise a simple built-in agent.' }),
      stateLine, binBox,
      el('div', { class: 'field' }, el('div', { class: 'label-row' }, el('h2', { text: 'Models on this computer' }), el('button', { class: 'btn-ghost btn-sm', id: 'neon-rescan', text: 'Rescan', onclick: () => post('/api/neon/rescan', {}) })),
        modelsBox, el('div', { class: 'row' }, dirIn, el('button', { class: 'btn-ghost btn-sm', id: 'neon-adddir', text: 'Add folder', onclick: async () => { if (dirIn.value.trim() && await post('/api/neon/dirs', { dirs: [dirIn.value.trim()] })) dirIn.value = ''; } }))),
      el('div', { class: 'row', style: 'align-items:center' }, el('span', { class: 'cmd-hint', text: 'threads (0 = auto)' }), threads, btnCheck, btnLaunch, btnStop), result, procLog,
      el('details', { class: 'ext', id: 'ext-details' }, el('summary', { text: 'External endpoint (OpenAI-compatible: llama.cpp, vLLM…)' }), ext.url, ext.model, ext.key,
        el('label', { class: 'check' }, ext.grammar, 'Send a GBNF grammar with each request (llama.cpp-style runtimes; NEON ignores it)'),
        el('div', { class: 'row' }, el('button', { class: 'btn-ghost btn-sm', id: 'ext-save', text: 'Save endpoint', onclick: saveExt }))),
      el('div', { class: 'btns' }, el('button', { class: 'btn-ghost btn-sm', id: 'modal-cancel', text: 'Close', onclick: closeModal })));
    let keys = { bin: '', models: '' }, alive = true;
    function render(s) {
      const p = s.process;
      stateLine.className = 'neon-state ' + (p ? p.state : '');
      const base = p ? `${p.state}${p.model ? ' · ' + p.model.split('/').pop() : ''}` : 'NEON is not running';
      const extra = p && p.state === 'ready' ? ` · context ${p.config.ctx_len || '?'} · ${p.requests} request${p.requests === 1 ? '' : 's'}${p.last_ms ? ` · last reply ${p.last_ms} ms` : ''}` : p && p.state === 'error' ? ` — ${p.error}` : '';
      stateLine.replaceChildren(el('i', { class: 'dot' }), el('span', { text: base + extra }));
      const bk = JSON.stringify([s.binary, s.source, s.compiler, s.build.state, s.build.error]);
      if (bk !== keys.bin) {
        keys.bin = bk; const row = [];
        if (s.binary) row.push(el('div', { class: 'cmd-hint ok-text', id: 'neon-bin-ok', text: `✓ NEON runtime: ${s.binary}` }));
        else if (s.source && s.compiler) row.push(el('p', { class: 'small', text: `NEON source found at ${s.source}, but it isn't built yet.` }),
          el('div', { class: 'row' }, el('button', { class: 'btn-submit', id: 'neon-build', text: s.build.state === 'building' ? 'Building…' : 'Build NEON', disabled: s.build.state === 'building', onclick: () => post('/api/neon/build', {}) })));
        else if (s.source) row.push(el('p', { class: 'small bad-text', text: `NEON source found at ${s.source}, but no C++ compiler (install g++ or clang++ to build it).` }));
        else row.push(el('p', { class: 'small', text: 'NEON was not found. Clone github.com/NANITY-project/NEON-Runtime-Beta-Variation next to Studio (it will be built from source here), or enter the path to an existing NEON binary:' }));
        if (s.build.error) row.push(el('p', { class: 'small bad-text', text: `Build failed: ${s.build.error}` }));
        if (s.build.state !== 'idle') row.push(el('pre', { class: 'logbox', id: 'neon-buildlog' }));
        const pathIn = el('input', { type: 'text', id: 'neon-bin-path', placeholder: '/path/to/neon', autocomplete: 'off' });
        row.push(el('div', { class: 'row' }, pathIn, el('button', { class: 'btn-ghost btn-sm', id: 'neon-bin-set', text: 'Use this binary', onclick: async () => { if (pathIn.value.trim() && await post('/api/neon/binary', { path: pathIn.value.trim() })) pathIn.value = ''; } })));
        binBox.replaceChildren(el('h2', { text: 'Runtime' }), ...row);
      }
      const bl = $('neon-buildlog'); if (bl) { bl.textContent = s.build.log.join('\n'); bl.scrollTop = bl.scrollHeight; }
      if (!st.neonSel) st.neonSel = (s.models.find(m => m.path === s.settings.last_model && m.runnable) || s.models.find(m => m.runnable) || {}).path || '';
      const mk = JSON.stringify([s.models.map(m => m.path + m.runnable), st.neonSel]);
      if (mk !== keys.models) {
        keys.models = mk; threads.value = threads.value === '0' ? String(s.settings.threads || 0) : threads.value;
        modelsBox.replaceChildren(...(s.models.length ? s.models.map(m => el('label', { class: 'model-row' + (m.runnable ? '' : ' off'), 'data-model': m.file },
          el('input', { type: 'radio', name: 'neon-model', value: m.path, checked: m.path === st.neonSel, disabled: !m.runnable, onchange: () => { st.neonSel = m.path; } }),
          el('div', {}, el('div', {}, m.name, m.runnable ? el('span', { class: 'badge good', text: 'nanity ✓' }) : el('span', { class: 'badge', text: m.arch || 'unusable' })),
            el('div', { class: 'meta', text: [m.file, fmtSize(m.size), m.quant, m.ctx_len ? `ctx ${m.ctx_len}` : ''].filter(Boolean).join(' · ') }), m.note ? el('div', { class: 'why', text: m.note }) : null)))
          : [el('p', { class: 'empty', text: 'No .gguf models found. Add a folder that contains them.' })]));
      }
      btnLaunch.disabled = !s.binary || (p && p.state === 'starting'); btnStop.disabled = !p; btnCheck.disabled = !s.binary;
      procLog.hidden = !p; if (p) { procLog.textContent = p.log.join('\n'); procLog.scrollTop = procLog.scrollHeight; }
    }
    async function refresh() { if (!alive) return; const s = await _fetch('/api/neon').then(r => r.json()).catch(() => null); if (s && alive) render(s); }
    const timer = win.setInterval(refresh, 1000);
    st.onModalClose = () => { alive = false; win.clearInterval(timer); };
    $('modal').hidden = false; refresh();
  }

  // ------------------------------------------------------------------ guide
  function openGuide() {
    if (!$('modal').hidden) closeModal();
    const card = $('modal-card'); card.classList.add('wide');
    const step = (n, title, body, btn) => el('div', { class: 'step' }, el('span', { class: 'n', text: String(n) }), el('div', {}, el('strong', { text: title }), el('p', { text: body }), btn || null));
    const go = (label, id, fn) => el('button', { class: 'btn-ghost btn-sm', id, text: label, onclick: () => { closeModal(); fn(); } });
    const cmds = [['SAY <text>', 'speak aloud'], ['WALK_TO <name>', 'walk to a prop or person'], ['GRAB <hand> <prop>', 'pick something up (walks over, crouches if needed)'], ['OFFER <hand> <person>', 'hold out what you carry'],
      ['DROP <hand>', 'put it down'], ['REACH / POINT <hand> <name>', 'touch or point at something'], ['WAVE <hand>', 'wave'], ['LOOK_AT <name>', 'turn the eyes'], ['WAIT <seconds>', 'pause'], ['RESET <hand>', 'relax an arm']];
    card.replaceChildren(
      el('h3', { text: 'Guide' }),
      el('p', { class: 'small', text: 'Nebula Studio is a stage where every character is driven by a language model: they see the scene, decide, and act through the IK engine. Direct them with plain text, record takes, and export the motion.' }),
      step(1, 'Connect a brain', 'Open Model. Studio looks for the NEON runtime on this computer and lists the models it can run. With none attached, a simple built-in agent stands in so you can try everything.', go('Open Model', 'guide-model', openModelPanel)),
      step(2, 'Add characters', 'Give each a name and a persona. They start as blacked-out entities; select one and choose a .glb to give it a body (humanoid skeletons from Mixamo, VRM/VRoid and Blender rigs animate; anything else becomes a static figure).', go('Add character', 'guide-char', addCharacter)),
      step(3, 'Build a stage', 'Import a .glb environment (exported from Blender or Unreal), or add props by hand. Every object becomes something characters can name. Tag a prop "fixed" to stop it being picked up.', go('Import environment', 'guide-import', () => $('file-import').click())),
      step(4, 'Direct', 'Type an instruction in the bar at the bottom, or tick "Act on their own" on a character. Press Rec to record a take, then scrub and export it.'),
      el('h2', { style: 'margin-top:8px', text: 'What characters can do' }),
      el('div', { class: 'cmdgrid' }, ...cmds.flatMap(([c, d]) => [el('code', { text: c }), el('span', { text: d })])),
      el('div', { class: 'step' }, el('span', { class: 'n', text: '★' }), el('div', {}, el('strong', { text: 'Try it' }), el('p', { text: 'Load a small demo: Alice the baker and Bob the customer, a table, a cup, a teapot. Then direct Alice: "give the cup to Bob".' }),
        go('Load demo scene', 'guide-demo', async () => { select(null); goLive(); await api('/api/reset', { demo: true }); }))),
      el('div', { class: 'btns' }, el('button', { class: 'btn-ghost btn-sm', id: 'modal-cancel', text: 'Close', onclick: closeModal })));
    $('modal').hidden = false;
  }

  // ------------------------------------------------------------------ director / timeline
  async function direct() {
    const inp = $('director-text'), text = inp.value.trim(); if (!text) return;
    if (await api('/api/direct', { text, character: $('director-target').value })) inp.value = '';
  }
  async function toggleRecord() {
    if (!st.recording) await api('/api/record', { action: 'start' });
    else { const r = await api('/api/record', { action: 'stop' }); if (r) { toast(`Take saved · ${r.duration.toFixed(1)} s`); await refreshTakes(); } }
  }
  async function loadTake() {
    const id = $('takes').value; if (id === '') return null;
    const take = await _fetch(`/api/take/${id}`).then(r => r.json()).catch(() => null);
    if (!take || !take.frames.length) { toast('That take is empty', 'error'); return null; }
    st.play = { take, idx: 0, t0: 0, from: 0 };
    $('scrub').max = String(take.frames.length - 1); $('scrub').value = '0'; $('scrub').disabled = false; $('btn-live').disabled = false;
    showFrame(0); return take;
  }
  function showFrame(i) {
    const p = st.play; if (!p) return;
    p.idx = Math.max(0, Math.min(i, p.take.frames.length - 1));
    st.view = { ...p.take.frames[p.idx], props: p.take.frames[p.idx].props };
    viewport?.freeze(st.view);
    $('scrub').value = String(p.idx); $('scrub-time').textContent = `${fmtTime(st.view.t)} / ${fmtTime(p.take.duration)}`;
  }
  async function togglePlay() {
    if (st.playing) { st.playing = false; $('btn-play').textContent = 'Play'; return; }
    if (!st.play && !(await loadTake())) return;
    if (st.play.idx >= st.play.take.frames.length - 1) showFrame(0);
    st.playing = true; $('btn-play').textContent = 'Pause';
    st.play.t0 = win.performance.now(); st.play.from = st.play.idx;
  }
  function goLive() {
    st.playing = false; st.play = null; $('btn-play').textContent = 'Play'; $('scrub').disabled = true; $('btn-live').disabled = true; $('scrub-time').textContent = '—';
    viewport?.freeze(null); st.view = st.snap;
  }

  // ------------------------------------------------------------------ environment import
  async function importFile(file) {
    if (!file) return;
    const buf = await file.arrayBuffer();
    const r = await api('/api/import', null, { raw: buf, ctype: 'model/gltf-binary' });
    if (!r) return;
    let note = '';
    if (viewport) { try { await viewport.loadEnvironment(buf); } catch (e) { note = ' (props imported; mesh preview failed — .glb works best)'; } }
    toast(`Environment: ${r.props.length} props${r.skipped.length ? `, ${r.skipped.length} structural skipped` : ''}${note}`);
  }

  // ------------------------------------------------------------------ labels (names, speech, prop names)
  function updateLabels() {
    const v = st.view, layer = $('labels'); if (!v || !viewport) return;
    const want = new Set();
    const place = (key, cls, text, world, extra) => {
      want.add(key);
      let n = st.labels.get(key);
      if (!n) { n = el('div', { class: `lbl ${cls}` }); layer.append(n); st.labels.set(key, n); }
      if (n.textContent !== text) n.textContent = text;
      n.className = `lbl ${cls}${extra || ''}`;
      const p = viewport.project(world);
      n.style.display = p.visible ? '' : 'none'; n.style.left = `${p.x}px`; n.style.top = `${p.y}px`;
    };
    for (const c of v.chars) {
      const head = viewport.headOf(c.id) || [c.pos[0], 1.7, c.pos[2]];
      place(`n:${c.id}`, 'name', c.name, [head[0], head[1] + 0.2, head[2]]);
      if (c.speech) {
        want.add(`s:${c.id}`);
        let b = st.labels.get(`s:${c.id}`);
        if (!b) { b = el('div', { class: 'bubble' }); layer.append(b); st.labels.set(`s:${c.id}`, b); }
        if (b.textContent !== c.speech) b.textContent = c.speech;
        const p = viewport.project([head[0], head[1] + 0.38, head[2]]);
        b.style.display = p.visible ? '' : 'none'; b.style.left = `${p.x}px`; b.style.top = `${p.y}px`;
      }
    }
    for (const p of v.props) place(`p:${p.name}`, 'prop', p.name, [p.pos[0], p.pos[1] + p.size[1] / 2 + 0.05, p.pos[2]], st.selected?.type === 'prop' && st.selected.id === p.name ? ' sel' : '');
    for (const [k, n] of st.labels) if (!want.has(k)) { n.remove(); st.labels.delete(k); }
  }

  function tick(now) {
    if (!alive) return;
    raf = win.requestAnimationFrame(tick);
    if (st.playing && st.play) {
      const p = st.play, i = p.from + Math.floor((win.performance.now() - p.t0) / 1000 * p.take.fps);
      if (i >= p.take.frames.length - 1) { showFrame(p.take.frames.length - 1); st.playing = false; $('btn-play').textContent = 'Play'; }
      else if (i !== p.idx) showFrame(i);
    }
    updateLabels();
  }

  // ------------------------------------------------------------------ viewport interaction
  function wireViewport() {
    const cv = $('viewport'); let down = null;
    cv.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
    cv.addEventListener('pointerup', async e => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) { down = null; return; }
      down = null;
      const hit = viewport.pick(e.clientX, e.clientY), me = st.selected?.type === 'char' ? st.selected.id : null;
      if (e.altKey && hit?.type === 'prop' && me) { await api('/api/command', { character: me, line: `GRAB right ${hit.id}` }); return; }
      if (hit) { select(hit); return; }
      if (e.shiftKey && me) { const g = viewport.ground(e.clientX, e.clientY); if (g) await api('/api/command', { character: me, line: `WALK_TO ${g[0].toFixed(2)} ${g[1].toFixed(2)}` }); return; }
      select(null);
    });
    const drop = $('drop'), stage = cv.parentElement;
    stage.addEventListener('dragover', e => { e.preventDefault(); drop.hidden = false; });
    stage.addEventListener('dragleave', () => { drop.hidden = true; });
    stage.addEventListener('drop', e => { e.preventDefault(); drop.hidden = true; importFile(e.dataTransfer?.files?.[0]); });
  }

  // ------------------------------------------------------------------ wire up
  const on = (id, ev, fn) => $(id).addEventListener(ev, fn);
  on('btn-add-char', 'click', addCharacter); on('btn-add-prop', 'click', addProp); on('btn-settings', 'click', openModelPanel); on('btn-guide', 'click', openGuide);
  on('es-char', 'click', addCharacter); on('es-import', 'click', () => $('file-import').click()); on('es-guide', 'click', openGuide);
  on('btn-clear', 'click', () => modal('Clear scene', [], 'Clear', async () => { select(null); goLive(); return !!(await api('/api/reset', { demo: false })); },
    el('p', { class: 'small', text: 'Removes every character and prop and discards recorded takes. (The Guide can load a demo scene again.)' })));
  on('btn-import', 'click', () => $('file-import').click());
  on('file-import', 'change', e => { importFile(e.target.files[0]); e.target.value = ''; });
  on('btn-direct', 'click', direct);
  on('director-text', 'keydown', e => { if (e.key === 'Enter') { e.preventDefault(); direct(); } });
  on('btn-rec', 'click', toggleRecord); on('btn-play', 'click', togglePlay); on('btn-live', 'click', goLive);
  on('takes', 'change', () => { st.play = null; st.playing = false; $('btn-play').textContent = 'Play'; });
  on('scrub', 'input', e => { if (!st.play) return; st.playing = false; $('btn-play').textContent = 'Play'; showFrame(parseInt(e.target.value, 10)); });
  on('btn-export', 'click', () => { const id = $('takes').value; if (id !== '') win.open(`/api/take/${id}?download=1`, '_blank'); });
  on('modal', 'click', e => { if (e.target === $('modal')) closeModal(); });
  doc.addEventListener('keydown', e => { if (e.key === 'Escape') { if (!$('modal').hidden) closeModal(); else select(null); } });

  const noViewport = why => {
    viewport = null; console.error('3D viewport unavailable:', why);
    $('viewport-note').textContent = `3D viewport unavailable: ${why}. Everything else still works (open the browser console, F12, for details).`; $('viewport-note').hidden = false;
  };
  if (createViewport) {
    try { viewport = createViewport($('viewport')); wireViewport(); }
    catch (e) { noViewport(e && e.message || String(e)); }
  } else noViewport(env.viewportError || 'the 3D module is missing');

  // initial state, then the live stream
  const first = await _fetch('/api/state').then(r => r.json()).catch(() => null);
  if (first) { applyConfig(first.config); applyTakes(first.takes); first.log.forEach(addLog); onSnapshot(first.snapshot); }
  es = new ES('/api/stream');
  es.onopen = () => setConn(true);
  es.onerror = () => setConn(false);
  es.onmessage = ev => { try { onSnapshot(JSON.parse(ev.data)); } catch (e) { console.warn('bad snapshot', e); } };
  es.addEventListener('log', ev => { try { addLog(JSON.parse(ev.data)); } catch (e) { console.warn('bad log', e); } });
  raf = win.requestAnimationFrame(tick);

  return { state: st, api, select, importFile, uploadModel, openModelPanel, openGuide, destroy() { alive = false; win.cancelAnimationFrame(raf); es?.close(); viewport?.dispose?.(); } };
}
