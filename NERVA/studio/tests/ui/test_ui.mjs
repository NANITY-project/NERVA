// Full-stack UI test: real Python server (subprocess) + the real app.js running in jsdom + a mock viewport.
import { JSDOM, VirtualConsole } from 'jsdom';
import { spawn } from 'node:child_process';
import { readFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import assert from 'node:assert/strict';
import net from 'node:net';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const STUDIO = path.resolve(HERE, '../..');
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function until(fn, ms = 20000, what = 'condition') { const t = Date.now(); for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t > ms) throw new Error('timeout: ' + what); await sleep(50); } }
const freePort = () => new Promise(r => { const s = net.createServer().listen(0, () => { const p = s.address().port; s.close(() => r(p)); }); });

const port = await freePort(), base = `http://127.0.0.1:${port}`;
// a throwaway home for settings + two tiny models (one valid NANITY, one wrong architecture) so the Model panel has something real to find
const tmp = mkdtempSync(path.join(tmpdir(), 'studio-ui-')), modelDir = path.join(tmp, 'models');
spawnSync('python3', ['-c', `import sys,os; sys.path.insert(0,'${path.join(STUDIO, 'tests')}'); import make_tiny_gguf as m; os.makedirs('${modelDir}'); m.build('${modelDir}/good.gguf', ctx=4096); m.build('${modelDir}/llama.gguf', arch='llama')`]);
const NEON_BIN = process.env.NEON_BIN || '', HAVE_NEON = !!NEON_BIN && (await import('node:fs')).existsSync(NEON_BIN);   // Makefile auto-detects it; the NEON scenarios skip without one
const server = spawn('python3', [path.join(STUDIO, 'server.py'), '--port', String(port), '--no-open'],      // NOTE: no --demo: the scene starts empty
  { env: { ...process.env, NEBULA_LIB: path.resolve(STUDIO, '../build/libnebula.so'), NEBULA_STUDIO_CONFIG: path.join(tmp, 'settings.json'), NEBULA_MODEL_DIRS: modelDir, ...(HAVE_NEON ? { NEON_BIN } : {}) }, stdio: ['ignore', 'pipe', 'pipe'] });
let serverErr = ''; server.stderr.on('data', d => { serverErr += d; });
await until(() => fetch(base + '/api/state').then(r => r.ok).catch(() => false), 15000, 'server start');

// ---- jsdom page from the real index.html (scripts not executed; we boot app.js ourselves)
const errors = [];
const vc = new VirtualConsole(); vc.on('jsdomError', e => errors.push(String(e))); vc.on('error', e => errors.push(String(e)));
const dom = new JSDOM(readFileSync(path.join(STUDIO, 'web/index.html'), 'utf8'), { pretendToBeVisual: true, virtualConsole: vc, url: base + '/' });
const { window: win } = dom, doc = win.document;
win.HTMLElement.prototype.scrollTo ??= function () {};

// EventSource over fetch streaming (Node 22 has none enabled by default)
class FakeES {
  constructor(url) { this.url = base + url; this.l = {}; this.ac = new AbortController(); this.open(); }
  addEventListener(t, f) { (this.l[t] ||= []).push(f); }
  emit(t, ev) { (t === 'message' ? [this.onmessage] : []).concat(this.l[t] || []).forEach(f => f && f(ev)); }
  async open() {
    try {
      const r = await fetch(this.url, { signal: this.ac.signal }); this.onopen?.();
      const rd = r.body.getReader(), dec = new TextDecoder(); let buf = '';
      for (;;) { const { done, value } = await rd.read(); if (done) break; buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf('\n\n')) >= 0) { const blk = buf.slice(0, i); buf = buf.slice(i + 2);
          let type = 'message', data = ''; for (const ln of blk.split('\n')) { if (ln.startsWith('event: ')) type = ln.slice(7); else if (ln.startsWith('data: ')) data += ln.slice(6); }
          if (data) this.emit(type, { data }); } }
    } catch (e) { if (!this.ac.signal.aborted) this.onerror?.(e); }
  }
  close() { this.ac.abort(); }
}
const mockVp = { calls: [], pickResult: null, groundResult: [1.5, 2.5], frozen: 'unset', env: null,
  push(s) { this.calls.push('push'); this.lastPush = s; }, freeze(s) { this.frozen = s; this.calls.push('freeze'); }, select(s) { this.sel = s; },
  pick() { return this.pickResult; }, ground() { return this.groundResult; }, project(w) { return { x: 100 + w[0] * 10, y: 100 - w[1] * 10, visible: true }; },
  headOf(id) { return [0, 1.7, 0]; }, async loadEnvironment(b) { this.env = b.byteLength; }, dispose() {},
  models: [], cleared: [], failNext: false,
  async setCharacterModel(id, buf) { if (this.failNext) { this.failNext = false; throw new Error('bad mesh'); } this.models.push([id, buf.byteLength]); return { mode: 'rigged', note: '' }; },
  clearCharacterModel(id) { this.cleared.push(id); } };

const { boot } = await import(STUDIO + '/web/app.js');
const ctl = await boot({ document: doc, window: win, fetch: (p, o) => fetch(p.startsWith('http') ? p : base + p, o), EventSource: FakeES, createViewport: () => mockVp });

let n = 0; const ok = m => { n++; console.log('  ok  ', m); };
const $ = id => doc.getElementById(id);
const txt = sel => [...doc.querySelectorAll(sel)].map(e => e.textContent);
const click = e => e.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
const key = (e, k) => e.dispatchEvent(new win.KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
const ptr = (type, init) => $('viewport').dispatchEvent(new win.MouseEvent(type, { bubbles: true, ...init }));
const post = (p, b) => fetch(base + p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }).then(r => r.json());
const state = () => fetch(base + '/api/state').then(r => r.json());
const logHas = s => ctl.state.log.some(e => e.text.includes(s));

try {
  // 1 ---------------------------------------------------------------- initial render + live stream
  await until(() => $('conn-text').textContent === 'live', 8000, 'connected');
  // 0 ---------------------------------------------------------------- empty by default; the Guide holds the demo
  await until(() => !$('empty-state').hidden, 4000, 'empty-state shown');
  assert.equal(txt('#chars-list .nm').length, 0); assert.equal(txt('#props-list .nm').length, 0); assert.match($('empty-state').textContent, /empty stage/i);
  click($('es-guide')); assert.equal($('modal').hidden, false); assert.match($('modal-card').textContent, /Connect a brain/); assert.match($('modal-card').textContent, /GRAB <hand> <prop>/);
  click($('modal-cancel')); assert.equal($('modal').hidden, true, 'Guide closes');
  click($('btn-guide')); click($('guide-demo')); assert.equal($('modal').hidden, true, 'a Guide action closes it');
  await until(() => $('empty-state').hidden && txt('#chars-list .nm').length === 2, 8000, 'demo loaded from the Guide');
  ok('starts as an empty stage; the Guide explains things and loads the demo scene on request');
  assert.deepEqual(txt('#chars-list .nm').sort(), ['Alice', 'Bob']);
  assert.ok(txt('#props-list .nm').includes('cup') && txt('#props-list .nm').includes('table'));
  assert.equal($('mode').textContent, 'mock agent');
  const t0 = $('simtime').textContent; await sleep(700); assert.notEqual($('simtime').textContent, t0, 'sim clock must advance');
  assert.ok(mockVp.calls.filter(c => c === 'push').length > 5, 'snapshots reach the viewport');
  ok('boots, connects, lists characters and props, clock and viewport are live');

  // 2 ---------------------------------------------------------------- selection + inspector
  click(doc.querySelector('[data-char="alice"]'));
  assert.equal(doc.querySelector('#inspector .name').textContent, 'Alice');
  assert.match($('insp-persona').value, /baker/); assert.deepEqual(mockVp.sel, { type: 'char', id: 'alice' });
  assert.ok(doc.querySelector('[data-char="alice"]').classList.contains('sel'));
  ok('selecting a character fills the inspector and tells the viewport');

  // 3 ---------------------------------------------------------------- typed command, completion, history, errors
  const cmd = $('insp-cmd'); cmd.value = 'GRAB right c'; key(cmd, 'Tab'); assert.equal(cmd.value, 'GRAB right c', 'ambiguous prefix (cup, crate) must not auto-complete'); assert.match($('cmd-hint').textContent, /cup.*crate|crate.*cup/);
  cmd.value = 'GRAB right cu'; key(cmd, 'Tab'); assert.equal(cmd.value, 'GRAB right cup ');
  cmd.value = 'GRAB ri'; key(cmd, 'Tab'); assert.equal(cmd.value, 'GRAB right ');
  cmd.value = 'WALK_TO '; key(cmd, 'Tab'); assert.match($('cmd-hint').textContent, /cup/); ok('Tab completion: verb, hand, name, and candidate hints');
  cmd.value = 'WAVE right'; key(cmd, 'Enter');
  await until(() => logHas('Alice: WAVE right'), 5000, 'command logged');
  await until(() => txt('[data-char="alice"] .st').join('').includes('WAVE'), 5000, 'busy label');
  assert.equal(cmd.value, ''); await until(() => $('insp-status').textContent.startsWith('WAVE'), 3000, 'inspector status');
  cmd.value = ''; key(cmd, 'ArrowUp'); assert.equal(cmd.value, 'WAVE right', 'history recall');
  cmd.value = 'DANCE'; key(cmd, 'Enter'); await until(() => !$('toast').hidden && /unknown command/.test($('toast').textContent), 4000, 'error toast');
  assert.ok($('toast').classList.contains('error')); assert.equal(cmd.value, 'DANCE', 'a rejected command stays editable');
  ok('command → server → transcript → busy label; errors surface as toasts');
  click($('btn-stop')); await until(() => !txt('[data-char="alice"] .st').join(''), 5000, 'stop');

  // 4 ---------------------------------------------------------------- persona + autonomy persist server-side
  const ta = $('insp-persona'); ta.value = 'A retired pirate.'; ta.dispatchEvent(new win.Event('change', { bubbles: true }));
  const ag = $('insp-agent'); ag.checked = true; ag.dispatchEvent(new win.Event('change', { bubbles: true }));
  await until(async () => { const c = (await state()).snapshot.chars.find(c => c.id === 'alice'); return c.persona === 'A retired pirate.' && c.agent; }, 4000, 'persona saved');
  await until(() => txt('[data-char="alice"] .tag').includes('auto'), 3000, 'auto tag'); ag.checked = false; ag.dispatchEvent(new win.Event('change', { bubbles: true }));
  ok('persona and autonomy edits persist on the server and reflect in the list');

  // 5 ---------------------------------------------------------------- XSS: hostile names/personas are inert text
  click($('btn-add-char'));
  const [nameIn, personaIn] = doc.querySelectorAll('#modal-card input, #modal-card textarea');
  nameIn.value = '<img src=x onerror="window.__pwned=1">'; personaIn.value = '<script>window.__pwned=2</script>';
  click($('modal-ok')); await until(() => $('modal').hidden && txt('#chars-list .nm').some(t => t.includes('<img')), 5000, 'hostile char added');
  assert.equal(doc.querySelectorAll('img, #chars-list script, #inspector script').length, 0, 'no element injected'); assert.equal(win.__pwned, undefined);
  assert.match(doc.querySelector('#inspector .name').textContent, /<img src=x/); ok('XSS attempt in name/persona renders as literal text');
  await post('/api/command', { character: 'img_src_x_onerror_window_pwned_1', line: 'SAY <b>bold</b> & <script>alert(1)</script>' }).catch(() => {});
  const evil = (await state()).snapshot.chars.find(c => c.name.includes('<img')); await post('/api/command', { character: evil.id, line: 'SAY <b>hi</b><img src=x onerror=alert(1)>' });
  await until(() => txt('#log .say').some(t => t.includes('<b>hi</b>')), 5000, 'hostile speech');
  await until(() => txt('#labels .bubble').some(t => t.includes('<b>hi</b>')), 5000, 'bubble');
  assert.equal(doc.querySelectorAll('#log b, #labels b, #labels img, #log img').length, 0); ok('hostile speech renders as text in transcript and speech bubble');
  click($('btn-del-char')); await until(() => !txt('#chars-list .nm').some(t => t.includes('<img')), 5000, 'removed');

  // 6 ---------------------------------------------------------------- modal validation
  click($('btn-add-prop')); click($('modal-ok')); await sleep(150);
  assert.equal($('modal').hidden, false, 'invalid form keeps the modal open'); assert.match($('toast').textContent, /required/);
  const [pn, ps, pt] = doc.querySelectorAll('#modal-card input'); pn.value = 'Lantern'; ps.value = '0.2 0.4 0.2'; pt.value = 'light, fragile'; click($('modal-ok'));
  await until(() => txt('#props-list .nm').includes('lantern'), 5000, 'prop added'); assert.ok(txt('[data-prop="lantern"] .tag').includes('fragile'));
  assert.equal(doc.querySelector('#inspector .name').textContent, 'lantern');
  const px = doc.querySelector('[data-axis="x"]'); px.value = '1.25'; px.dispatchEvent(new win.Event('change', { bubbles: true }));
  await until(async () => (await state()).snapshot.props.find(p => p.name === 'lantern').pos[0] === 1.25, 4000, 'prop moved');
  click($('btn-del-prop')); await until(() => !txt('#props-list .nm').includes('lantern'), 5000, 'prop deleted');
  ok('add/edit/delete prop through the UI, with validation');

  // 7 ---------------------------------------------------------------- viewport gestures
  click(doc.querySelector('[data-char="bob"]'));
  mockVp.pickResult = { type: 'prop', id: 'ball' }; ptr('pointerdown', { clientX: 10, clientY: 10 }); ptr('pointerup', { clientX: 10, clientY: 10, altKey: true });
  await until(() => logHas('Bob: GRAB right ball'), 5000, 'alt-click grab');
  ptr('pointerdown', { clientX: 10, clientY: 10 }); ptr('pointerup', { clientX: 90, clientY: 10 });          // a drag (orbit) must not act
  await sleep(200); assert.equal(ctl.state.selected.id, 'bob', 'orbit-drag is not a click');
  mockVp.pickResult = null; ptr('pointerdown', { clientX: 5, clientY: 5 }); ptr('pointerup', { clientX: 5, clientY: 5, shiftKey: true });
  await until(() => logHas('Bob: WALK_TO 1.50 2.50'), 5000, 'shift-click walk');
  mockVp.pickResult = { type: 'prop', id: 'table' }; ptr('pointerdown', { clientX: 5, clientY: 5 }); ptr('pointerup', { clientX: 5, clientY: 5 });
  assert.deepEqual(ctl.state.selected, { type: 'prop', id: 'table' }); mockVp.pickResult = null; ptr('pointerdown', { clientX: 5, clientY: 5 }); ptr('pointerup', { clientX: 5, clientY: 5 });
  assert.equal(ctl.state.selected, null); assert.match($('inspector').textContent, /Select a character/);
  ok('viewport: click selects, alt+click grabs, shift+click walks, drags and empty clicks behave');
  click(doc.querySelector('[data-char="bob"]')); click($('btn-stop'));

  // 8 ---------------------------------------------------------------- director → mock agent → world
  $('director-target').value = 'alice'; $('director-text').value = 'pick up the cup'; click($('btn-direct'));
  await until(() => logHas('Director'), 4000, 'director logged'); assert.equal($('director-text').value, '');
  await until(() => txt('#props-list [data-prop="cup"] .tag').includes('held'), 60000, 'cup picked up by the agent');
  ok('director prompt → agent → commands → cup held (props list shows it)');

  // 9 ---------------------------------------------------------------- record, play, scrub, live
  click($('btn-rec')); await until(() => !$('rec-indicator').hidden && $('btn-rec').classList.contains('on'), 4000, 'recording on');
  await post('/api/command', { character: 'bob', line: 'WAVE left' }); await sleep(1600); click($('btn-rec'));
  await until(() => $('rec-indicator').hidden && /Take saved/.test($('toast').textContent), 6000, 'recording stopped'); await until(() => $('takes').options.length >= 1 && $('takes').value !== '', 4000, 'take listed');
  assert.equal($('btn-play').disabled, false); click($('btn-play'));
  await until(() => st_play() && mockVp.frozen && mockVp.frozen !== 'unset', 6000, 'playback started');
  function st_play() { return ctl.state.play; }
  assert.equal($('scrub').disabled, false); const total = parseInt($('scrub').max, 10); assert.ok(total > 30, `take has ${total} frames`);
  $('scrub').value = '10'; $('scrub').dispatchEvent(new win.Event('input', { bubbles: true }));
  assert.equal(ctl.state.play.idx, 10); assert.equal(mockVp.frozen.t, ctl.state.play.take.frames[10].t); assert.equal(ctl.state.playing, false);
  assert.match($('scrub-time').textContent, /\d\d:\d\d\.\d \/ \d\d:\d\d\.\d/);
  click($('btn-play')); await sleep(500); assert.ok(ctl.state.play.idx > 10, 'playback advances'); click($('btn-play'));
  click($('btn-live')); assert.equal(mockVp.frozen, null); assert.equal(ctl.state.play, null); assert.equal($('btn-live').disabled, true);
  ok('record → take listed → play/scrub/pause → back to live');

  // 10 --------------------------------------------------------------- environment import
  const j = JSON.stringify({ asset: { version: '2.0' }, scenes: [{ nodes: [0, 1, 2] }], meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }], accessors: [{ min: [-1, -.5, -.5], max: [1, .5, .5] }],
    nodes: [{ mesh: 0, name: 'Sofa', translation: [-2, .4, 2] }, { mesh: 0, name: 'Vase', translation: [0, 1, 0], scale: [.1, .2, .1] }, { mesh: 0, name: 'Floor', scale: [20, .1, 20] }] });
  const pad = Buffer.from(j + ' '.repeat(-Buffer.byteLength(j) % 4 + 4 * (Buffer.byteLength(j) % 4 === 0 ? 0 : 0)), 'utf8'); const jb = Buffer.concat([pad, Buffer.alloc((4 - pad.length % 4) % 4, 0x20)]);
  const head = Buffer.alloc(20); head.write('glTF', 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(20 + jb.length, 8); head.writeUInt32LE(jb.length, 12); head.writeUInt32LE(0x4E4F534A, 16);
  await ctl.importFile(new File([Buffer.concat([head, jb])], 'room.glb'));
  await until(() => txt('#props-list .nm').includes('sofa') && txt('#props-list .nm').includes('vase'), 5000, 'imported props listed');
  assert.ok(mockVp.env > 0, 'GLB bytes handed to the viewport for display'); assert.match($('toast').textContent, /2 props, 1 structural skipped/);
  await ctl.importFile(new File([Buffer.from('definitely not a model')], 'junk.glb')); await until(() => /import failed/.test($('toast').textContent), 4000, 'bad import message');
  assert.ok(txt('#props-list .nm').includes('sofa'), 'a failed import leaves the scene untouched'); ok('environment import: props appear, structure skipped, junk rejected cleanly');

  // 11 --------------------------------------------------------------- external endpoint (second choice after NEON)
  click($('btn-settings')); $('ext-details').open = true;
  $('ext-url').value = 'ftp://nope'; click($('ext-save')); await until(() => /http/.test($('toast').textContent), 4000, 'bad url toast');
  $('ext-url').value = 'http://127.0.0.1:9'; $('ext-model').value = 'my-model'; click($('ext-save'));
  await until(() => /^model · my-model/.test($('mode').textContent) && $('mode').classList.contains('model'), 5000, 'pill follows the saved endpoint');
  $('ext-url').value = ''; click($('ext-save')); await until(() => $('mode').textContent === 'mock agent', 5000, 'back to mock'); click($('modal-cancel'));
  ok('external endpoint: validated, shown in the pill, clearable');

  // 11b -------------------------------------------------------------- NEON: find it, pick a model, run it, use it
  if (HAVE_NEON) {
    click($('btn-settings')); await until(() => $('neon-bin-ok'), 5000, 'neon binary detected');
    assert.match($('neon-bin-ok').textContent, /neon_test|neon/);
    await until(() => doc.querySelector('[data-model="good.gguf"]') && doc.querySelector('[data-model="llama.gguf"]'), 5000, 'models listed');
    const good = doc.querySelector('[data-model="good.gguf"] input'), bad = doc.querySelector('[data-model="llama.gguf"] input');
    assert.equal(good.disabled, false); assert.equal(bad.disabled, true, 'a wrong-architecture model cannot be selected');
    assert.match(doc.querySelector('[data-model="good.gguf"]').textContent, /nanity ✓/); assert.match(doc.querySelector('[data-model="llama.gguf"]').textContent, /llama.*nanity_convert/s);
    assert.ok(good.checked, 'the first runnable model is preselected');
    click($('neon-check')); await until(() => /passes NANITY spec validation/.test($('neon-result').textContent), 20000, 'probe result'); assert.ok($('neon-result').classList.contains('ok-text'));
    $('neon-threads').value = '1'; click($('neon-launch'));
    await until(() => /starting|ready/.test($('neon-state').textContent), 5000, 'launching');
    await until(() => /ready/.test($('neon-state').textContent) && /context 4096/.test($('neon-state').textContent), 30000, 'NEON ready');
    await until(() => /^neon · good\.gguf/.test($('mode').textContent), 5000, 'top-bar pill shows the active brain'); assert.ok($('mode').classList.contains('model'));
    assert.ok(/Interactive stdin\/stdout loop/.test($('neon-log').textContent), 'NEON\'s own log is shown');
    click($('modal-cancel')); assert.equal($('modal').hidden, true); assert.equal(st_modalHook(), null, 'closing the panel stops its polling');
    function st_modalHook() { return ctl.state.onModalClose; }
    await post('/api/direct', { character: 'bob', text: 'say hello' });
    await until(async () => ((await (await fetch(base + '/api/neon')).json()).process || {}).requests >= 1, 40000, 'a request reached the real NEON');
    click($('btn-settings')); await until(() => /1 request/.test($('neon-state').textContent), 5000, 'request counter in the panel');
    $('ext-details').open = true; $('ext-url').value = 'http://127.0.0.1:9'; click($('ext-save')); await sleep(500);
    assert.match($('mode').textContent, /^neon · good/, 'a running NEON wins over a saved endpoint');
    $('ext-url').value = ''; click($('ext-save')); await sleep(300);
    click($('neon-stop')); await until(() => /not running/.test($('neon-state').textContent), 8000, 'stopped'); await until(() => $('mode').textContent === 'mock agent', 5000, 'falls back to the mock agent');
    $('ext-details').open = true; $('ext-url').value = 'ftp://nope'; click($('ext-save')); await until(() => /http/.test($('toast').textContent), 4000, 'bad endpoint refused');
    click($('modal-cancel'));
    ok('Model panel: finds the real NEON, lists models (wrong-arch disabled), checks, launches, shows ready + context, drives the pill, stops');
  }

  // 11c -------------------------------------------------------------- character body (.glb) upload
  click(doc.querySelector('[data-char="alice"]'));
  assert.equal($('insp-body').textContent, 'blacked-out entity (default)'); assert.equal($('btn-model-clear').disabled, true);
  await ctl.uploadModel('alice', new File([Buffer.concat([head, jb])], 'hero.glb'));
  await until(() => mockVp.models.some(m => m[0] === 'alice' && m[1] > 100), 6000, 'the model reached the viewport');
  await until(() => /hero\.glb · rigged/.test($('insp-body').textContent), 4000, 'body label'); assert.equal($('btn-model-clear').disabled, false);
  click($('btn-model-clear')); await until(() => mockVp.cleared.includes('alice') && $('insp-body').textContent === 'blacked-out entity (default)', 5000, 'cleared');
  mockVp.failNext = true; await ctl.uploadModel('alice', new File([Buffer.concat([head, jb])], 'broken.glb'));
  await until(() => /couldn't use that model \(bad mesh\)/.test($('toast').textContent), 6000, 'failure toast'); assert.ok($('toast').classList.contains('error'));
  await ctl.uploadModel('alice', new File([Buffer.from('this is not a model')], 'junk.glb')); await until(() => /readable|glTF/.test($('toast').textContent), 4000, 'server refused junk');
  ok('character body: upload → viewport → label; clear returns to the blackout entity; viewport failures and junk files are reported');

  // 12 --------------------------------------------------------------- clear scene, then the Guide brings the demo back
  click($('btn-clear')); click($('modal-ok')); await until(() => $('modal').hidden, 4000);
  await until(() => txt('#chars-list .nm').length === 0 && txt('#props-list .nm').length === 0 && !$('empty-state').hidden, 8000, 'cleared');
  click($('btn-guide')); click($('guide-demo'));
  await until(() => txt('#chars-list .nm').sort().join() === 'Alice,Bob' && !txt('#props-list .nm').includes('sofa'), 8000, 'demo back');
  ok('Clear scene empties the stage; the Guide restores the demo');

  // 13 --------------------------------------------------------------- labels follow the scene, nothing leaks
  await sleep(300); assert.ok(doc.querySelectorAll('#labels .name').length === 2 && doc.querySelectorAll('#labels .prop').length === 5, 'name + prop labels match the scene');
  assert.deepEqual(errors, [], 'no jsdom/page errors: ' + errors.join('; ')); ok('labels match the scene; no page errors');
  console.log(`\n${n} UI scenarios passed`);
} catch (e) {
  console.error('\nFAILED:', e.stack || e); if (serverErr) console.error('server stderr:\n' + serverErr.slice(-1500)); process.exitCode = 1;
} finally { ctl.destroy(); server.kill(); setTimeout(() => process.exit(process.exitCode || 0), 100); }
