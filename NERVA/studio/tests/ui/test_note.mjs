// The 3D-viewport failure note must say WHY, whether the module failed to load or WebGL refused to start.
import { JSDOM } from 'jsdom';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const STUDIO = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../..');
const { boot } = await import(STUDIO + '/web/app.js');
const snap = { t: 0, recording: false, meta: { llm: 'mock agent', mode: 'mock' }, chars: [], props: [] };
async function run(env) {
  const dom = new JSDOM(readFileSync(path.join(STUDIO, 'web/index.html'), 'utf8'), { pretendToBeVisual: true, url: 'http://127.0.0.1/' });
  const fetchStub = async p => ({ ok: true, json: async () => p.startsWith('/api/state') ? { snapshot: snap, log: [], config: { mode: 'mock' }, takes: [] } : [] });
  class ES { constructor() { setTimeout(() => this.onopen?.(), 0); } addEventListener() {} close() {} }
  const ctl = await boot({ document: dom.window.document, window: dom.window, fetch: fetchStub, EventSource: ES, ...env });
  const note = dom.window.document.getElementById('viewport-note'); ctl.destroy(); return note;
}
const origErr = console.error; console.error = () => {};
let n = await run({ createViewport: () => { throw new Error('could not create a WebGL2 context (Error creating WebGL context.)'); } });
assert.equal(n.hidden, false); assert.match(n.textContent, /could not create a WebGL2 context \(Error creating WebGL context\.\)/); assert.match(n.textContent, /F12/);
n = await run({ createViewport: null, viewportError: "the 3D module failed to load: Failed to resolve module specifier 'three'" });
assert.match(n.textContent, /failed to load: Failed to resolve module specifier 'three'/);
n = await run({ createViewport: null }); assert.match(n.textContent, /3D module is missing/);
n = await run({ createViewport: () => ({ push() {}, freeze() {}, select() {}, project: () => ({ x: 0, y: 0, visible: false }), headOf: () => null, pick() {}, ground() {}, dispose() {} }) }); assert.equal(n.hidden, true, 'no note when the viewport works');
console.error = origErr; console.log('  ok   viewport failure note names the real reason (WebGL2 refusal, module load failure, missing module) and is hidden when all is well');
