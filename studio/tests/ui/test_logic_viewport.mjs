// Node tests: pure logic + the three.js Rig (no GPU needed for scene-graph math).
import assert from 'node:assert/strict';
import * as THREE from 'three';
import * as L from '../../web/logic.js';
import { Rig } from '../../web/viewport.js';

let n = 0; const ok = (name) => { n++; console.log('  ok  ', name); };
const near = (a, b, e = 1e-4, m = '') => assert.ok(Math.abs(a - b) < e, `${m} ${a} vs ${b}`);
const vnear = (a, b, e = 1e-3, m = '') => a.forEach((v, i) => near(v, b[i], e, m));

// ---- lerp / angles
near(L.lerpAngle(3.0, -3.0, 0.5), 3.0 + ((-3.0 - 3.0 + 3 * Math.PI) % (2 * Math.PI) - Math.PI) * 0.5);
near(Math.cos(L.lerpAngle(Math.PI - 0.1, -Math.PI + 0.1, 0.5)), -1, 1e-6, 'wraps the short way round');
ok('angle lerp takes the short way across ±π');

const mk = (x, yaw = 0, walk = 0) => ({ t: 0, recording: false, chars: [{ id: 'a', name: 'A', color: '#c70000', pos: [x, 0, 0], yaw, crouch: 0, walk, speech: '', busy: null, queued: 0, agent: false, persona: '',
  arms: { left: [[0.19, 1.42, 0], [0.2, 1.15, -0.1], [0.3, 1.0, 0.1]], right: [[-0.19, 1.42, 0], [-0.2, 1.15, -0.1], [-0.3, 1.0, 0.1]] }, gaze: [0, 0, 1], holding: { left: null, right: null } }],
  props: [{ name: 'cup', pos: [x, 1, 1], size: [.1, .1, .1], tags: [], kind: 'box', color: '#a60000', held: false }] });
const mid = L.interpolateSnapshot(mk(0), mk(2), 0.5);
near(mid.chars[0].pos[0], 1); near(mid.props[0].pos[0], 1); near(mid.chars[0].arms.left[2][0], 0.3);
assert.equal(L.interpolateSnapshot(null, mk(2), 0.5).chars[0].pos[0], 2);
assert.equal(L.interpolateSnapshot(mk(0), mk(2), 5).chars[0].pos[0], 2);
const newcomer = mk(2); newcomer.chars.push({ ...newcomer.chars[0], id: 'b' });
assert.equal(L.interpolateSnapshot(mk(0), newcomer, 0.5).chars.length, 2);
ok('snapshot interpolation: midpoint, no-prev, t>1, new characters');

// ---- leg IK: lengths preserved, knee bends the right way, unreachable clamps
for (const foot of [[0.1, 0.05, 0.2], [0.0, 0.5, 0.0], [0, -10, 0], [0.3, 0.1, 0.5]]) {
  const hip = [0, 0.95, 0], knee = L.legIK(hip, foot, [0, 0, 1]);
  const d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  near(d(hip, knee), 0.45, 1e-4, 'thigh length'); assert.ok(Number.isFinite(knee[0] + knee[1] + knee[2]));
  if (d(hip, foot) < 0.89) near(d(knee, foot), 0.45, 1e-3, 'shin length');
  assert.ok(knee[2] >= -1e-6 || foot[2] < 0, 'knee bends toward facing direction');
}
ok('leg IK: bone lengths preserved, knee forward, clamps out-of-reach');

// ---- feet: stationary feet planted & symmetric, walking feet alternate
const stand = L.footTargets({ pos: [0, 0, 0], yaw: 0, walk: 7.3, crouch: 0 }, 0);
vnear(stand.left, [0.11, 0.05, 0]); vnear(stand.right, [-0.11, 0.05, 0]);
const w1 = L.footTargets({ pos: [0, 0, 0], yaw: 0, walk: 0.5, crouch: 0 }, 1), w2 = L.footTargets({ pos: [0, 0, 0], yaw: 0, walk: 1.5, crouch: 0 }, 1);
assert.ok(w1.left[2] * w1.right[2] <= 0 && w2.left[2] * w2.right[2] <= 0, 'feet on opposite sides of the body while walking');
assert.ok(w1.left[2] * w2.left[2] <= 0, 'a foot swings from front to back across one step');
ok('footTargets: standing neutral, walking alternates');

// ---- structure key stable under pose changes, sensitive to real changes
assert.equal(L.structureKey(mk(0)), L.structureKey(mk(5, 1, 3)));
const busy = mk(0); busy.chars[0].busy = 'WALK_TO x';
assert.notEqual(L.structureKey(mk(0)), L.structureKey(busy));
ok('structureKey: ignores motion, notices commands');

// ---- completions
const snap = mk(0);
assert.deepEqual(L.completions('GR', snap), ['GRAB']); assert.deepEqual(L.completions('grab ri', snap), ['right']);
assert.deepEqual(L.completions('GRAB right c', snap), ['cup']); assert.deepEqual(L.completions('WALK_TO ', snap).sort(), ['a', 'cup']);
assert.deepEqual(L.completions('SAY he', snap), []);
assert.equal(L.applyCompletion('GRAB right c', 'cup'), 'GRAB right cup ');
assert.equal(L.fmtTime(75.25), '01:15.3'.replace('.3', '.3'));
ok('completions and fmtTime');

// ---- Rig: limb endpoints must coincide with engine joint positions
const scene = new THREE.Scene(), rig = new Rig(scene);
const s = mk(1.5, 0.7, 2.2); s.chars[0].crouch = 0.3;
rig.sync(s, 1 / 60, null);
const cr = rig.chars.get('a');
const endpoints = (m) => { m.updateMatrixWorld(true); const l = m.scale.y, d = new THREE.Vector3(0, 1, 0).applyQuaternion(m.quaternion).multiplyScalar(l / 2);
  return [m.position.clone().sub(d).toArray(), m.position.clone().add(d).toArray()]; };
for (const side of ['left', 'right']) {
  const [a, e, h] = s.chars[0].arms[side];
  const [u0, u1] = endpoints(cr.arms[side][0]), [f0, f1] = endpoints(cr.arms[side][1]);
  vnear(u0, a, 1e-4, 'upper arm start'); vnear(u1, e, 1e-4, 'upper arm end'); vnear(f0, e, 1e-4, 'forearm start'); vnear(f1, h, 1e-4, 'forearm end');
  vnear(cr.arms[side][2].position.toArray(), h, 1e-4, 'hand');
}
const knee = (side) => endpoints(cr.legs[side][0])[1], shank = (side) => endpoints(cr.legs[side][1]);
for (const side of ['left', 'right']) { vnear(shank(side)[0], knee(side), 1e-4, 'shin starts at knee'); near(Math.hypot(...endpoints(cr.legs[side][0]).reduce((p, q) => p.map((v, i) => q[i] - v))), 0.45, 1e-3, 'thigh'); }
ok('Rig: arm/leg cylinders start and end exactly on the engine joint positions');

rig.sync(s, 1 / 60, { type: 'char', id: 'a' }); assert.equal(cr.ring.visible, true);
rig.sync(s, 1 / 60, null); assert.equal(cr.ring.visible, false);
const hits = rig.pickables(); assert.equal(hits.length, 2); assert.deepEqual(hits.map(h => h.userData.type).sort(), ['char', 'prop']);
const prop = rig.props.get('cup'); vnear(prop.group.position.toArray(), [1.5, 1, 1]); vnear(prop.group.scale.toArray(), [.1, .1, .1]);
ok('Rig: selection ring, pick targets, prop transform');

const empty = { ...s, chars: [], props: [] }; rig.sync(empty, 1 / 60, null);
assert.equal(rig.chars.size, 0); assert.equal(rig.props.size, 0); assert.equal(scene.children.length, 0);
ok('Rig: removed characters/props leave nothing behind in the scene');

for (let i = 0; i < 40; i++) { const t = mk(i * 0.04, 0, i * 0.08); rig.sync(t, 1 / 60, null); }
assert.ok(rig.chars.get('a').moving > 0.5, 'walking character reaches full stride blend');
for (let i = 0; i < 60; i++) rig.sync(mk(1.6, 0, 3.2), 1 / 60, null);
assert.ok(rig.chars.get('a').moving < 0.05, 'stationary character settles to standing');
ok('Rig: walk blend ramps up while moving and back down at rest');

console.log(`\n${n} groups passed`);
