// Avatar tests: bone-name classification, binding errors, and a REAL Blender-exported skinned GLB driven through
// three's GLTFLoader (rigged) and a real unskinned GLB (static).
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync, existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import * as THREE from 'three';
import { classifyBone, tokenize, bindHumanoid, aimBone, parseAvatar } from '../../web/avatar.js';
import { frame } from '../../web/logic.js';
import { Rig } from '../../web/viewport.js';

const HERE = path.dirname(new URL(import.meta.url).pathname);
let n = 0; const ok = m => { n++; console.log('  ok  ', m); };
const near = (a, b, e, m = '') => assert.ok(Math.abs(a - b) < e, `${m}: ${a} vs ${b} (tol ${e})`);

// ---------------------------------------------------------------- classification across rig conventions
const T = {
  // Mixamo (three's GLTFLoader strips ':' so both spellings must work)
  'mixamorig:LeftArm': ['left', 'upper'], 'mixamorigLeftArm': ['left', 'upper'], 'mixamorig:LeftForeArm': ['left', 'fore'], 'mixamorigRightHand': ['right', 'hand'],
  'mixamorig:LeftUpLeg': ['left', 'thigh'], 'mixamorig:RightLeg': ['right', 'shin'], 'mixamorig:RightFoot': ['right', 'foot'], 'mixamorig:Hips': [null, 'hips'], 'mixamorig:Head': [null, 'head'],
  // VRM / VRoid
  leftUpperArm: ['left', 'upper'], leftLowerArm: ['left', 'fore'], rightHand: ['right', 'hand'], leftUpperLeg: ['left', 'thigh'], rightLowerLeg: ['right', 'shin'], leftFoot: ['left', 'foot'],
  J_Bip_L_UpperArm: ['left', 'upper'], J_Bip_R_LowerArm: ['right', 'fore'], J_Bip_L_Hand: ['left', 'hand'], J_Bip_C_Hips: [null, 'hips'],
  // Rigify / Blender
  'upper_arm.L': ['left', 'upper'], 'forearm.R': ['right', 'fore'], 'hand.L': ['left', 'hand'], 'thigh.R': ['right', 'thigh'], 'shin.L': ['left', 'shin'], 'foot.R': ['right', 'foot'],
  // generic
  UpperArm_L: ['left', 'upper'], Forearm_R: ['right', 'fore'], Pelvis: [null, 'hips'], Calf_L: ['left', 'shin'], Thigh_R: ['right', 'thigh'],
  // things that must NOT be driven
  'mixamorig:LeftHandIndex1': [ 'left', null], 'mixamorig:LeftToeBase': ['left', null], 'mixamorig:LeftShoulder': ['left', null], 'mixamorig:Spine1': [null, null], HeadTop_End: [null, null],
  'forearm_twist.L': ['left', null], 'LeftArmRoll': ['left', null], thumb_02_R: ['right', null], Armature: [null, null], '': [null, null],
};
for (const [name, [side, part]] of Object.entries(T)) { const r = classifyBone(name); assert.deepEqual([r.side, r.part], [side, part], `${JSON.stringify(name)} -> ${JSON.stringify(r)}`); }
assert.deepEqual(tokenize('mixamorig:LeftForeArm'), ['mixamorig', 'left', 'fore', 'arm']);
ok(`bone-name classification: ${Object.keys(T).length} names across Mixamo, VRM/VRoid, Rigify and generic rigs (fingers/twist/toes/clavicles excluded)`);

// ---------------------------------------------------------------- binding errors (synthetic skeletons)
const bone = (name, parent, pos = [0, 0, 0]) => { const b = new THREE.Bone(); b.name = name; b.position.set(...pos); parent?.add(b); return b; };
{
  const root = new THREE.Object3D(), hips = bone('Hips', root);
  const sp = bone('Spine', hips, [0, 0.2, 0]), lu = bone('LeftArm', sp, [0.2, 0, 0]); bone('LeftForeArm', lu, [0.3, 0, 0]); // left hand missing
  assert.match(bindHumanoid(root).error, /left arm chain/);
  const r2 = new THREE.Object3D(), h2 = bone('Hips', r2), s2 = bone('Spine', h2);
  const a = bone('LeftArm', s2), f = bone('LeftForeArm', h2), hd = bone('LeftHand', f);          // forearm NOT under the upper arm
  assert.match(bindHumanoid(r2).error, /not parented/);
  assert.equal(bindHumanoid(new THREE.Object3D()).error !== undefined, true);
}
ok('binding reports a specific, human-readable reason when the skeleton is not a usable humanoid');

// ---------------------------------------------------------------- aimBone: exact, idempotent, preserves length
{
  const root = new THREE.Object3D(), a = bone('A', root, [0, 0, 0]), b = bone('B', a, [0, 1, 0]); root.updateMatrixWorld(true);
  const target = new THREE.Vector3(0.6, 0.3, -0.7); aimBone(a, b, target);
  const dir = new THREE.Vector3().setFromMatrixPosition(b.matrixWorld).normalize(), want = target.clone().normalize();
  near(dir.dot(want), 1, 1e-6, 'bone points at target'); near(new THREE.Vector3().setFromMatrixPosition(b.matrixWorld).length(), 1, 1e-6, 'length kept');
  aimBone(a, b, target); near(new THREE.Vector3().setFromMatrixPosition(b.matrixWorld).normalize().dot(want), 1, 1e-6, 'idempotent');
}
ok('aimBone points a bone exactly at a target, keeps its length, and is idempotent');

// ---------------------------------------------------------------- real Blender GLBs
const blender = spawnSync('blender', ['--version']).status === 0;
if (!blender) { console.log('  skip real-GLB tests: blender not installed'); console.log(`\n${n} groups passed`); process.exit(0); }
const dir = mkdtempSync(path.join(tmpdir(), 'avatar-'));
const make = (script, out) => { const r = spawnSync('blender', ['-b', '--factory-startup', '--python', path.join(HERE, '..', script), '--', out], { encoding: 'utf8' }); assert.ok(r.stdout.includes('EXPORTED'), r.stdout.slice(-400) + r.stderr.slice(-400)); };
make('make_rigged_glb.py', path.join(dir, 'robot.glb')); make('make_glb.py', path.join(dir, 'kitchen'));
const buf = f => { const b = readFileSync(f); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

const { avatar: av, mode, note } = await parseAvatar(buf(path.join(dir, 'robot.glb')));
assert.equal(mode, 'rigged', note); assert.equal(note, '');
const B = av.binding;
assert.ok(B.arms.left.upper && B.arms.right.hand && B.legs.left.foot && B.legs.right.thigh && B.head && B.hips);
assert.ok(!Object.values(B.arms.left).some(b => /index|shoulder/i.test(b.name)), 'fingers and clavicle are not driven');
near(av.s, 1.75 / 1.68, 0.02, 'scale to 1.75 m'); near(av.len.left.arm[0], 0.30 * av.s, 2e-3, 'measured upper arm'); near(av.len.left.arm[1], 0.26 * av.s, 2e-3, 'measured forearm');
near(av.len.right.leg[0], 0.43 * av.s, 2e-3, 'measured thigh');
ok(`real Blender skinned GLB loads through three's GLTFLoader and binds as rigged (scale ${av.s.toFixed(3)}, arm ${av.len.left.arm.map(v => v.toFixed(3))})`);

const wp = o => new THREE.Vector3().setFromMatrixPosition(o.matrixWorld);
const mkChar = (over = {}) => ({ pos: [0.5, 0, -0.3], yaw: 0.6, crouch: 0, walk: 0, arms: { left: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], right: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] }, gaze: [0, 0, 1], ...over });
const setArms = (c, side, hand, elbowHint, shoulder = [0, 1.4, 0]) => { c.arms[side] = [shoulder, elbowHint, hand]; return c; };   // engine arm: [shoulder, elbow, hand]

// place the holder, read where the model's shoulders really are, then ask for reachable hand targets near them
let c = mkChar(); av.update(c, 1 / 60);
const { fwd, left } = frame(c.yaw);
for (const [side, sgn] of [['left', 1], ['right', -1]]) {
  const sh = wp(B.arms[side].upper), reach = av.len[side].arm[0] + av.len[side].arm[1];
  for (const [f, u, o] of [[0.45, -0.20, 0.05], [0.3, 0.25, 0.1], [0.5, 0.0, sgn * 0.15], [0.0, -0.45, 0.0]]) {
    const hand = sh.clone().addScaledVector(new THREE.Vector3(...fwd), f).addScaledVector(new THREE.Vector3(...left), sgn * o).add(new THREE.Vector3(0, u, 0));
    assert.ok(hand.distanceTo(sh) < reach - 0.01, 'target must be reachable for this check');
    const mid = sh.clone().add(hand).multiplyScalar(0.5), elbowHint = mid.clone().add(new THREE.Vector3(0, -0.2, 0).addScaledVector(new THREE.Vector3(...fwd), -0.15));
    const cc = mkChar(); setArms(cc, side, hand.toArray(), elbowHint.toArray(), sh.toArray()); av.update(cc, 1 / 60);
    const A = B.arms[side], tip = wp(A.hand), sh2 = wp(A.upper), el = wp(A.fore);
    near(tip.distanceTo(hand), 0, 2e-3, `${side} hand reaches target`); near(sh2.distanceTo(el), av.len[side].arm[0], 1e-3, 'upper arm length kept'); near(el.distanceTo(tip), av.len[side].arm[1], 1e-3, 'forearm length kept');
    const offLine = el.clone().sub(mid), pole = elbowHint.clone().sub(mid);
    assert.ok(offLine.dot(pole) > 0, `${side} elbow bends toward the engine's elbow side`);
  }
}
ok('rigged arms: hand bone lands within 2 mm of every reachable target, bone lengths exact, elbow follows the engine\'s bend side');

c = mkChar(); setArms(c, 'left', [0.5 + 3, 1.4, -0.3], [0.5, 1.3, -0.3]); setArms(c, 'right', [0.5, 1.4, -0.3 + 3], [0.5, 1.3, -0.3]); av.update(c, 1 / 60);
for (const side of ['left', 'right']) { const A = B.arms[side], d = wp(A.upper).distanceTo(wp(A.hand)); near(d, av.len[side].arm[0] + av.len[side].arm[1], 2e-3, `${side} fully extended`); assert.ok(Number.isFinite(d)); }
ok('unreachable target: the arm stretches straight toward it, no NaN');

// legs / ground contact / facing
const toe = (() => { let t; av.holder.traverse(o => { if (o.isBone && /ToeBase/.test(o.name) && /Left/.test(o.name)) t = o; }); return t; })();
const stand = mkChar({ yaw: 1.1 }); for (let i = 0; i < 3; i++) av.update(stand, 1 / 60);
for (const side of ['left', 'right']) {
  const L = B.legs[side]; near(wp(L.foot).y, av.footRest[side], 2e-3, `${side} foot on the ground when standing`);
  near(wp(L.thigh).distanceTo(wp(L.shin)), av.len[side].leg[0], 1e-3, 'thigh length'); near(wp(L.shin).distanceTo(wp(L.foot)), av.len[side].leg[1], 1e-3, 'shin length');
}
const f1 = frame(1.1).fwd; assert.ok(wp(toe).sub(wp(B.legs.left.foot)).dot(new THREE.Vector3(...f1)) > 0.03, 'the model faces the character\'s forward direction');
const crouch = mkChar({ yaw: 1.1, crouch: 0.6 }); for (let i = 0; i < 3; i++) av.update(crouch, 1 / 60);
for (const side of ['left', 'right']) {
  const L = B.legs[side]; near(wp(L.foot).y, av.footRest[side], 2e-3, 'feet stay planted while crouching');
  near(wp(L.thigh).distanceTo(wp(L.shin)), av.len[side].leg[0], 1e-3, 'thigh length kept'); assert.ok(wp(L.thigh).y < wp(B.legs[side].foot).y + av.len[side].leg[0] + av.len[side].leg[1] - 0.2, 'hips came down');
}
assert.ok(wp(B.hips).y < 0.95 * av.s - 0.3, 'hips are lower when crouching');
let strideOk = false; for (let i = 0; i < 40; i++) { const w = mkChar({ yaw: 1.1, walk: i * 0.1 }); av.update(w, 1 / 60); if (wp(B.legs.left.foot).y > av.footRest.left + 0.02) strideOk = true; }
assert.ok(strideOk, 'a walking character lifts its feet');
ok('legs: feet planted when standing and crouching, bone lengths exact, model faces forward, feet lift when walking');

// static (unskinned) model
const st = await parseAvatar(buf(path.join(dir, 'kitchen.glb')));
assert.equal(st.mode, 'static'); assert.match(st.note, /no skeleton/);
st.avatar.update(mkChar({ pos: [2, 0, 1], yaw: 0.5, crouch: 0.4 }), 1 / 60);
const box = new THREE.Box3().setFromObject(st.avatar.holder); st.avatar.holder.updateMatrixWorld(true);
const box2 = new THREE.Box3().setFromObject(st.avatar.holder);
near(st.avatar.holder.position.x, 2, 1e-6); near(box2.max.y - box2.min.y, 1.75, 0.02, 'static figure scaled to height');
ok('unskinned GLB falls back to a static figure, scaled to 1.75 m and carried by the character');

// the viewport Rig: model arrives before the character is drawn; mode switches; cleanup
{
  const scene = new THREE.Scene(), rig = new Rig(scene);
  const snap = id => ({ chars: [{ id, name: id, color: '#c70000', pos: [1, 0, 2], yaw: 0.4, crouch: 0, walk: 0, speech: '', gaze: [0, 0, 1], holding: {},
    arms: { left: [[0.2, 1.4, 2], [0.3, 1.2, 2.1], [0.4, 1.1, 2.3]], right: [[-0.2, 1.4, 2], [-0.3, 1.2, 2.1], [-0.4, 1.1, 2.3]] } }], props: [] });
  const vis = (cr, g) => cr.parts[g].every(m => m.visible);
  const robot = await parseAvatar(buf(path.join(dir, 'robot.glb'))); rig.setAvatar('a', robot.avatar);                // before the character exists
  rig.sync(snap('a'), 1 / 60, null); const cr = rig.chars.get('a');
  assert.equal(cr.avatar, robot.avatar); assert.ok(!cr.parts.body.some(m => m.visible) && !cr.parts.arms.some(m => m.visible), 'rigged: all procedural parts hidden');
  assert.ok(cr.group.children.includes(robot.avatar.holder)); assert.equal(cr.headPos.y > 1.2, true, 'label anchor follows the model\'s head');
  const stat = await parseAvatar(buf(path.join(dir, 'kitchen.glb'))); rig.setAvatar('a', stat.avatar); rig.sync(snap('a'), 1 / 60, null);
  assert.ok(!cr.parts.body.some(m => m.visible) && vis(cr, 'arms'), 'static: body hidden, engine arms still shown so actions are visible');
  assert.ok(!cr.group.children.includes(robot.avatar.holder), 'the previous model is removed');
  rig.setAvatar('a', null); rig.sync(snap('a'), 1 / 60, null);
  assert.ok(vis(cr, 'body') && vis(cr, 'arms') && cr.avatar === null, 'clearing returns to the blacked-out entity');
  rig.setAvatar('a', (await parseAvatar(buf(path.join(dir, 'robot.glb')))).avatar); rig.sync({ chars: [], props: [] }, 1 / 60, null);
  assert.equal(rig.chars.size, 0); assert.equal(rig.avatars.size, 0); assert.equal(scene.children.length, 0, 'removing a character removes its model too');
  const m = cr.parts.body[0].material; assert.ok(m.color.r < 0.05 && m.color.g < 0.05, 'default body is blacked out');
}
ok('viewport integration: model-before-character, rigged/static/blackout switching, label anchor, cleanup');

await assert.rejects(parseAvatar(new TextEncoder().encode('not a glb at all').buffer), 'garbage input rejects instead of hanging');
ok('garbage input is rejected cleanly');
console.log(`\n${n} groups passed`);
