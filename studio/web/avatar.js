// Character models: bind a humanoid .glb to the engine's joint positions. Pure three.js (no GPU needed), unit-tested in Node.
//
// rigged  - a humanoid skeleton is found by bone NAME (Mixamo, VRM/VRoid, Rigify, generic). Arms and legs are posed with
//           two-bone IK using the MODEL'S OWN bone lengths, so hands reach their targets even if the proportions differ
//           from the engine's. Each bone is aimed (shortest arc, parent first): no rest-pose axis conventions needed.
// static  - no skeleton: the mesh is scaled to height and carried around like a puppet (the engine's arms stay visible).
// Models should face +Z (the glTF convention) and stand on y=0 in their bind pose.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { legIK, footTargets, frame } from './logic.js';

const V = a => new THREE.Vector3(a[0], a[1], a[2]);
const FINGER = new Set(['thumb', 'index', 'middle', 'ring', 'pinky', 'little', 'finger', 'toe', 'toes', 'twist', 'roll']);
const NOISE = new Set(['mixamorig', 'bip', 'j', 'armature', 'bone', 'def', 'org', 'cc', 'base']);

export const tokenize = name => (String(name).match(/[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+/g) || []).map(t => t.toLowerCase());

// Which limb part is this bone? -> { side: 'left'|'right'|null, part } (part null = not one we drive)
export function classifyBone(name) {
  const toks = tokenize(name);
  const side = toks.includes('left') || toks.includes('l') ? 'left' : toks.includes('right') || toks.includes('r') ? 'right' : null;
  const core = toks.filter(t => !['left', 'right', 'l', 'r'].includes(t) && !NOISE.has(t) && !/^\d+$/.test(t));
  const S = new Set(core), str = core.join('');
  const none = { side, part: null };
  if (core.some(t => FINGER.has(t))) return none;
  if (!side) {
    if (S.has('head') && !S.has('top') && !S.has('end')) return { side: null, part: 'head' };
    if (S.has('hips') || S.has('pelvis')) return { side: null, part: 'hips' };
    return none;
  }
  if (S.has('hand')) return { side, part: 'hand' };
  if (S.has('foot')) return { side, part: 'foot' };
  if (str.includes('forearm') || str.includes('lowerarm') || (S.has('fore') && S.has('arm')) || (S.has('lower') && S.has('arm'))) return { side, part: 'fore' };
  if (str.includes('upperarm') || (S.has('upper') && S.has('arm')) || (S.has('arm') && core.length === 1)) return { side, part: 'upper' };
  if (str.includes('upleg') || str.includes('upperleg') || S.has('thigh') || (S.has('up') && S.has('leg')) || (S.has('upper') && S.has('leg'))) return { side, part: 'thigh' };
  if (S.has('shin') || S.has('calf') || str.includes('lowerleg') || (S.has('lower') && S.has('leg')) || (S.has('leg') && core.length === 1)) return { side, part: 'shin' };
  return none;
}

const isDescendant = (a, anc) => { for (let o = a.parent; o; o = o.parent) if (o === anc) return true; return false; };

// -> { arms:{left:{upper,fore,hand},right}, legs:{left|null,right|null}, head, hips } or { error }
export function bindHumanoid(root) {
  const found = { left: {}, right: {} }, extra = {};
  root.traverse(o => {
    if (!o.isBone) return;
    const { side, part } = classifyBone(o.name);
    if (!part) return;
    if (side) { if (!found[side][part]) found[side][part] = o; } else if (!extra[part]) extra[part] = o;
  });
  const arms = {}, legs = {};
  for (const side of ['left', 'right']) {
    const f = found[side];
    if (!f.upper || !f.fore || !f.hand) return { error: `no ${side} arm chain (upper arm / forearm / hand) found in the bone names` };
    if (!isDescendant(f.fore, f.upper) || !isDescendant(f.hand, f.fore)) return { error: `${side} arm bones are not parented as upper > fore > hand` };
    arms[side] = { upper: f.upper, fore: f.fore, hand: f.hand };
    legs[side] = f.thigh && f.shin && f.foot && isDescendant(f.shin, f.thigh) && isDescendant(f.foot, f.shin) ? { thigh: f.thigh, shin: f.shin, foot: f.foot } : null;
  }
  return { arms, legs, head: extra.head || null, hips: extra.hips || null };
}

const wpos = o => new THREE.Vector3().setFromMatrixPosition(o.matrixWorld);

// Rotate `bone` (parents already posed) so the direction bone -> child points at `target` (world space).
export function aimBone(bone, child, target) {
  bone.updateWorldMatrix(true, true);
  const p = wpos(bone), cur = wpos(child).sub(p), want = target.clone().sub(p);
  if (cur.lengthSq() < 1e-12 || want.lengthSq() < 1e-12) return;
  const q = new THREE.Quaternion().setFromUnitVectors(cur.normalize(), want.normalize());
  const wq = bone.getWorldQuaternion(new THREE.Quaternion());
  const pq = bone.parent ? bone.parent.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
  bone.quaternion.copy(pq.invert().multiply(q.multiply(wq)));
  bone.updateMatrixWorld(true);
}

function measureBox(root) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty() || !isFinite(box.min.y) || box.max.y - box.min.y < 1e-4) {
    box.makeEmpty(); root.traverse(o => { if (o.isBone) box.expandByPoint(wpos(o)); });
  }
  return box;
}

export class Avatar {
  constructor(scene, binding, height = 1.75) {
    this.mode = 'rigged'; this.binding = binding;
    this.holder = new THREE.Group(); this.holder.add(scene);
    const box = measureBox(scene), h = box.max.y - box.min.y;
    this.s = height / h; this.groundY = -box.min.y * this.s;
    this.holder.scale.setScalar(this.s); this.holder.position.y = this.groundY; this.holder.updateMatrixWorld(true);
    scene.traverse(o => { if (o.isMesh) o.frustumCulled = false; });          // skinned vertices leave the bind-pose bounds
    const d = (a, b) => wpos(a).distanceTo(wpos(b));
    this.len = { left: {}, right: {} }; this.footRest = {};
    for (const side of ['left', 'right']) {
      const A = binding.arms[side]; this.len[side].arm = [d(A.upper, A.fore), d(A.fore, A.hand)];
      const L = binding.legs[side]; if (L) { this.len[side].leg = [d(L.thigh, L.shin), d(L.shin, L.foot)]; this.footRest[side] = wpos(L.foot).y; }
    }
    this.moving = 0; this.lastWalk = null; this.headPos = new THREE.Vector3(0, 1.7, 0);
  }

  update(c, dt) {
    if (this.lastWalk !== null && dt > 0) this.moving += (Math.min(1, Math.abs(c.walk - this.lastWalk) / dt / 3.2) - this.moving) * Math.min(1, dt * 10);
    this.lastWalk = c.walk;
    this.holder.position.set(c.pos[0], this.groundY - c.crouch * 0.8, c.pos[2]);
    this.holder.rotation.y = c.yaw; this.holder.updateMatrixWorld(true);
    const { fwd } = frame(c.yaw), B = this.binding;
    for (const side of ['left', 'right']) {
      const L = B.legs[side];
      if (L) {
        const ft = footTargets(c, this.moving)[side], foot = new THREE.Vector3(ft[0], this.footRest[side] + (ft[1] - 0.05), ft[2]);
        const [l1, l2] = this.len[side].leg, hip = wpos(L.thigh);
        const knee = V(legIK([hip.x, hip.y, hip.z], [foot.x, foot.y, foot.z], fwd, l1, l2));
        aimBone(L.thigh, L.shin, knee); aimBone(L.shin, L.foot, foot);
      }
      const A = B.arms[side], [se, ee, he] = c.arms[side].map(V), [a1, a2] = this.len[side].arm;
      const sh = wpos(A.upper), pole = ee.clone().sub(se.clone().add(he).multiplyScalar(0.5));
      if (pole.lengthSq() < 1e-8) pole.set(-fwd[0], -1, -fwd[2]);
      const elbow = V(legIK([sh.x, sh.y, sh.z], [he.x, he.y, he.z], [pole.x, pole.y, pole.z], a1, a2));
      aimBone(A.upper, A.fore, elbow); aimBone(A.fore, A.hand, he);
    }
    this.headPos.copy(B.head ? wpos(B.head) : new THREE.Vector3(c.pos[0], 1.7, c.pos[2]));
  }
  headWorld() { return this.headPos.toArray(); }
  dispose() { this.holder.traverse(o => { o.geometry?.dispose?.(); const m = o.material; (Array.isArray(m) ? m : m ? [m] : []).forEach(x => { x.map?.dispose?.(); x.dispose?.(); }); }); }
}

export class StaticAvatar {
  constructor(scene, height = 1.75) {
    this.mode = 'static'; this.holder = new THREE.Group(); this.holder.add(scene);
    const box = measureBox(scene), h = Math.max(box.max.y - box.min.y, 1e-3);
    this.s = height / h; this.groundY = -box.min.y * this.s;
    this.holder.scale.setScalar(this.s); this.holder.position.y = this.groundY;
    this.headPos = new THREE.Vector3(0, 1.7, 0);
  }
  update(c) {
    this.holder.position.set(c.pos[0], this.groundY - c.crouch * 0.45, c.pos[2]); this.holder.rotation.y = c.yaw;
    this.headPos.set(c.pos[0], 1.7 - c.crouch * 0.6, c.pos[2]);
  }
  headWorld() { return this.headPos.toArray(); }
  dispose() { this.holder.traverse(o => { o.geometry?.dispose?.(); const m = o.material; (Array.isArray(m) ? m : m ? [m] : []).forEach(x => { x.map?.dispose?.(); x.dispose?.(); }); }); }
}

// ArrayBuffer (.glb) -> Avatar | StaticAvatar. `note` explains a static fallback.
export async function parseAvatar(buffer, height = 1.75) {
  const gltf = await new Promise((res, rej) => new GLTFLoader().parse(buffer, '', res, rej));
  const scene = gltf.scene;
  let skinned = false; scene.traverse(o => { if (o.isSkinnedMesh) skinned = true; });
  if (!skinned) return { avatar: new StaticAvatar(scene, height), mode: 'static', note: 'no skeleton in the file: used as a static figure' };
  const binding = bindHumanoid(scene);
  if (binding.error) return { avatar: new StaticAvatar(scene, height), mode: 'static', note: binding.error + ' (used as a static figure)' };
  return { avatar: new Avatar(scene, binding, height), mode: 'rigged', note: binding.legs.left && binding.legs.right ? '' : 'no leg bones found: legs will not animate' };
}
