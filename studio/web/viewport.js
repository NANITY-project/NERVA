// three.js scene: characters as articulated figures driven by engine joint positions, props, imported environment.
// `Rig` needs no GPU and is unit-tested in Node; `createViewport` adds the renderer (browser only).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { interpolateSnapshot, legIK, footTargets, hipHeight, frame } from './logic.js';
import { parseAvatar } from './avatar.js';

const IVORY = 0xe8e2d9, RED = 0xa60000, RED_HI = 0xc70000, RED_DIM = 0x6b0000, VOID = 0x050505;
const UP = new THREE.Vector3(0, 1, 0);
const V = a => new THREE.Vector3(a[0], a[1], a[2]);

function setLimb(mesh, a, b) {
  const d = b.clone().sub(a), l = d.length();
  mesh.position.copy(a).addScaledVector(d, 0.5);
  mesh.scale.set(1, Math.max(l, 1e-5), 1);
  if (l > 1e-6) mesh.quaternion.setFromUnitVectors(UP, d.multiplyScalar(1 / l));
}

class CharRig {
  constructor(id, color) {
    this.id = id;
    this.group = new THREE.Group();
    // the default body is a blacked-out entity: a near-black glossy silhouette, with the character's colour only on accents
    const skin = new THREE.MeshStandardMaterial({ color: VOID, roughness: 0.28, metalness: 0.6, emissive: color, emissiveIntensity: 0.04 });
    const accent = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.2, emissive: color, emissiveIntensity: 0.45 });
    this.parts = { body: [], arms: [] };
    const limb = (r, m = skin, g = 'body') => { const x = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 10), m); this.group.add(x); this.parts[g].push(x); return x; };
    const ball = (r, m = skin, g = 'body') => { const x = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 10), m); this.group.add(x); this.parts[g].push(x); return x; };
    this.torso = limb(0.12); this.shoulderBar = limb(0.05); this.head = ball(0.105);
    this.nose = new THREE.Mesh(new THREE.ConeGeometry(0.028, 0.09, 8), accent); this.group.add(this.nose); this.parts.body.push(this.nose);
    const arm = () => [limb(0.045, skin, 'arms'), limb(0.036, skin, 'arms'), ball(0.042, accent, 'arms')];
    this.arms = { left: arm(), right: arm() };
    this.avatar = null;
    this.legs = { left: [limb(0.065), limb(0.05)], right: [limb(0.065), limb(0.05)] };
    this.feet = { left: ball(0.06), right: ball(0.06) };
    this.pelvis = ball(0.12);
    this.ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.012, 6, 48), new THREE.MeshBasicMaterial({ color: RED_HI }));
    this.ring.rotation.x = Math.PI / 2; this.ring.position.y = 0.01; this.ring.visible = false; this.group.add(this.ring);
    this.hit = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 1.8, 8), new THREE.MeshBasicMaterial({ visible: false }));
    this.hit.position.y = 0.9; this.hit.userData = { type: 'char', id }; this.group.add(this.hit);
    this.moving = 0; this.lastWalk = null;
    this.headPos = new THREE.Vector3();
  }

  setAvatar(av) {
    if (this.avatar) { this.group.remove(this.avatar.holder); this.avatar.dispose(); }
    this.avatar = av || null;
    if (av) this.group.add(av.holder);
    const mode = av?.mode || 'blackout';
    for (const m of this.parts.body) m.visible = mode === 'blackout';                       // body is replaced by the model
    for (const m of this.parts.arms) m.visible = mode !== 'rigged';                          // a rigged model animates its own arms
  }

  update(c, dt, selected) {
    this.avatar?.update(c, dt);
    if (this.lastWalk !== null && dt > 0) {
      const speed = Math.abs(c.walk - this.lastWalk) / dt;                 // steps/s
      this.moving += (Math.min(1, speed / 3.2) - this.moving) * Math.min(1, dt * 10);
    }
    this.lastWalk = c.walk;
    const { fwd } = frame(c.yaw);
    const hipY = hipHeight(c);
    const pelvis = new THREE.Vector3(c.pos[0], hipY, c.pos[2]);
    const sL = V(c.arms.left[0]), sR = V(c.arms.right[0]);
    const neck = sL.clone().add(sR).multiplyScalar(0.5); neck.y += 0.05;
    this.pelvis.position.copy(pelvis);
    setLimb(this.torso, pelvis, neck); setLimb(this.shoulderBar, sL, sR);
    this.head.position.copy(neck).add(new THREE.Vector3(0, 0.16, 0)); this.headPos.copy(this.head.position);
    const g = V(c.gaze).normalize();
    this.nose.position.copy(this.head.position).addScaledVector(g, 0.115);
    this.nose.quaternion.setFromUnitVectors(UP, g);
    for (const side of ['left', 'right']) {
      const [a, e, h] = c.arms[side].map(V);
      setLimb(this.arms[side][0], a, e); setLimb(this.arms[side][1], e, h); this.arms[side][2].position.copy(h);
    }
    const feet = footTargets(c, this.moving);
    const { left } = frame(c.yaw);
    for (const [side, s] of [['left', 1], ['right', -1]]) {
      const hip = pelvis.clone().addScaledVector(V(left), s * 0.10);
      const foot = V(feet[side]);
      const knee = V(legIK([hip.x, hip.y, hip.z], feet[side], fwd));
      setLimb(this.legs[side][0], hip, knee); setLimb(this.legs[side][1], knee, foot); this.feet[side].position.copy(foot);
    }
    this.ring.position.set(c.pos[0], 0.01, c.pos[2]); this.ring.visible = !!selected;
    this.hit.position.set(c.pos[0], 0.9, c.pos[2]);
    if (this.avatar) this.headPos.set(...this.avatar.headWorld());
  }
  dispose() { this.avatar?.dispose(); this.group.traverse(o => { if (!this.avatar || !this.avatar.holder.getObjectById?.(o.id)) { o.geometry?.dispose(); o.material?.dispose?.(); } }); }
}

class PropRig {
  constructor(p) {
    this.name = p.name;
    this.group = new THREE.Group();
    const fillOpacity = p.kind === 'mesh' ? 0.07 : 0.32;
    this.box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color: p.kind === 'mesh' ? RED_DIM : RED, transparent: true, opacity: fillOpacity, roughness: 0.6 }));
    this.edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: RED_HI, transparent: true, opacity: 0.75 }));
    this.box.userData = { type: 'prop', id: p.name };
    this.group.add(this.box, this.edges);
  }
  update(p, selected) {
    this.group.position.set(p.pos[0], p.pos[1], p.pos[2]);
    this.group.scale.set(p.size[0], p.size[1], p.size[2]);
    this.edges.material.opacity = selected ? 1 : (p.held ? 0.4 : 0.6);
    this.edges.material.color.setHex(selected ? 0xffffff : RED_HI);
  }
  dispose() { this.box.geometry.dispose(); this.box.material.dispose(); this.edges.geometry.dispose(); this.edges.material.dispose(); }
}

export class Rig {
  constructor(scene) { this.scene = scene; this.chars = new Map(); this.props = new Map(); this.last = 0; this.avatars = new Map(); }
  // a model may arrive before its character has been drawn: remember it and attach on creation
  setAvatar(id, av) { const old = this.avatars.get(id); if (old && old !== av && !this.chars.get(id)) old.dispose(); if (av) this.avatars.set(id, av); else this.avatars.delete(id); this.chars.get(id)?.setAvatar(av); }
  sync(snap, dt, selected) {
    const seenC = new Set(), seenP = new Set();
    for (const c of snap.chars) {
      seenC.add(c.id);
      let r = this.chars.get(c.id);
      if (!r) { r = new CharRig(c.id, new THREE.Color(c.color)); this.chars.set(c.id, r); this.scene.add(r.group); if (this.avatars.has(c.id)) r.setAvatar(this.avatars.get(c.id)); }
      r.update(c, dt, selected?.type === 'char' && selected.id === c.id);
    }
    for (const p of snap.props) {
      seenP.add(p.name);
      let r = this.props.get(p.name);
      if (!r) { r = new PropRig(p); this.props.set(p.name, r); this.scene.add(r.group); }
      r.update(p, selected?.type === 'prop' && selected.id === p.name);
    }
    for (const [k, r] of this.chars) if (!seenC.has(k)) { this.scene.remove(r.group); r.dispose(); this.chars.delete(k); this.avatars.delete(k); }
    for (const [k, r] of this.props) if (!seenP.has(k)) { this.scene.remove(r.group); r.dispose(); this.props.delete(k); }
  }
  pickables() { return [...this.chars.values()].map(r => r.hit).concat([...this.props.values()].map(r => r.box)); }
}

export function createViewport(canvas) {
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true }); }
  catch (e) {
    // three.js r163+ needs WebGL2. Firefox reports the real reason in about:support under Graphics.
    throw new Error(`could not create a WebGL2 context (${e && e.message || e}). Check that hardware acceleration is on and that about:support shows "WebGL 2 Renderer"`);
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  scene.fog = new THREE.Fog(0x000000, 12, 34);
  scene.add(new THREE.HemisphereLight(0xe8e2d9, 0x1a0505, 0.75));
  const key = new THREE.DirectionalLight(0xffe9dd, 1.1); key.position.set(3, 6, -2); scene.add(key);
  const rim = new THREE.PointLight(RED, 18, 14); rim.position.set(-3, 2.2, 3); scene.add(rim);
  const grid = new THREE.GridHelper(40, 80, RED_DIM, 0x1a0505); grid.material.transparent = true; grid.material.opacity = 0.55; scene.add(grid);
  const camera = new THREE.PerspectiveCamera(48, 1, 0.05, 120);
  camera.position.set(3.6, 2.5, -3.2);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 1.0, 1.4); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.495; controls.update();
  const rig = new Rig(scene);
  const env = new THREE.Group(); scene.add(env);
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

  let prev = null, curr = null, tPrev = 0, tCurr = 0, selected = null, lastNow = performance.now(), frozen = null;
  const resize = () => {
    const r = canvas.getBoundingClientRect(); if (!r.width || !r.height) return;
    renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize); ro.observe(canvas); resize();

  const setNdc = (cx, cy) => { const r = canvas.getBoundingClientRect(); ndc.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1); ray.setFromCamera(ndc, camera); };
  let raf = 0;
  const loop = now => {
    raf = requestAnimationFrame(loop);
    const dt = Math.min((now - lastNow) / 1000, 0.1); lastNow = now;
    const snap = frozen || (curr && (prev && tCurr > tPrev ? interpolateSnapshot(prev, curr, (now - tCurr) / (tCurr - tPrev)) : curr));
    if (snap) rig.sync(snap, dt, selected);
    controls.update(); renderer.render(scene, camera);
  };
  raf = requestAnimationFrame(loop);

  return {
    rig, camera,
    push(snap) { prev = curr; tPrev = tCurr; curr = snap; tCurr = performance.now(); },
    freeze(snap) { frozen = snap; },                  // playback: show exactly this snapshot (null = live)
    select(sel) { selected = sel; },
    pick(cx, cy) {
      setNdc(cx, cy);
      const hit = ray.intersectObjects(rig.pickables(), false)[0];
      return hit ? { type: hit.object.userData.type, id: hit.object.userData.id } : null;
    },
    ground(cx, cy) { setNdc(cx, cy); const p = new THREE.Vector3(); return ray.ray.intersectPlane(plane, p) ? [p.x, p.z] : null; },
    project(world) {
      const v = new THREE.Vector3(world[0], world[1], world[2]).project(camera);
      const r = canvas.getBoundingClientRect();
      return { x: (v.x * 0.5 + 0.5) * r.width, y: (-v.y * 0.5 + 0.5) * r.height, visible: v.z > -1 && v.z < 1 };
    },
    headOf(id) { return rig.chars.get(id)?.headPos.toArray() ?? null; },
    async setCharacterModel(id, buffer) { const r = await parseAvatar(buffer); rig.setAvatar(id, r.avatar); return { mode: r.mode, note: r.note }; },
    clearCharacterModel(id) { rig.setAvatar(id, null); },
    async loadEnvironment(buffer) {
      const gltf = await new Promise((res, rej) => new GLTFLoader().parse(buffer, '', res, rej));
      env.clear(); env.add(gltf.scene);
      return new THREE.Box3().setFromObject(gltf.scene);
    },
    clearEnvironment() { env.clear(); },
    dispose() { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); renderer.dispose(); },
  };
}
