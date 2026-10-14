// Pure logic (no DOM, no three.js): unit-tested in Node. World is Y-up meters.
export const VERBS = ['SAY', 'WALK_TO', 'LOOK_AT', 'GRAB', 'OFFER', 'DROP', 'REACH', 'POINT', 'WAVE', 'WAIT', 'RESET'];
const NEEDS_HAND = new Set(['GRAB', 'OFFER', 'DROP', 'REACH', 'POINT', 'WAVE', 'RESET']);
const NEEDS_NAME = new Set(['WALK_TO', 'LOOK_AT', 'GRAB', 'OFFER', 'REACH', 'POINT']);

export const lerp = (a, b, t) => a + (b - a) * t;
export const lerpVec = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
export function lerpAngle(a, b, t) {
  let d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return a + d * t;
}

// Blend two snapshots (matching characters/props by id) so 60 fps rendering is smooth from 30 Hz data.
export function interpolateSnapshot(prev, curr, t) {
  if (!prev || t >= 1) return curr;
  t = Math.max(0, t);
  const pc = new Map(prev.chars.map(c => [c.id, c]));
  const pp = new Map(prev.props.map(p => [p.name, p]));
  const chars = curr.chars.map(c => {
    const o = pc.get(c.id);
    if (!o) return c;
    return {
      ...c,
      pos: lerpVec(o.pos, c.pos, t), yaw: lerpAngle(o.yaw, c.yaw, t),
      crouch: lerp(o.crouch, c.crouch, t), walk: lerp(o.walk, c.walk, t), gaze: lerpVec(o.gaze, c.gaze, t),
      arms: Object.fromEntries(Object.keys(c.arms).map(s => [s, o.arms[s] ? c.arms[s].map((p, i) => lerpVec(o.arms[s][i], p, t)) : c.arms[s]])),
    };
  });
  const props = curr.props.map(p => { const o = pp.get(p.name); return o ? { ...p, pos: lerpVec(o.pos, p.pos, t) } : p; });
  return { ...curr, chars, props };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// Two-bone leg IK (law of cosines). Knee bends toward `fwd`.
export function legIK(hip, foot, fwd, thigh = 0.45, shin = 0.45) {
  const to = sub(foot, hip);
  const d = Math.min(Math.max(len(to), Math.abs(thigh - shin) + 1e-4), thigh + shin - 1e-4);
  const dir = unit(to);
  const a = (thigh * thigh - shin * shin + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(thigh * thigh - a * a, 0));
  let perp = sub(fwd, mul(dir, dot(fwd, dir)));
  perp = len(perp) < 1e-6 ? [0, 0, 1] : unit(perp);
  return add(add(hip, mul(dir, a)), mul(perp, h));
}

export const HIP_Y = 0.95, STANCE = 0.11, ANKLE_Y = 0.05;
export const frame = yaw => ({ fwd: [Math.sin(yaw), 0, Math.cos(yaw)], left: [Math.cos(yaw), 0, -Math.sin(yaw)] });

// Foot positions for a character; `moving` in 0..1 blends from standing to full stride.
export function footTargets(c, moving) {
  const { fwd, left } = frame(c.yaw);
  const wide = 1 + c.crouch * 0.9;                     // feet spread a little when crouching
  const out = {};
  for (const [side, s] of [['left', 1], ['right', -1]]) {
    const ang = c.walk * Math.PI + (s < 0 ? Math.PI : 0);
    const fwdOff = Math.sin(ang) * 0.22 * moving, lift = Math.max(0, Math.cos(ang)) * 0.10 * moving;
    out[side] = [c.pos[0] + left[0] * s * STANCE * wide + fwd[0] * fwdOff, ANKLE_Y + lift,
                 c.pos[2] + left[2] * s * STANCE * wide + fwd[2] * fwdOff];
  }
  return out;
}
export const hipHeight = c => HIP_Y - c.crouch * 0.8;

// Changes only when the *structure* of the scene changes, so lists aren't rebuilt (and inputs cleared) every frame.
export function structureKey(s) {
  return JSON.stringify([s.chars.map(c => [c.id, c.name, c.busy, c.queued, c.agent, c.color, c.holding.left, c.holding.right]),
                         s.props.map(p => [p.name, p.kind, p.held, p.tags.join(',')])]);
}

export const fmtTime = t => { const m = Math.floor(t / 60), s = t - m * 60; return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`; };

// Tab-completion candidates for the command box.
export function completions(text, snap) {
  const parts = text.replace(/^\s+/, '').split(/\s+/);
  const names = snap ? [...snap.props.map(p => p.name), ...snap.chars.map(c => c.id)] : [];
  if (parts.length <= 1) return VERBS.filter(v => v.startsWith((parts[0] || '').toUpperCase()));
  const verb = parts[0].toUpperCase(), last = parts[parts.length - 1].toLowerCase();
  if (NEEDS_HAND.has(verb) && parts.length === 2) return ['left', 'right'].filter(h => h.startsWith(last));
  if (NEEDS_NAME.has(verb)) return names.filter(n => n.startsWith(last));
  return [];
}
export function applyCompletion(text, choice) {
  const m = text.match(/^(.*?)(\S*)$/s);
  return m[1] + choice + (VERBS.includes(choice) ? ' ' : ' ');
}
