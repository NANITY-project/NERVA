#!/usr/bin/env python3
"""
view_clip.py - watch clips from gen_dataset.py. Writes ONE html file (no server, no install) that you open in a browser.

  python tools/view_clip.py                                  # first 6 clips of data/clips.npz -> view.html
  python tools/view_clip.py data/clips.npz --clips 3 17 40   # pick clips
  python tools/view_clip.py --clips 3 --corrupt noise10      # the engine's answer to a BAD clip, over the clean one (grey ghost)
  python tools/view_clip.py --open                           # open it in the default browser when done

Every clip is replayed through the engine (the same tracking the verdict uses), so what you see is the engine's body, not just the numbers:
  - skeleton (left blue, right orange), planted feet as green discs, centre of mass (pink) with its line to the ground, pelvis trail
  - red border + the matching verdict text on frames the engine had to correct or lost balance on
  - red crosses with a line: where the clip ASKED a foot/hand to be, when the engine could not (or would not) put it there
Controls: drag = orbit, wheel = zoom, space = play/pause, left/right = step a frame, [ and ] = previous/next clip, 1-5 = camera presets.
--corrupt: noise5 noise10 noise20 blur blend flip far slow   (same corruptions as gen_dataset --stress)
"""
import argparse, base64, json, os, re, sys, webbrowser
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_dataset as g

CORRUPT = {"noise5": "noise 5 mm", "noise10": "noise 10 mm", "noise20": "noise 20 mm", "blur": "blur (7-frame average)", "blend": "blend with another clip",
           "flip": "5% contact flags flipped", "far": "feet 30% farther out", "slow": "slowed x1.4 (control: should pass)"}


def replay(lib, actions, dt):
    """Track the clip from a standing body with the engine's own clip tracking (landing look-ahead included, so it agrees with the verdict)."""
    b = g.new_standing(lib); a = np.ascontiguousarray(actions, np.float32); F, nb = len(a), len(b.names)
    rep = np.zeros((F, 12), np.float32); st = np.zeros((F, lib.S), np.float32); bo = np.zeros((F, nb, 10), np.float32)
    lib.L.nebula_body_replay_clip(b.h, g.fptr(a), F, dt, g.fptr(rep), g.fptr(st), g.fptr(bo)); b.close()
    pts = np.concatenate([st[:, 11:14], st[:, 17:20], st[:, 23:26], st[:, 26:29], a[:, 4:7], a[:, 9:12], a[:, 14:17], a[:, 18:21]], axis=1)
    return dict(rep9=rep[:, [0, 1, 2, 3, 4, 5, 6, 8, 9]], bones=bo[:, :, :6], com=st[:, 4:7], margin=rep[:, 10], bad=(rep[:, 11] > 0.5).astype(np.int8), pts=pts)


def pack(rp, actions):
    mm = lambda x: np.clip(np.rint(np.asarray(x) * 1000), -32000, 32000).astype(np.int16)
    return dict(bones=base64.b64encode(mm(rp["bones"]).tobytes()).decode(), com=mm(rp["com"]).ravel().tolist(), margin=mm(rp["margin"]).tolist(),
                bad=rp["bad"].tolist(), pts=mm(rp["pts"]).ravel().tolist(), rep=mm(rp["rep9"]).ravel().tolist() if "rep9" in rp else [], contact=np.asarray(actions)[:, [8, 13]].round().astype(int).ravel().tolist(), frames=len(actions))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="?", default="data/clips.npz"); ap.add_argument("--clips", type=int, nargs="+"); ap.add_argument("--corrupt", choices=sorted(CORRUPT))
    ap.add_argument("--out", default="view.html"); ap.add_argument("--open", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib"); ap.add_argument("--repo"); ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.npz):
        sys.exit(f"{args.npz} not found: run  python tools/gen_dataset.py --n 100 --out data  first (or pass the path to your clips.npz)")
    z = np.load(args.npz); acts, off, dt = z["actions"], z["offsets"], float(z["dt"]); N = len(off) - 1
    ids = args.clips or list(range(min(6 if not args.corrupt else 3, N)))
    if any(i < 0 or i >= N for i in ids):
        sys.exit(f"clip numbers must be 0..{N - 1}")
    lib = g.Lib(args.lib, args.repo, build=not args.no_build); chk = g.new_standing(lib); rng = np.random.default_rng(args.seed)
    names = None; clips = []
    for i in ids:
        a = acts[off[i]:off[i + 1]]; st = z["style"][i]; style = ", ".join(f"{n} {v:.2f}" for n, v in zip(z["style_names"], st) if n in ("stride", "step_time", "turn_deg", "torso_lean", "look_down_deg"))
        shown, ghost, label = a, None, f"clip {i}  ({style})"
        if args.corrupt:
            other = acts[off[(i + 1) % N]:off[(i + 1) % N + 1]]; cs, _ = g.corruptions(rng, a, other, dt); shown = cs[CORRUPT[args.corrupt]].astype(np.float32)
            ghost = replay(lib, a, dt); label = f"clip {i} + {CORRUPT[args.corrupt]}: engine's answer (colour) over the clean clip (grey)"
        v = chk.evaluate(shown, dt); rp = replay(lib, shown, dt)
        if names is None:
            b = g.new_standing(lib); names = b.names; b.close()
        c = pack(rp, shown); c["label"] = label
        c["verdict"] = dict(feasible=v["feasible"], score=v["score"], bad_frames=v["bad_frames"], issues=v["issues"])
        c["spans"] = [dict(a=int(m.group(1)), b=int(m.group(2)), t=t) for t in v["issues"] for m in [re.search(r"frames (\d+)-(\d+)", t)] if m]
        if ghost is not None:
            c["ghost"] = pack(ghost, a)["bones"]
        clips.append(c)
        print(f"  clip {i}: {'feasible' if v['feasible'] else 'NOT feasible'}, score {v['score']:.4f}, {len(shown)} frames")
    html = TEMPLATE.replace("__DATA__", json.dumps(dict(names=names, dt=dt, clips=clips), separators=(",", ":")))
    open(args.out, "w", encoding="utf-8").write(html)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB): open it in a browser")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.out))


TEMPLATE = r'''<!doctype html><html><head><meta charset="utf-8"><title>NERVA clip viewer</title><style>
html,body{margin:0;height:100%;background:#15171c;color:#d8dbe2;font:13px system-ui,sans-serif;overflow:hidden}
#top{position:absolute;left:0;right:0;top:0;padding:8px 10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;background:#1d2027cc;z-index:2}
select,button{background:#2a2e38;color:#d8dbe2;border:1px solid #3a3f4c;border-radius:5px;padding:4px 8px;font:inherit;cursor:pointer}button:hover{background:#343947}
#cv{position:absolute;left:0;top:0;width:100%;height:100%}
#bot{position:absolute;left:0;right:0;bottom:0;padding:8px 12px 10px;background:#1d2027cc;z-index:2}
#info{margin-bottom:4px;line-height:1.45}#strip{width:100%;height:10px;display:block;margin-top:4px;border-radius:3px}input[type=range]{width:100%;margin:2px 0}
.bad{color:#ff6b6b}details{margin:2px 0}#alli{max-height:28vh;overflow:auto}summary{cursor:pointer}.good{color:#6fdc8c}.dim{color:#8b91a1}.iss{margin-left:10px}.iss.on{color:#ffb4b4;font-weight:600}
</style></head><body><canvas id="cv"></canvas>
<div id="top"><select id="clip"></select><button id="play">pause</button><select id="speed"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
<button data-v="0">3/4</button><button data-v="1">side</button><button data-v="2">front</button><button data-v="3">back</button><button data-v="4">top</button>
<label><input type="checkbox" id="follow" checked> follow</label><label><input type="checkbox" id="ghost" checked> clean ghost</label><span class="dim">red ring: engine corrected this · red cross: asked vs delivered · drag: orbit · wheel: zoom · space · ←→ · [ ]</span></div>
<div id="bot"><div id="info"></div><div id="act"></div><details id="all"><summary class="dim"></summary><div id="alli"></div></details><input type="range" id="seek" min="0" value="0"><canvas id="strip"></canvas></div>
<script>
const D=__DATA__, NB=D.names.length, cv=document.getElementById('cv'), ctx=cv.getContext('2d');
function dec(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new Int16Array(u.buffer)}
D.clips.forEach(c=>{c.B=dec(c.bones);if(c.ghost)c.G=dec(c.ghost)});
let ci=0,f=0,playing=true,speed=1,follow=true,showGhost=true,cam={yaw:0.75,pitch:0.22,dist:3.6},last=performance.now();
const sel=document.getElementById('clip');D.clips.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'. '+c.label;sel.appendChild(o)});
const VIEWS=[[0.75,0.22,3.6],[Math.PI/2,0.05,3.6],[0,0.05,3.6],[Math.PI,0.05,3.6],[0.0,1.5,4.2]];
function setClip(i){ci=Math.max(0,Math.min(D.clips.length-1,i));sel.value=ci;f=0;lastKey='';const c=D.clips[ci];seek.max=c.frames-1;drawStrip();document.getElementById('ghost').parentElement.style.display=c.G?'':'none'}
const seek=document.getElementById('seek'),strip=document.getElementById('strip');
function drawStrip(){const c=D.clips[ci],W=strip.width=strip.clientWidth,s=strip.getContext('2d');strip.height=10;s.fillStyle='#2b3a2f';s.fillRect(0,0,W,10);s.fillStyle='#e05555';for(let i=0;i<c.frames;i++)if(c.bad[i])s.fillRect(Math.floor(i/c.frames*W),0,Math.ceil(W/c.frames)+1,10)}
// ---- camera
let eye,R,U,Fw,FOC;
function setCam(W,H,t){const cp=Math.cos(cam.pitch),sp=Math.sin(cam.pitch);eye=[t[0]+cam.dist*cp*Math.sin(cam.yaw),t[1]+cam.dist*sp,t[2]+cam.dist*cp*Math.cos(cam.yaw)];
 let fw=[t[0]-eye[0],t[1]-eye[1],t[2]-eye[2]];const n=Math.hypot(...fw);fw=fw.map(v=>v/n);let r=[fw[1]*0-fw[2]*1,fw[2]*0-fw[0]*0,fw[0]*1-fw[1]*0];const rn=Math.hypot(...r)||1;r=r.map(v=>v/rn);
 Fw=fw;R=r;U=[r[1]*fw[2]-r[2]*fw[1],r[2]*fw[0]-r[0]*fw[2],r[0]*fw[1]-r[1]*fw[0]];FOC=H*1.15}
function cy0(H){return H*0.40}
function P(x,y,z,W,H){const dx=x-eye[0],dy=y-eye[1],dz=z-eye[2];const cx=dx*R[0]+dy*R[1]+dz*R[2],cy=dx*U[0]+dy*U[1]+dz*U[2],cz=dx*Fw[0]+dy*Fw[1]+dz*Fw[2];if(cz<0.1)return null;const s=FOC/cz;return[W/2+cx*s,cy0(H)-cy*s,cz,s]}
function bone(arr,fr,k){const o=(fr*NB+k)*6;return[arr[o]/1000,arr[o+1]/1000,arr[o+2]/1000,arr[o+3]/1000,arr[o+4]/1000,arr[o+5]/1000]}
const side=n=>n.endsWith('_L')?1:n.endsWith('_R')?2:0, COL=['#d9dde6','#58a6ff','#ffa24a'];
function line(a,b,col,w,W,H,out,dep){const p=P(a[0],a[1],a[2],W,H),q=P(b[0],b[1],b[2],W,H);if(p&&q)out.push({p,q,col,w:w*Math.min(p[3],q[3])/300,d:(p[2]+q[2])/2+(dep||0)})}
function draw(){const W=cv.width=innerWidth,H=cv.height=innerHeight,c=D.clips[ci],fr=Math.min(c.frames-1,Math.floor(f));
 ctx.fillStyle='#15171c';ctx.fillRect(0,0,W,H);
 const pel=bone(c.B,fr,0);let tgt=follow?[pel[0],0.9,pel[2]]:[0,0.9,1.5];setCam(W,H,tgt);
 // ground grid
 const gx=Math.round(tgt[0]*2)/2,gz=Math.round(tgt[2]*2)/2;ctx.lineWidth=1;
 for(let i=-10;i<=10;i++){for(const ax of [0,1]){const a=ax?[gx-5,0,gz+i*0.5]:[gx+i*0.5,0,gz-5],b=ax?[gx+5,0,gz+i*0.5]:[gx+i*0.5,0,gz+5];const p=P(...a,W,H),q=P(...b,W,H);if(!p||!q)continue;
  const major=Math.abs((ax?gz+i*0.5:gx+i*0.5)%1)<1e-6;ctx.strokeStyle=major?'#2b3040':'#1f232d';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}}
 // pelvis trail
 ctx.strokeStyle='#3d5a80';ctx.lineWidth=2;ctx.beginPath();let st=false;for(let i=0;i<=fr;i++){const b=bone(c.B,i,0),p=P(b[0],0.005,b[2],W,H);if(!p)continue;if(!st){ctx.moveTo(p[0],p[1]);st=true}else ctx.lineTo(p[0],p[1])}ctx.stroke();
 const segs=[];
 if(c.G&&showGhost)for(let k=0;k<NB;k++){const b=bone(c.G,fr,k);line(b.slice(0,3),b.slice(3,6),'rgba(160,165,180,0.35)',9,W,H,segs,0.5)}
 for(let k=0;k<NB;k++){const b=bone(c.B,fr,k),s=side(D.names[k]),fing=/thumb|index|middle|ring|pinky/.test(D.names[k]);line(b.slice(0,3),b.slice(3,6),COL[s],fing?3:(/spine|neck|pelvis/.test(D.names[k])?13:10),W,H,segs,0)}
 segs.sort((a,b)=>b.d-a.d);ctx.lineCap='round';for(const s of segs){ctx.strokeStyle=s.col;ctx.lineWidth=Math.max(1.5,s.w);ctx.beginPath();ctx.moveTo(s.p[0],s.p[1]);ctx.lineTo(s.q[0],s.q[1]);ctx.stroke()}
 // corrections: a red ring on the foot / pelvis the engine had to move or could not reach
 const ring=(p3,r)=>{const p=P(p3[0],p3[1],p3[2],W,H);if(!p)return;ctx.strokeStyle='#ff4d4d';ctx.lineWidth=3;ctx.beginPath();ctx.arc(p[0],p[1],r*p[3],0,7);ctx.stroke()};
 if(c.rep.length){const o9=fr*9,m=i=>c.rep[o9+i]/1000;[['foot_L',0],['foot_R',1]].forEach(([n,sd])=>{if(Math.max(m(sd),m(2+sd),m(5+sd))>0.005){const k=D.names.indexOf(n);if(k>=0){const b=bone(c.B,fr,k);ring(b.slice(0,3),0.09)}}});if(m(4)>0.005){const b=bone(c.B,fr,0);ring(b.slice(0,3),0.14)}}
 // head
 const hi=D.names.indexOf('head');if(hi>=0){const b=bone(c.B,fr,hi),m=[(b[0]+b[3])/2,(b[1]+b[4])/2,(b[2]+b[5])/2],p=P(...m,W,H);if(p){ctx.fillStyle='rgba(217,221,230,0.85)';ctx.beginPath();ctx.arc(p[0],p[1],0.095*p[3],0,7);ctx.fill()}}
 // planted feet: green discs
 [['foot_L',0],['foot_R',1]].forEach(([n,j])=>{if(!c.contact[fr*2+j])return;const k=D.names.indexOf(n);if(k<0)return;const b=bone(c.B,fr,k),cx=b[3],cz=b[5];ctx.fillStyle='rgba(80,220,120,0.45)';ctx.beginPath();let s0=true;for(let a=0;a<=12;a++){const p=P(cx+0.09*Math.cos(a/12*6.283),0.003,cz+0.13*Math.sin(a/12*6.283),W,H);if(!p)continue;if(s0){ctx.moveTo(p[0],p[1]);s0=false}else ctx.lineTo(p[0],p[1])}ctx.fill()});
 // centre of mass
 const cm=[c.com[fr*3]/1000,c.com[fr*3+1]/1000,c.com[fr*3+2]/1000],pc=P(...cm,W,H),pg=P(cm[0],0.004,cm[2],W,H);if(pc&&pg){ctx.strokeStyle='#ff5fa8';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(pc[0],pc[1]);ctx.lineTo(pg[0],pg[1]);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#ff5fa8';ctx.beginPath();ctx.arc(pc[0],pc[1],5,0,7);ctx.fill();ctx.beginPath();ctx.arc(pg[0],pg[1],3,0,7);ctx.fill()}
 // asked-for vs delivered
 for(let j=0;j<4;j++){const o=fr*24,a=[c.pts[o+j*3]/1000,c.pts[o+j*3+1]/1000,c.pts[o+j*3+2]/1000],t=[c.pts[o+12+j*3]/1000,c.pts[o+12+j*3+1]/1000,c.pts[o+12+j*3+2]/1000];
  if(j>=2&&Math.hypot(t[0],t[1],t[2])<1e-6)continue;if(Math.hypot(a[0]-t[0],a[1]-t[1],a[2]-t[2])<0.02)continue;const pa=P(...a,W,H),pt=P(...t,W,H);if(!pa||!pt)continue;
  ctx.strokeStyle='#ff4d4d';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pt[0],pt[1]);ctx.stroke();ctx.beginPath();ctx.moveTo(pt[0]-6,pt[1]-6);ctx.lineTo(pt[0]+6,pt[1]+6);ctx.moveTo(pt[0]+6,pt[1]-6);ctx.lineTo(pt[0]-6,pt[1]+6);ctx.stroke()}
 if(c.bad[fr]){ctx.strokeStyle='#e05555';ctx.lineWidth=6;ctx.strokeRect(3,3,W-6,H-6)}
 // text
 const v=c.verdict,mg=c.margin[fr]/10;
 document.getElementById('info').innerHTML=`<b>${c.label}</b><br><span class="${v.feasible?'good':'bad'}">${v.feasible?'feasible':'NOT feasible'}</span> · score ${v.score.toFixed(4)} m/frame · ${v.bad_frames}/${c.frames} frames need correction · frame ${fr+1}/${c.frames} (${(fr*D.dt).toFixed(2)} s) · balance margin <span class="${mg<0?'bad':''}">${mg.toFixed(1)} cm</span>`;
 const on=v.issues.filter(t=>c.spans.some(s=>s.t===t&&fr>=s.a&&fr<=s.b)),key=ci+'|'+on.join('#');
 if(key!==lastKey){lastKey=key;const A=document.getElementById('act');A.innerHTML=on.map(t=>`<div class="iss on">▶ ${t}</div>`).join('');
  const all=document.getElementById('all');all.style.display=v.issues.length?'':'none';all.querySelector('summary').textContent=`all ${v.issues.length} issue${v.issues.length==1?'':'s'} in this clip`;
  document.getElementById('alli').innerHTML=v.issues.map(t=>`<div class="iss ${on.includes(t)?'on':'dim'}">${on.includes(t)?'▶ ':'· '}${t}</div>`).join('')}
 seek.value=fr}
let lastKey='';
function tick(now){const dt=(now-last)/1000;last=now;const c=D.clips[ci];if(playing){f+=dt*speed/D.dt;if(f>=c.frames)f=0}draw();requestAnimationFrame(tick)}
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'pause':'play'};
document.getElementById('speed').onchange=e=>speed=+e.target.value;document.getElementById('follow').onchange=e=>follow=e.target.checked;document.getElementById('ghost').onchange=e=>showGhost=e.target.checked;
sel.onchange=()=>setClip(+sel.value);seek.oninput=()=>{f=+seek.value;playing=false;document.getElementById('play').textContent='play'};
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{const v=VIEWS[+b.dataset.v];cam.yaw=v[0];cam.pitch=v[1];cam.dist=v[2]});
let drag=null;cv.onmousedown=e=>drag=[e.clientX,e.clientY];addEventListener('mouseup',()=>drag=null);addEventListener('mousemove',e=>{if(!drag)return;cam.yaw-=(e.clientX-drag[0])*0.008;cam.pitch=Math.max(-0.2,Math.min(1.5,cam.pitch+(e.clientY-drag[1])*0.006));drag=[e.clientX,e.clientY]});
cv.onwheel=e=>{cam.dist=Math.max(1.2,Math.min(12,cam.dist*(e.deltaY>0?1.1:0.9)));e.preventDefault()};
addEventListener('keydown',e=>{if(e.code==='Space'){document.getElementById('play').click();e.preventDefault()}else if(e.key==='ArrowRight'){playing=false;f=Math.min(D.clips[ci].frames-1,Math.floor(f)+1)}else if(e.key==='ArrowLeft'){playing=false;f=Math.max(0,Math.floor(f)-1)}else if(e.key===']')setClip(ci+1);else if(e.key==='[')setClip(ci-1);else if('12345'.includes(e.key)){const v=VIEWS[+e.key-1];cam.yaw=v[0];cam.pitch=v[1];cam.dist=v[2]}});
addEventListener('resize',drawStrip);setClip(0);requestAnimationFrame(tick);window.__v={set:(i,fr)=>{setClip(i);f=fr;playing=false},cam:cam};
</script></body></html>'''

if __name__ == "__main__":
    main()
