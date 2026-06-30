"use strict";
// neuroviz — animated 2D ERD topomap (inverse-distance interpolation) + waveforms for BCI IV-2a.
// No build step: serve this dir (`python -m http.server`) and open index.html.

// state.map ∈ {"mu","beta","csp0","csp1",…}; one unified selector instead of view+band+component toggles.
const state = { data:null, map:"mu", cls:null, frame:25, playing:false };
const $ = (id) => document.getElementById(id);
const isCsp = () => state.map.startsWith("csp");
const cspIdx = () => +state.map.slice(3);

// layout / render constants (no magic numbers inline)
const LAYOUT = {
  headRadiusFrac: 0.42,   // head-circle radius as a fraction of canvas width
  electrodeR: 2.2,        // electrode dot radius (px)
  electrodeRHot: 4,       // motor-trio electrode dot radius (px)
  idwEps: 1e-3,           // inverse-distance interpolation epsilon (avoid /0 at electrodes)
  wavePad: 34,            // waveform panel left padding (px, room for channel labels)
  waveAmp: 0.42,          // trace amplitude as a fraction of its row height
  frameMs: 90,            // animation frame interval (ms)
};

// diverging colormap (RdBu_r): t in [0,1] -> [r,g,b]
const STOPS = [[59,76,192],[170,184,255],[242,242,242],[255,176,160],[180,4,38]];
function cmap(t){
  t=Math.max(0,Math.min(1,t));
  const x=t*(STOPS.length-1), i=Math.floor(x), f=x-i;
  const a=STOPS[i], b=STOPS[Math.min(i+1,STOPS.length-1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

const nFrames = () => state.data.frame_times.length;

function currentValues(){
  const d=state.data;
  if (isCsp()) return d.csp_patterns[cspIdx()];
  return d.frames[state.map][state.cls][state.frame];
}
// stable color scale (CSP: the pattern; band: across the whole animation so playback is comparable)
function scaleMax(){
  const d=state.data;
  if (isCsp()) return Math.max(...d.csp_patterns[cspIdx()].map(Math.abs))||1;
  let m=1e-9;
  for (const fr of d.frames[state.map][state.cls]) for (const v of fr) m=Math.max(m,Math.abs(v));
  return m;
}

function renderTopo(){
  const d=state.data, cv=$("topo"), ctx=cv.getContext("2d");
  const W=cv.width,H=cv.height,cx=W/2,cy=H/2,R=W*LAYOUT.headRadiusFrac;
  const vals=currentValues(), m=scaleMax();
  const pos=d.pos.map(([x,y])=>[cx+x*R, cy-y*R]);
  ctx.clearRect(0,0,W,H);
  const img=ctx.createImageData(W,H), px=img.data;
  for(let yy=0;yy<H;yy++)for(let xx=0;xx<W;xx++){
    const dx=xx-cx,dy=yy-cy,o=(yy*W+xx)*4;
    if(dx*dx+dy*dy>R*R){px[o+3]=0;continue;}
    let num=0,den=0;
    for(let i=0;i<pos.length;i++){const ex=pos[i][0]-xx,ey=pos[i][1]-yy,w=1/(ex*ex+ey*ey+LAYOUT.idwEps);num+=w*vals[i];den+=w;}
    const c=cmap((num/den+m)/(2*m));
    px[o]=c[0];px[o+1]=c[1];px[o+2]=c[2];px[o+3]=235;
  }
  ctx.putImageData(img,0,0);
  ctx.strokeStyle="#7c879b";ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.stroke();
  ctx.beginPath();ctx.moveTo(cx-12,cy-R+3);ctx.lineTo(cx,cy-R-14);ctx.lineTo(cx+12,cy-R+3);ctx.stroke();
  ctx.beginPath();ctx.arc(cx-R,cy,10,1.6,4.7);ctx.stroke();
  ctx.beginPath();ctx.arc(cx+R,cy,10,-1.6,1.6);ctx.stroke();
  // electrode dots sized + outlined by their CONTRIBUTION to the current view (|value|/max), not hardcoded
  for(let i=0;i<pos.length;i++){
    const [ex,ey]=pos[i], c=Math.min(1,Math.abs(vals[i])/m);
    const rr=LAYOUT.electrodeR+(LAYOUT.electrodeRHot-LAYOUT.electrodeR)*c;
    ctx.beginPath();ctx.arc(ex,ey,rr,0,7);
    ctx.fillStyle="#0e1116";ctx.fill();
    ctx.strokeStyle=`rgba(230,233,239,${0.2+0.7*c})`;ctx.lineWidth=1;ctx.stroke();
  }
  $("hint").textContent = isCsp()
    ? `CSP component ${cspIdx()+1} — the spatial filter the baseline decoder learned (weight per electrode).`
    : `${state.map} ERD, ${state.cls.replace("_"," ")} — blue = motor cortex desynchronizing; switch class to see the active side move.`;
}

function renderWaves(){
  const d=state.data, cv=$("waves"), ctx=cv.getContext("2d");
  const W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
  const cls=state.cls||d.classes[0], wf=d.waveforms.trials[cls], chans=d.waveforms.chans, t=d.waveforms.t;
  // color each channel by its CONTRIBUTION to the current view (the same per-channel values the topomap uses)
  const vals=currentValues(), mm=scaleMax(), idxOf={};
  d.channels.forEach((n,i)=>idxOf[n]=i);
  const pad=LAYOUT.wavePad, rowH=H/chans.length, tc=d.frame_times[state.frame], cx=pad+(W-pad-6)*(tc/t[t.length-1]);
  ctx.font="10px system-ui"; ctx.textBaseline="middle";
  chans.forEach((ch,r)=>{
    const y0=r*rowH+rowH/2, trace=wf[ch], m=Math.max(...trace.map(Math.abs))||1;
    const vi=idxOf[ch], c=(vi==null)?0:Math.min(1,Math.abs(vals[vi])/mm);   // contribution 0..1
    const col=cmap((vi==null?0:vals[vi])/(2*mm)+0.5);                       // sign-aware color (matches topomap)
    const R=Math.round(70+(col[0]-70)*c), G=Math.round(78+(col[1]-78)*c), B=Math.round(95+(col[2]-95)*c);
    ctx.strokeStyle="#222937";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad,y0);ctx.lineTo(W-6,y0);ctx.stroke();
    ctx.strokeStyle=`rgb(${R},${G},${B})`;ctx.lineWidth=0.7+1.3*c;ctx.beginPath();
    for(let i=0;i<trace.length;i++){const x=pad+(W-pad-6)*i/(trace.length-1),y=y0-(trace[i]/m)*(rowH*LAYOUT.waveAmp);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
    ctx.stroke();
    ctx.fillStyle=`rgba(230,233,239,${0.4+0.5*c})`;ctx.fillText(ch,3,y0);
  });
  ctx.strokeStyle="#ff6a5a";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();
  ctx.fillStyle="#8b94a3";ctx.textBaseline="alphabetic";ctx.fillText(`${t[t.length-1].toFixed(1)} s`,W-34,H-4);
}

function fitCanvas(cv){
  const r=cv.getBoundingClientRect();
  const w=Math.max(120,Math.floor(r.width)), h=Math.max(120,Math.floor(r.height));
  if(cv.width!==w||cv.height!==h){ cv.width=w; cv.height=h; }
}

function render(){
  fitCanvas($("waves"));
  renderTopo(); renderWaves();
  $("scrub").value=state.frame;
  $("tlabel").textContent=`${state.data.frame_times[state.frame].toFixed(1)} s`;
}

let timer=null;
function play(on){
  state.playing=on; $("play").textContent=on?"❚❚":"▶";
  if(timer){clearInterval(timer);timer=null;}
  if(on) timer=setInterval(()=>{ state.frame=(state.frame+1)%nFrames(); render(); }, LAYOUT.frameMs);
}

function buildMapbar(){
  const bar=$("map"); bar.innerHTML="";
  const group=(label, items)=>{
    const g=document.createElement("div"); g.className="mapgroup";
    const l=document.createElement("span"); l.className="glabel"; l.textContent=label; g.appendChild(l);
    const seg=document.createElement("div"); seg.className="seg wrap";
    items.forEach(([k,t])=>{
      const b=document.createElement("button"); b.textContent=t; b.dataset.k=k;
      b.className=k===state.map?"on":""; b.onclick=()=>{ state.map=k; sync(); render(); };
      seg.appendChild(b);
    });
    g.appendChild(seg); bar.appendChild(g);
  };
  group("signal (band power)", [["mu","mu"],["beta","beta"]]);
  group("filters (CSP)", state.data.csp_patterns.map((_,i)=>["csp"+i,(i+1).toString()]));
}
function buildClassbar(){
  const bar=$("classbar"); bar.innerHTML="";
  state.data.classes.forEach(c=>{
    const b=document.createElement("button");
    b.textContent=c.replace("_"," "); b.dataset.c=c; b.className=c===state.cls?"on":"";
    b.onclick=()=>{ state.cls=c; sync(); render(); };
    bar.appendChild(b);
  });
}
function sync(){
  $("map").querySelectorAll("button").forEach(b=>b.classList.toggle("on", b.dataset.k===state.map));
  [...$("classbar").children].forEach(b=>b.classList.toggle("on", b.dataset.c===state.cls));
  const csp=isCsp();                          // class + time apply only to band-power maps
  $("classbar").hidden=csp;
  $("player").hidden=csp;
  if(csp) play(false);
}

async function loadSubject(s){
  state.data=await (await fetch(`data/subject${s}.json`)).json();
  state.cls=state.data.classes.includes("left_hand")?"left_hand":state.data.classes[0];
  state.map="mu"; state.frame=Math.floor(nFrames()/2);
  $("scrub").max=nFrames()-1;
  buildMapbar(); buildClassbar(); sync(); render();
}

async function init(){
  const man=await (await fetch("data/manifest.json")).json();
  const sel=$("subject");
  man.subjects.forEach(s=>{const o=document.createElement("option");o.value=s;o.textContent="subject "+s;sel.appendChild(o);});
  sel.onchange=()=>{play(false);loadSubject(sel.value);};
  $("play").onclick=()=>play(!state.playing);
  $("scrub").oninput=()=>{play(false);state.frame=+$("scrub").value;render();};
  let rz; window.addEventListener("resize",()=>{ clearTimeout(rz); rz=setTimeout(()=>{ if(state.data) render(); },120); });
  await loadSubject(man.subjects[0]);
}
init().catch(e=>{document.body.insertAdjacentHTML("beforeend",
  `<p style="color:#ffb0a0;padding:24px">load error: ${e}. Serve this dir: <code>python -m http.server</code> in neuroviz/web, then open http://localhost:8000</p>`);});
