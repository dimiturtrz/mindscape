"use strict";
// neuroviz — 2D topomap (inverse-distance interpolation) + waveforms for BCI IV-2a motor imagery.
// No build step: serve this dir (`python -m http.server`) and open index.html.

const state = { data: null, view: "bandpower", band: "mu", cls: null, comp: 0 };
const $ = (id) => document.getElementById(id);

// --- diverging colormap (RdBu_r): t in [0,1] -> [r,g,b] ---
const STOPS = [[59,76,192],[170,184,255],[242,242,242],[255,176,160],[180,4,38]];
function cmap(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (STOPS.length - 1), i = Math.floor(x), f = x - i;
  const a = STOPS[i], b = STOPS[Math.min(i + 1, STOPS.length - 1)];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

function currentValues() {
  const d = state.data;
  if (state.view === "csp") return d.csp_patterns[state.comp];
  return d.bandpower[state.band][state.cls];
}

function renderTopo() {
  const d = state.data, cv = $("topo"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, cx = W/2, cy = H/2, R = W*0.42;
  const vals = currentValues();
  const pos = d.pos.map(([x,y]) => [cx + x*R, cy - y*R]);   // +y = nose (up); canvas y flipped
  const m = Math.max(...vals.map(Math.abs)) || 1;

  ctx.clearRect(0,0,W,H);
  const img = ctx.createImageData(W,H), px = img.data;
  for (let yy=0; yy<H; yy++) for (let xx=0; xx<W; xx++) {
    const dx=xx-cx, dy=yy-cy;
    const o=(yy*W+xx)*4;
    if (dx*dx+dy*dy > R*R) { px[o+3]=0; continue; }          // outside scalp -> transparent
    let num=0, den=0;
    for (let i=0;i<pos.length;i++){
      const ex=pos[i][0]-xx, ey=pos[i][1]-yy;
      let w=1/(ex*ex+ey*ey+1e-3); num+=w*vals[i]; den+=w;
    }
    const t=(num/den + m)/(2*m);
    const c=cmap(t);
    px[o]=c[0]; px[o+1]=c[1]; px[o+2]=c[2]; px[o+3]=235;
  }
  ctx.putImageData(img,0,0);

  // head outline + nose + ears
  ctx.strokeStyle="#7c879b"; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(cx,cy,R,0,7); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx-12,cy-R+3); ctx.lineTo(cx,cy-R-14); ctx.lineTo(cx+12,cy-R+3); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx-R,cy,10,1.6,4.7); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx+R,cy,10,-1.6,1.6); ctx.stroke();

  // electrodes (highlight the motor trio)
  const key = new Set(["C3","Cz","C4"]);
  for (let i=0;i<pos.length;i++){
    const [ex,ey]=pos[i], name=d.channels[i], hot=key.has(name);
    ctx.beginPath(); ctx.arc(ex,ey,hot?4:2.2,0,7);
    ctx.fillStyle=hot?"#0e1116":"rgba(20,24,32,.6)"; ctx.fill();
    if(hot){ ctx.strokeStyle="#e6e9ef"; ctx.lineWidth=1.5; ctx.stroke();
      ctx.fillStyle="#e6e9ef"; ctx.font="11px system-ui"; ctx.fillText(name, ex+6, ey-6); }
  }
  $("hint").textContent = state.view==="csp"
    ? `CSP component ${state.comp+1} — the spatial filter the baseline decoder uses (expect weight over C3/C4).`
    : `${state.band} power, ${state.cls.replace("_"," ")} — watch the hot side flip C3↔C4 between left/right hand.`;
}

function renderWaves() {
  const d=state.data, cv=$("waves"), ctx=cv.getContext("2d");
  const W=cv.width, H=cv.height; ctx.clearRect(0,0,W,H);
  const cls = state.cls || d.classes[0];
  const wf = d.waveforms.trials[cls], chans = d.waveforms.chans, t=d.waveforms.t;
  const rowH=H/chans.length, pad=28;
  ctx.font="11px system-ui";
  chans.forEach((ch,r)=>{
    const y0=r*rowH+rowH/2, trace=wf[ch];
    const m=Math.max(...trace.map(Math.abs))||1;
    ctx.strokeStyle="#2a3140"; ctx.beginPath(); ctx.moveTo(pad,y0); ctx.lineTo(W-6,y0); ctx.stroke();
    ctx.strokeStyle="#5b8def"; ctx.lineWidth=1.2; ctx.beginPath();
    for(let i=0;i<trace.length;i++){
      const x=pad+(W-pad-6)*i/(trace.length-1), y=y0-(trace[i]/m)*(rowH*0.4);
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle="#e6e9ef"; ctx.fillText(ch, 4, y0-rowH*0.32);
  });
  ctx.fillStyle="#8b94a3"; ctx.fillText(`${t[t.length-1].toFixed(1)} s`, W-34, H-4);
}

function render(){ renderTopo(); renderWaves(); }

function buildClassbar(){
  const bar=$("classbar"); bar.innerHTML="";
  state.data.classes.forEach(c=>{
    const b=document.createElement("button");
    b.textContent=c.replace("_"," "); b.className=c===state.cls?"on":"";
    b.onclick=()=>{ state.cls=c; syncBars(); render(); };
    bar.appendChild(b);
  });
}
function buildCspbar(){
  const bar=$("cspbar"); bar.innerHTML="";
  state.data.csp_patterns.forEach((_,i)=>{
    const b=document.createElement("button");
    b.textContent="comp "+(i+1); b.className=i===state.comp?"on":"";
    b.onclick=()=>{ state.comp=i; syncBars(); render(); };
    bar.appendChild(b);
  });
}
function syncBars(){
  [...$("classbar").children].forEach(b=>b.className=b.textContent===state.cls.replace("_"," ")?"on":"");
  [...$("cspbar").children].forEach((b,i)=>b.className=i===state.comp?"on":"");
  $("classbar").hidden = state.view==="csp";
  $("band").style.opacity = state.view==="csp"?0.4:1;
  $("cspbar").hidden = state.view!=="csp";
}

async function loadSubject(s){
  state.data = await (await fetch(`data/subject${s}.json`)).json();
  state.cls = state.data.classes.includes("left_hand") ? "left_hand" : state.data.classes[0];
  state.comp = 0;
  buildClassbar(); buildCspbar(); syncBars(); render();
}

async function init(){
  const man = await (await fetch("data/manifest.json")).json();
  const sel=$("subject");
  man.subjects.forEach(s=>{ const o=document.createElement("option"); o.value=s; o.textContent="subject "+s; sel.appendChild(o); });
  sel.onchange=()=>loadSubject(sel.value);
  $("view").querySelectorAll("button").forEach(b=>b.onclick=()=>{
    state.view=b.dataset.view;
    $("view").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));
    syncBars(); render();
  });
  $("band").querySelectorAll("button").forEach(b=>b.onclick=()=>{
    state.band=b.dataset.band;
    $("band").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));
    render();
  });
  await loadSubject(man.subjects[0]);
}
init().catch(e=>{ document.body.insertAdjacentHTML("beforeend",
  `<p style="color:#ffb0a0;padding:24px">load error: ${e}. Serve this dir: <code>python -m http.server</code> in neuroviz/web, then open http://localhost:8000</p>`); });
