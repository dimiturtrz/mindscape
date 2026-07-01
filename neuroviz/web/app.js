"use strict";
// neuroviz — animated 2D ERD topomap (inverse-distance interpolation) + waveforms for BCI IV-2a.
// No build step: serve this dir (`python -m http.server`) and open index.html.

// modality-aware: EEG (mu/beta + CSP/Riemann) or fNIRS (HbO/HbR + LDA). state.map is a frame key
// ("mu"/"beta"/"HbO"/"HbR") or a decoder view ("csp0…", "riemann", "lda"). Controls are built from the
// data's own keys, so one viewer renders both modalities.
const state = { data:null, modality:null, map:null, cls:null, frame:25, playing:false };
const $ = (id) => document.getElementById(id);
const isCsp = () => state.map.startsWith("csp");
const isRiemann = () => state.map === "riemann";
const isLda = () => state.map === "lda";
const isSignal = () => state.data && (state.map in state.data.frames);   // a time-resolved frame map
const isPerClass = () => !isCsp();          // signal + Riemann + LDA are per-class; CSP filters are not
const cspIdx = () => +state.map.slice(3);
const firstSignal = () => Object.keys(state.data.frames)[0];
const family = () => isCsp() ? "csp" : isRiemann() ? "riemann" : isLda() ? "lda" : "signal";

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
  if (isRiemann()) return d.riemann_patterns[state.cls];
  if (isLda()) return d.lda_patterns[state.cls];
  return d.frames[state.map][state.cls][state.frame];
}
// stable color scale (decoder views: the static pattern; signal: across the whole animation so playback is comparable)
function scaleMax(){
  const d=state.data;
  if (isCsp()) return Math.max(...d.csp_patterns[cspIdx()].map(Math.abs))||1;
  if (isRiemann()) return Math.max(...d.riemann_patterns[state.cls].map(Math.abs))||1;
  if (isLda()) return Math.max(...d.lda_patterns[state.cls].map(Math.abs))||1;
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
  const cls=(state.cls||"").replace("_"," ");
  $("hint").textContent = isCsp()
    ? `CSP component ${cspIdx()+1} — the spatial filter the baseline decoder learned (weight per electrode).`
    : isRiemann()
    ? `Riemann discriminant, ${cls} — per-channel weight of the tangent-space classifier (covariance is the feature; no spatial filter). Switch class to see the pattern move.`
    : isLda()
    ? `LDA workload discriminant, ${cls} — per-channel weight of the amplitude-feature decoder (mean HbO). Switch class to compare load levels.`
    : state.modality==="fnirs"
    ? `Hemodynamic response (HbO), ${cls} — building over the trial (red = concentration rise); scrub time to watch it peak ~5–8 s. Both chromophores are in the waveforms →`
    : `${state.map} ERD, ${cls} — blue = motor cortex desynchronizing; switch class to see the active side move.`;
}

function renderWaves(){
  const d=state.data, cv=$("waves"), ctx=cv.getContext("2d");
  const W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
  const cls=state.cls||d.classes[0], wf=d.waveforms.trials[cls], chans=d.waveforms.chans, t=d.waveforms.t;
  // color each channel by its CONTRIBUTION to the current view (the same per-channel values the topomap uses)
  const vals=currentValues(), mm=scaleMax(), idxOf={};
  d.channels.forEach((n,i)=>idxOf[n]=i);
  const twoTrace = wf[chans[0]] && !Array.isArray(wf[chans[0]]);    // fNIRS: {hbo,hbr} per channel
  const top = twoTrace ? 16 : 0;                                    // gutter so the HbO/HbR legend clears row 0
  const pad=LAYOUT.wavePad, rowH=(H-top)/chans.length, tc=d.frame_times[state.frame], cx=pad+(W-pad-6)*(tc/t[t.length-1]);
  ctx.font="10px system-ui"; ctx.textBaseline="middle";
  chans.forEach((ch,r)=>{
    const y0=top+r*rowH+rowH/2, raw=wf[ch];
    const line=(trace,m,col,w)=>{ ctx.strokeStyle=col;ctx.lineWidth=w;ctx.beginPath();
      for(let i=0;i<trace.length;i++){const x=pad+(W-pad-6)*i/(trace.length-1),y=y0-(trace[i]/m)*(rowH*LAYOUT.waveAmp);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
      ctx.stroke(); };
    ctx.strokeStyle="#222937";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad,y0);ctx.lineTo(W-6,y0);ctx.stroke();
    if(twoTrace){                                                   // both chromophores, one scale, fixed colors
      const m=Math.max(...raw.hbo.map(Math.abs),...raw.hbr.map(Math.abs))||1;
      line(raw.hbr,m,"#5b9dff",1.0);                               // HbR cool
      line(raw.hbo,m,"#ff7a5c",1.2);                               // HbO warm (on top)
      ctx.fillStyle="rgba(230,233,239,0.55)";ctx.fillText(ch,3,y0);
    } else {                                                        // EEG single trace, colored by contribution
      const m=Math.max(...raw.map(Math.abs))||1;
      const vi=idxOf[ch], c=(vi==null)?0:Math.min(1,Math.abs(vals[vi])/mm);
      const col=cmap((vi==null?0:vals[vi])/(2*mm)+0.5);
      line(raw, m, `rgb(${Math.round(70+(col[0]-70)*c)},${Math.round(78+(col[1]-78)*c)},${Math.round(95+(col[2]-95)*c)})`, 0.7+1.3*c);
      ctx.fillStyle=`rgba(230,233,239,${0.4+0.5*c})`;ctx.fillText(ch,3,y0);
    }
  });
  if(twoTrace){ ctx.fillStyle="#ff7a5c";ctx.fillText("HbO",pad,7); ctx.fillStyle="#5b9dff";ctx.fillText("HbR",pad+30,7); }
  ctx.strokeStyle="#ff6a5a";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();
  ctx.fillStyle="#8b94a3";ctx.textBaseline="alphabetic";ctx.fillText(`${t[t.length-1].toFixed(1)} s`,W-34,H-4);
}

function fitCanvas(cv){
  const r=cv.getBoundingClientRect();
  const w=Math.max(120,Math.floor(r.width)), h=Math.max(120,Math.floor(r.height));
  if(cv.width!==w||cv.height!==h){ cv.width=w; cv.height=h; }
}

// the honest output: ground truth of the shown example trial vs the decoder's prediction + the LOSO score
function renderResult(){
  const d=state.data, el=$("result");
  if(!d.predictions || !d.score){ el.textContent=""; return; }
  const s=d.score, p=d.predictions[state.cls];
  const head=`${s.decoder} · ${s.regime} acc <b>${s.acc}</b> (chance ${s.chance})`;
  if(!p){ el.innerHTML=head; return; }
  const nm=(x)=>x.replace("_"," ");
  el.innerHTML=`<span class="rlabel">example ${nm(state.cls)} trial</span> — truth <b>${nm(p.truth)}</b> · `+
    `predicted <b class="${p.correct?'ok':'no'}">${nm(p.pred)}</b> ${p.correct?'✓':'✗'}`+
    `<span class="rmut">  ·  ${head}</span>`;
}

function render(){
  fitCanvas($("waves"));
  renderTopo(); renderWaves(); renderResult();
  $("scrub").value=state.frame;
  $("tlabel").textContent=`${state.data.frame_times[state.frame].toFixed(1)} s`;
}

let timer=null;
function play(on){
  state.playing=on; $("play").textContent=on?"❚❚":"▶";
  if(timer){clearInterval(timer);timer=null;}
  if(on) timer=setInterval(()=>{ state.frame=(state.frame+1)%nFrames(); render(); }, LAYOUT.frameMs);
}

// TWO separate axes: `view` = the SIGNAL being looked at (the raw response — mu/beta or the HbO response);
// `method` = the DECODER (CSP/Riemann/LDA), whose learned pattern the topomap can show. They are not the
// same thing, so they get their own dropdowns.
function buildView(){
  const d=state.data, sel=$("view"); sel.innerHTML="";
  Object.keys(d.frames).forEach(k=>{ const o=document.createElement("option"); o.value=k; o.textContent=k; sel.appendChild(o); });
  sel.onchange=()=>{ state.map=sel.value; buildSubmaps(); sync(); render(); };   // -> a signal topomap
}
function buildMethod(){
  const d=state.data, sel=$("method"); sel.innerHTML="";
  const opts=[];
  if(d.csp_patterns) opts.push(["csp","CSP"]);
  if(d.riemann_patterns) opts.push(["riemann","Riemann"]);
  if(d.lda_patterns) opts.push(["lda","LDA"]);
  opts.forEach(([k,t])=>{ const o=document.createElement("option"); o.value=k; o.textContent=t; sel.appendChild(o); });
  sel.onchange=()=>{ state.map = sel.value==="csp" ? "csp0" : sel.value; buildSubmaps(); sync(); render(); };  // -> the decoder's pattern
}
// only CSP needs a sub-control (which of its 6 filters); signal maps + Riemann/LDA have none.
function buildSubmaps(){
  const bar=$("map"); bar.innerHTML=""; const d=state.data;
  if(family()==="csp"){
    const fg=document.createElement("div"); fg.className="mapgroup";
    const fl=document.createElement("span"); fl.className="glabel"; fl.textContent="filter"; fg.appendChild(fl);
    const sel=document.createElement("select"); sel.id="cspsel";
    d.csp_patterns.forEach((_,i)=>{ const o=document.createElement("option"); o.value="csp"+i; o.textContent="CSP "+(i+1); sel.appendChild(o); });
    sel.value=state.map;
    sel.onchange=()=>{ state.map=sel.value; sync(); render(); };
    fg.appendChild(sel); bar.appendChild(fg);
  }
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
  // the active axis updates its own dropdown; the other keeps its last value
  if(isSignal()) $("view").value=state.map;
  else $("method").value = isCsp() ? "csp" : state.map;
  const sel=$("cspsel"); if(sel) sel.value=state.map;
  [...$("classbar").children].forEach(b=>b.classList.toggle("on", b.dataset.c===state.cls));
  $("classbar").hidden=!isPerClass();         // class applies to signal + Riemann + LDA (per-class), not CSP
  const animated=isSignal();                  // only the signal maps have time frames
  $("player").hidden=!animated;
  if(!animated) play(false);
}

// modality-aware copy — the descriptions swap with the data and describe what's actually on screen.
const TEXTS = {
  eeg: {
    sub: `<b>EEG · motor imagery</b> (BCI IV-2a). Imagining one hand <b>desynchronizes the opposite motor cortex</b> — mu/beta power drops over C3↔C4. A fast electrical signal; the decoder reads the covariance.`,
    head: `waveforms — all 22 EEG channels <span class="mut">(colored by contribution to the current view)</span>`,
    hint: `One example trial; each channel's brightness + width = how much it drives the selected view. The red cursor tracks the topomap frame.`,
  },
  fnirs: {
    sub: `<b>fNIRS · mental workload</b> (Shin n-back). Prefrontal blood oxygen tracks cognitive load — <b>HbO rises, HbR falls</b>, peaking ~5–8 s. A slow hemodynamic signal; the decoder reads the amplitude.`,
    head: `waveforms — HbO + HbR per optode <span class="mut">(the raw two-signal data)</span>`,
    hint: `One example trial: <b style="color:#ff7a5c">HbO</b> (warm) rises while <b style="color:#5b9dff">HbR</b> (cool) falls — the anti-correlated hemodynamic response. The red cursor tracks the topomap frame.`,
  },
};
function setTexts(modality){
  const tx=TEXTS[modality]||TEXTS.eeg;
  $("sub").innerHTML=tx.sub; $("wavehead").innerHTML=tx.head; $("wavehint").innerHTML=tx.hint;
}

async function loadSubject(modality, s){
  const prefix = modality==="fnirs" ? "fnirs_" : "";
  state.data=await (await fetch(`data/${prefix}subject${s}.json`)).json();
  state.data.modality=modality; state.modality=modality;
  setTexts(modality);
  state.cls=state.data.classes.includes("left_hand")?"left_hand":state.data.classes[0];
  state.map=firstSignal(); state.frame=Math.floor(nFrames()/2);
  $("scrub").max=nFrames()-1;
  buildView(); buildMethod(); buildSubmaps(); buildClassbar(); sync(); render();
}

async function init(){
  const man=await (await fetch("data/manifest.json")).json();
  const mods=man.modalities||{eeg:man.subjects||[]};      // back-compat with the old flat manifest
  const modBar=$("modality"), subjSel=$("subject");

  function loadModality(mod){
    subjSel.innerHTML="";
    mods[mod].forEach(s=>{const o=document.createElement("option");o.value=s;o.textContent="subject "+s;subjSel.appendChild(o);});
    [...modBar.children].forEach(b=>b.classList.toggle("on", b.dataset.m===mod));
    play(false); loadSubject(mod, mods[mod][0]);
  }
  const MOD_LABEL={eeg:"EEG",fnirs:"fNIRS"};
  Object.keys(mods).filter(m=>mods[m] && mods[m].length).forEach(mod=>{
    const b=document.createElement("button"); b.textContent=MOD_LABEL[mod]||mod.toUpperCase(); b.dataset.m=mod;
    b.onclick=()=>loadModality(mod); modBar.appendChild(b);
  });
  subjSel.onchange=()=>{play(false);loadSubject(state.modality, subjSel.value);};
  $("play").onclick=()=>play(!state.playing);
  $("scrub").oninput=()=>{play(false);state.frame=+$("scrub").value;render();};
  let rz; window.addEventListener("resize",()=>{ clearTimeout(rz); rz=setTimeout(()=>{ if(state.data) render(); },120); });
  loadModality(Object.keys(mods).find(m=>mods[m] && mods[m].length));
}
init().catch(e=>{document.body.insertAdjacentHTML("beforeend",
  `<p style="color:#ffb0a0;padding:24px">load error: ${e}. Serve this dir: <code>python -m http.server</code> in neuroviz/web, then open http://localhost:8000</p>`);});
