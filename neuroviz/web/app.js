"use strict";
// neuroviz — animated 2D ERD topomap (inverse-distance interpolation) + waveforms for BCI IV-2a.
// No build step: serve this dir (`python -m http.server`) and open index.html.

// modality-aware: EEG (mu/beta + CSP/Riemann) or fNIRS (HbO/HbR + LDA). state.map is a frame key
// ("mu"/"beta"/"HbO"/"HbR") or a decoder view ("csp0…", "riemann", "lda"). Controls are built from the
// data's own keys, so one viewer renders both modalities.
const state = { data:null, modality:null, map:null, cls:null, frame:25, playing:false, speed:1 };
const fmtSpeed = (s) => (s<1 ? s.toFixed(2) : s.toFixed(1)) + "×";
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
  playbackSpeed: 3,       // animation plays this many × faster than real time (EEG ~1.5 s, fNIRS ~7 s)
  minFrameMs: 20,         // floor so the fast EEG animation doesn't blur
};

// ideal ms per frame from the REAL signal duration / speed (NO floor — the floor is handled by advancing
// multiple frames per tick, so high speeds keep scaling instead of clamping).
function frameIntervalMs(){
  const ft=state.data.frame_times, n=ft.length;
  return (ft[n-1]-ft[0])/(n-1)*1000/state.speed;   // seconds/frame * 1000 / speed
}

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
    if(twoTrace){                                                   // both chromophores: red/blue hue, but
      const m=Math.max(...raw.hbo.map(Math.abs),...raw.hbr.map(Math.abs))||1;   // paler/sharper by CONTRIBUTION
      const vi=idxOf[ch], c=(vi==null)?0.4:Math.min(1,Math.abs(vals[vi])/mm);   // to the current view (like EEG)
      const a=0.22+0.78*c, w=0.6+1.3*c;
      line(raw.hbr,m,`rgba(91,157,255,${a})`,w);                   // HbR cool
      line(raw.hbo,m,`rgba(255,122,92,${a})`,w);                   // HbO warm (on top)
      ctx.fillStyle=`rgba(230,233,239,${0.35+0.5*c})`;ctx.fillText(ch,3,y0);
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

function fitTopo(){                               // square canvas filling its flex box (leftover panel height)
  const cv=$("topo"), box=cv.parentElement;
  const s=Math.max(1, Math.floor(Math.min(box.clientWidth, box.clientHeight)));
  if(cv.width!==s){ cv.width=s; cv.height=s; }
  cv.style.width=s+"px"; cv.style.height=s+"px";
}

function render(){
  fitTopo(); fitCanvas($("waves"));
  renderTopo(); renderWaves(); renderResult();
  $("scrub").value=state.frame;
  $("tlabel").textContent=`${state.data.frame_times[state.frame].toFixed(1)} s`;
}

let timer=null;
function play(on){
  state.playing=on; $("play").textContent=on?"❚❚":"▶";
  if(timer){clearInterval(timer);timer=null;}
  if(on){
    const ideal=frameIntervalMs();
    const step=Math.max(1, Math.ceil(LAYOUT.minFrameMs/ideal));   // sub-floor interval -> skip frames instead of clamping
    timer=setInterval(()=>{ state.frame=(state.frame+step)%nFrames(); render(); }, ideal*step);
  }
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
  $("speedbar").hidden=!animated;
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

// ---- fusion / complementarity view -----------------------------------------------------------------
// A different question from the single-modality views: on the SAME n-back blocks, where does EEG succeed,
// where does fNIRS, and do they miss the SAME blocks? Each cell = one block, colored by which modality got
// it right. Blue+orange scattered = complementarity (they fail independently); the oracle bar towers over
// either single modality, yet naive late fusion sits at the best single — the headroom no combiner cashes.
const FCAT = {
  both:  { c:"#3fb96b", label:"both correct" },
  eeg:   { c:"#5b9dff", label:"EEG only right" },
  fnirs: { c:"#ff7a5c", label:"fNIRS only right" },
  none:  { c:"#2b3342", label:"both wrong" },
};
const fcatOf = (b) => { const e=b.eeg===b.truth, f=b.fnirs===b.truth;
  return e&&f ? "both" : e ? "eeg" : f ? "fnirs" : "none"; };

function showFusion(on){
  document.querySelectorAll("main > .panel:not(.fusion-panel)").forEach(p=>p.hidden=on);
  $("fusionview").hidden=!on;
}

async function loadFusion(){
  const d = await (await fetch("data/fusion.json")).json();
  state.data=d; state.modality="fusion";
  $("sub").innerHTML = `<b>Fusion · EEG + fNIRS</b> (Shin n-back, same blocks). Two weak modalities that fail `+
    `on <b>different</b> blocks — a per-block map of the complementarity the numbers imply, and why averaging can't use it.`;
  showFusion(true); renderFusion();
}

function renderFusion(){
  const d=state.data; if(!d||d.modality!=="fusion" && state.modality!=="fusion") return;
  const s=d.summary, pct=(x)=> (100*x).toFixed(1)+"%";
  // group blocks by subject -> rows; each subject's blocks -> columns
  const bySub={}; d.blocks.forEach(b=>{ (bySub[b.subject]=bySub[b.subject]||[]).push(b); });
  const subs=Object.keys(bySub).sort((a,b)=>(+a)-(+b));
  const cols=Math.max(...subs.map(k=>bySub[k].length));

  const cv=$("fgrid"), ctx=cv.getContext("2d");
  // fit width to the column, but cap the height so the grid can't overrun the panel/footer (square cells)
  const box=cv.parentElement, MAXH=400;
  let W=Math.max(160,Math.floor(Math.min(box.clientWidth, 560))), H=Math.floor(W*subs.length/cols);
  if(H>MAXH){ H=MAXH; W=Math.floor(H*cols/subs.length); }
  cv.width=W; cv.height=H; cv.style.width=W+"px"; cv.style.height=H+"px";   // display = buffer (no CSS rescale)
  ctx.clearRect(0,0,cv.width,cv.height);
  const cw=cv.width/cols, ch=cv.height/subs.length, g=Math.max(0.5, cw*0.08);
  subs.forEach((sub,r)=>{ bySub[sub].forEach((b,c)=>{
    ctx.fillStyle=FCAT[fcatOf(b)].c;
    ctx.fillRect(c*cw+g/2, r*ch+g/2, cw-g, ch-g);
  });});

  // headline
  const best=Math.max(s.eeg,s.fnirs);
  $("fresult").innerHTML = `oracle (either modality right) <b class="ok">${pct(s.oracle)}</b> `+
    `— <b>+${(100*(s.oracle-best)).toFixed(0)} pts</b> over the best single (${pct(best)}); `+
    `late fusion <b class="no">${s.late!=null?pct(s.late):"—"}</b> captures none of it. `+
    `<span class="rmut">error corr φ=${s.err_corr.toFixed(2)} · both wrong only ${pct(s.both_wrong)}</span>`;

  // legend with counts
  $("flegend").innerHTML = ["both","eeg","fnirs","none"].map(k=>{
    const frac = k==="both"?s.both_correct : k==="eeg"?s.eeg_only : k==="fnirs"?s.fnirs_only : s.both_wrong;
    return `<span class="fkey"><i style="background:${FCAT[k].c}"></i>${FCAT[k].label} <b>${pct(frac)}</b></span>`;
  }).join("");

  // bars: EEG / fNIRS / late / oracle, with chance + the oracle ceiling marked
  const rows=[["EEG (Riemann)",s.eeg,"#5b9dff"],["fNIRS (LDA)",s.fnirs,"#ff7a5c"],
              ["late fusion",s.late,"#9aa4b2"],["oracle (ceiling)",s.oracle,"#3fb96b"]];
  const maxv=Math.max(s.oracle,0.7);
  $("fbars").innerHTML = `<div class="fbtitle">accuracy (chance ${pct(s.chance)}, ${s.n} blocks)</div>` +
    rows.map(([n,v,c])=> v==null?"" :
      `<div class="fbar"><span class="fbn">${n}</span>`+
      `<span class="fbtrack"><i style="width:${100*v/maxv}%;background:${c}"></i>`+
      `<u style="left:${100*s.chance/maxv}%"></u></span>`+
      `<span class="fbv">${pct(v)}</span></div>`).join("");

  $("fhint").innerHTML = `Each cell is one held-out block (rows = subjects, 5-fold GroupKFold). Blue + orange `+
    `are blocks only one modality gets — they're scattered and roughly balanced (φ≈0), so the modalities are `+
    `genuinely complementary. Yet <b>late fusion ≈ the best single</b>: averaging probabilities can't tell which `+
    `modality to trust per block (confidence doesn't track correctness), so the <b>+${(100*(s.oracle-best)).toFixed(0)}-pt</b> `+
    `oracle headroom stays on the table. That's the honest fusion result.`;
}

async function init(){
  const man=await (await fetch("data/manifest.json")).json();
  const mods=man.modalities||{eeg:man.subjects||[]};      // back-compat with the old flat manifest
  const modBar=$("modality"), subjSel=$("subject");

  function loadModality(mod){
    [...modBar.children].forEach(b=>b.classList.toggle("on", b.dataset.m===mod));
    play(false);
    if(mod==="fusion"){ subjSel.parentElement.hidden=true; loadFusion(); return; }
    showFusion(false); subjSel.parentElement.hidden=false;
    subjSel.innerHTML="";
    mods[mod].forEach(s=>{const o=document.createElement("option");o.value=s;o.textContent="subject "+s;subjSel.appendChild(o);});
    loadSubject(mod, mods[mod][0]);
  }
  const MOD_LABEL={eeg:"EEG",fnirs:"fNIRS"};
  Object.keys(mods).filter(m=>mods[m] && mods[m].length).forEach(mod=>{
    const b=document.createElement("button"); b.textContent=MOD_LABEL[mod]||mod.toUpperCase(); b.dataset.m=mod;
    b.onclick=()=>loadModality(mod); modBar.appendChild(b);
  });
  if(man.fusion){ const b=document.createElement("button"); b.textContent="Fusion"; b.dataset.m="fusion";
    b.onclick=()=>loadModality("fusion"); modBar.appendChild(b); }
  subjSel.onchange=()=>{play(false);loadSubject(state.modality, subjSel.value);};
  $("play").onclick=()=>play(!state.playing);
  $("scrub").oninput=()=>{play(false);state.frame=+$("scrub").value;render();};
  $("speed").oninput=()=>{ state.speed=Math.pow(10, +$("speed").value);   // log slider -> speed
                          $("speedlabel").textContent=fmtSpeed(state.speed);
                          if(state.playing) play(true); };   // restart timer with the new interval
  let rz; window.addEventListener("resize",()=>{ clearTimeout(rz); rz=setTimeout(()=>{
    if(!state.data) return; state.modality==="fusion" ? renderFusion() : render(); },120); });
  // deep-link: #fusion / #eeg / #fnirs selects the initial view (also lets a headless render target it)
  const want=(location.hash||"").slice(1);
  const first=Object.keys(mods).find(m=>mods[m] && mods[m].length);
  loadModality(want==="fusion" && man.fusion ? "fusion" : (mods[want] && mods[want].length ? want : first));
}
init().catch(e=>{document.body.insertAdjacentHTML("beforeend",
  `<p style="color:#ffb0a0;padding:24px">load error: ${e}. Serve this dir: <code>python -m http.server</code> in neuroviz/web, then open http://localhost:8000</p>`);});
