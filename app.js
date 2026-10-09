'use strict';
// ===== INSIGHTX - browser edition (all AI runs on YOUR device, camera = your device) =====
const $ = id => document.getElementById(id);
const VEHICLES = new Set(['car','motorcycle','bus','truck']);
const ANIMALS = new Set(['dog','cat','horse','sheep','cow','elephant','bear','zebra','giraffe','bird']);
const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];
const COLOR_HEX = {Red:'#ff3333',Orange:'#ff9900',Yellow:'#ffff00',Green:'#00cc44',Cyan:'#00ffff',Blue:'#3388ff',Purple:'#9933ff',Pink:'#ff66b2',Brown:'#8b4513',Black:'#222222',White:'#eeeeee',Gray:'#888888'};
const CONF_MIN = 0.35, MATCH_MIN = 0.363;
const TEMPLATE = [[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]];

// ---------- audio ----------
let AC = null, voiceOn = true, lastSpoke = 0;
function beep(f,d,vol){ try{ AC = AC || new (window.AudioContext||window.webkitAudioContext)(); const o=AC.createOscillator(),g=AC.createGain(); o.connect(g); g.connect(AC.destination); o.frequency.value=f; g.gain.value=vol||0.2; o.start(); o.stop(AC.currentTime+(d||0.1)); }catch(e){} }
function speak(t, force){ if(!voiceOn) return; const n=Date.now(); if(!force && n-lastSpoke<3000) return; lastSpoke=n; try{ if(window.NativeTTS){ NativeTTS.speak(t); return; } const u=new SpeechSynthesisUtterance(t); u.rate=1.1; u.pitch=0.9; speechSynthesis.cancel(); speechSynthesis.speak(u);}catch(e){} }
function IC(n){return '<svg class="ic"><use href="#i-'+n+'"/></svg>';}
function toggleVoice(){ voiceOn=!voiceOn; $('voiceBtn').innerHTML=IC(voiceOn?'vol':'mute')+' VOICE: '+(voiceOn?'ON':'OFF'); if(!voiceOn) try{speechSynthesis.cancel()}catch(e){} }

// ---------- models ----------
let yolo=null, yunet=null, sface=null, modelsReady=false;
async function loadBuf(url, label){
  $('camMsgT').textContent = 'Loading '+label+'...';
  const r = await fetch(url); if(!r.ok) throw new Error(url+' '+r.status);
  return new Uint8Array(await r.arrayBuffer());
}
async function loadModels(){
  try{
    ort.env.logLevel='error'; ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
    ort.env.wasm.numThreads = (self.crossOriginIsolated ? Math.min(3, navigator.hardwareConcurrency||2) : 1); ort.env.wasm.proxy = true; /* inference in a worker keeps video/UI smooth */
    const opt = {executionProviders:['wasm'], graphOptimizationLevel:'all'};
    yunet = await ort.InferenceSession.create(await loadBuf('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx','face detector'), opt);
    yolo  = await ort.InferenceSession.create(await loadBuf('https://huggingface.co/deepghs/yolos/resolve/main/yolov8n/model.onnx','object detector'), opt);
    sface = await ort.InferenceSession.create(await loadBuf('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx','face recognizer (38 MB, first time only)'), opt);
    await loadFaces();
    modelsReady = true;
    $('pwr-mode').textContent='[BROWSER AI + FACE ID: ON]'; $('pwr-mode').style.color='#00ffa3';
    $('camMsgT').innerHTML = 'AI ready<br>Tap <b>START CAMERA</b> and allow camera permission';
    const b = document.createElement('button'); b.className='upb'; b.style.cssText='max-width:260px;background:#00ffa3;color:#000'; b.innerHTML=IC('play')+' START CAMERA'; b.onclick=startCam; $('camMsg').appendChild(b); b.id='bigStart';
  }catch(e){ $('camMsgT').textContent='Error loading AI: '+e.message; $('pwr-mode').textContent='[AI ERROR]'; $('pwr-mode').style.color='#ff3333'; console.error(e); }
}

// ---------- image prep ----------
const lb = document.createElement('canvas'); lb.width=lb.height=640;
const lbx = lb.getContext('2d',{willReadFrequently:true});
function letterbox(src, w, h){ // top-left letterbox into 640x640, returns scale and pixel data
  const s = Math.min(640/w, 640/h);
  lbx.fillStyle='#727272'; lbx.fillRect(0,0,640,640);
  lbx.drawImage(src,0,0,w,h,0,0,Math.round(w*s),Math.round(h*s));
  return {s, data: lbx.getImageData(0,0,640,640).data};
}
function toRGB01(d){ const n=640*640, f=new Float32Array(3*n); for(let i=0,j=0;i<n;i++,j+=4){ f[i]=d[j]/255; f[n+i]=d[j+1]/255; f[2*n+i]=d[j+2]/255; } return f; }
function toBGR255(d){ const n=640*640, f=new Float32Array(3*n); for(let i=0,j=0;i<n;i++,j+=4){ f[i]=d[j+2]; f[n+i]=d[j+1]; f[2*n+i]=d[j]; } return f; }

// ---------- YOLO ----------
function iou(a,b){ const x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]),x2=Math.min(a[2],b[2]),y2=Math.min(a[3],b[3]); const i=Math.max(0,x2-x1)*Math.max(0,y2-y1); const u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i; return u>0?i/u:0; }
function nms(items, thr){ items.sort((a,b)=>b.score-a.score); const keep=[]; for(const it of items){ let ok=true; for(const k of keep){ if(k.cls!==undefined && it.cls!==undefined && k.cls!==it.cls) continue; if(iou(k.box,it.box)>thr){ok=false;break;} } if(ok) keep.push(it);} return keep; }
const YS=(new URLSearchParams(location.search).get('hi')?320:256); const yc=document.createElement('canvas'); yc.width=yc.height=YS; const ycx=yc.getContext('2d',{willReadFrequently:true});
async function runYolo(src, w, h){
  const s=Math.min(YS/w,YS/h); ycx.fillStyle='#727272'; ycx.fillRect(0,0,YS,YS); ycx.drawImage(src,0,0,w,h,0,0,Math.round(w*s),Math.round(h*s));
  const d=ycx.getImageData(0,0,YS,YS).data, n=YS*YS, f=new Float32Array(3*n);
  for(let i=0,j=0;i<n;i++,j+=4){ f[i]=d[j]/255; f[n+i]=d[j+1]/255; f[2*n+i]=d[j+2]/255; }
  const out = await yolo.run({images:new ort.Tensor('float32', f, [1,3,YS,YS])});
  const ot = out[Object.keys(out)[0]], o=ot.data, N=ot.dims[2], res=[];
  for(let i=0;i<N;i++){
    let best=0, bc=-1;
    for(let c=0;c<80;c++){ const v=o[(4+c)*N+i]; if(v>best){best=v;bc=c;} }
    if(best<CONF_MIN) continue;
    const cx=o[i],cy=o[N+i],w=o[2*N+i],h=o[3*N+i];
    res.push({box:[(cx-w/2)/s,(cy-h/2)/s,(cx+w/2)/s,(cy+h/2)/s], score:best, cls:bc});
  }
  return nms(res,0.7);
}
// ---------- YuNet ----------
async function runYunet(data, s){
  const out = await yunet.run({input:new ort.Tensor('float32', toBGR255(data), [1,3,640,640])});
  const faces=[];
  for(const st of [8,16,32]){
    const cls=out['cls_'+st].data, obj=out['obj_'+st].data, bb=out['bbox_'+st].data, kp=out['kps_'+st].data, cols=640/st, n=cols*cols;
    for(let i=0;i<n;i++){
      const c=Math.min(1,Math.max(0,cls[i])), ob=Math.min(1,Math.max(0,obj[i])), sc=Math.sqrt(c*ob);
      if(sc<0.8) continue;
      const r=Math.floor(i/cols), cc=i%cols;
      const cx=(cc+bb[i*4])*st, cy=(r+bb[i*4+1])*st, w=Math.exp(bb[i*4+2])*st, h=Math.exp(bb[i*4+3])*st;
      const lm=[]; for(let k=0;k<5;k++) lm.push([(kp[i*10+2*k]+cc)*st/s,(kp[i*10+2*k+1]+r)*st/s]);
      faces.push({box:[(cx-w/2)/s,(cy-h/2)/s,(cx+w/2)/s,(cy+h/2)/s], score:sc, lm});
    }
  }
  return nms(faces,0.3);
}
// ---------- SFace ----------
const ac = document.createElement('canvas'); ac.width=ac.height=112; const acx=ac.getContext('2d',{willReadFrequently:true});
function simTransform(src, dst){ // least squares similarity: x'=a x - b y + tx ; y'=b x + a y + ty
  const n=src.length; let sx=0,sy=0,dx=0,dy=0; for(let i=0;i<n;i++){sx+=src[i][0];sy+=src[i][1];dx+=dst[i][0];dy+=dst[i][1];} sx/=n;sy/=n;dx/=n;dy/=n;
  let num1=0,num2=0,den=0; for(let i=0;i<n;i++){ const x=src[i][0]-sx,y=src[i][1]-sy,u=dst[i][0]-dx,v=dst[i][1]-dy; num1+=x*u+y*v; num2+=x*v-y*u; den+=x*x+y*y; }
  const a=num1/den, b=num2/den; return {a,b,tx:dx-(a*sx-b*sy), ty:dy-(b*sx+a*sy)};
}
async function embed(srcCanvas, lm){
  const t=simTransform(lm, TEMPLATE);
  acx.setTransform(1,0,0,1,0,0); acx.clearRect(0,0,112,112);
  acx.setTransform(t.a,t.b,-t.b,t.a,t.tx,t.ty); acx.imageSmoothingQuality='high';
  acx.drawImage(srcCanvas,0,0); acx.setTransform(1,0,0,1,0,0);
  const d=acx.getImageData(0,0,112,112).data, n=112*112, f=new Float32Array(3*n);
  for(let i=0,j=0;i<n;i++,j+=4){ f[i]=d[j]; f[n+i]=d[j+1]; f[2*n+i]=d[j+2]; } // RGB, 0-255
  const out = await sface.run({data:new ort.Tensor('float32',f,[1,3,112,112])});
  return normalize(out[Object.keys(out)[0]].data);
}
function normalize(v){ let s=0; for(const x of v) s+=x*x; s=Math.sqrt(s)||1; const o=new Float32Array(v.length); for(let i=0;i<v.length;i++) o[i]=v[i]/s; return o; }
function dot(a,b){ let s=0; for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }

// ---------- known faces ----------
let known = []; // {name, emb, thumb, builtin}
const LS_KEY='sv_faces_v1', LS_DEL='sv_deleted_builtin_v1';
async function loadFaces(){
  const del = JSON.parse(localStorage.getItem(LS_DEL)||'[]');
  try{ const d = await (await fetch('default_faces.json')).json(); for(const k in d) if(!del.includes(k)) known.push({name:k, emb:normalize(Float32Array.from(d[k])), thumb:null, builtin:true}); }catch(e){}
  try{ for(const u of JSON.parse(localStorage.getItem(LS_KEY)||'[]')) known.push({name:u.name, emb:normalize(Float32Array.from(u.emb)), thumb:u.thumb, builtin:false}); }catch(e){}
  renderRefs();
}
function saveFaces(){ localStorage.setItem(LS_KEY, JSON.stringify(known.filter(k=>!k.builtin).map(k=>({name:k.name,emb:Array.from(k.emb),thumb:k.thumb})))); }
function triggerUp(nameId, fileId){ const n=$(nameId).value.trim(); if(!n){ speak('Please enter the name first.',true); alert('Enter name first!'); return; } $(fileId).click(); }
async function doUp(nameId, fileId){
  if(!modelsReady){ alert('AI is still loading, wait a few seconds'); return; }
  const name=$(nameId).value.trim().replace(/[^A-Za-z0-9 ]/g,'').trim().toUpperCase(); if(!name) return;
  const file=$(fileId).files[0]; $(fileId).value=''; if(!file) return;
  const img = await new Promise((res,rej)=>{ const i=new Image(); i.onload=()=>res(i); i.onerror=rej; i.src=URL.createObjectURL(file); });
  const sc=Math.min(1,1280/Math.max(img.naturalWidth,img.naturalHeight)), w=Math.round(img.naturalWidth*sc), h=Math.round(img.naturalHeight*sc);
  const c=document.createElement('canvas'); c.width=w; c.height=h; c.getContext('2d').drawImage(img,0,0,w,h);
  const L=letterbox(c,w,h); const faces=await runYunet(L.data,L.s);
  if(!faces.length){ speak('No face found in that photo.',true); alert('No clear face found in this photo. Try a front-facing, well lit photo.'); return; }
  const emb=await embed(c, faces[0].lm);
  const t=document.createElement('canvas'); t.width=t.height=120; const fb=faces[0].box, fw=fb[2]-fb[0], fh=fb[3]-fb[1], m=Math.max(fw,fh)*0.7;
  t.getContext('2d').drawImage(c,(fb[0]+fb[2])/2-m,(fb[1]+fb[3])/2-m,2*m,2*m,0,0,120,120);
  known = known.filter(k=>k.name!==name); known.push({name,emb,thumb:t.toDataURL('image/jpeg',0.8),builtin:false});
  saveFaces(); renderRefs(); speak('Target '+name+' saved to database.',true); $(nameId).value='';
}
function delFace(name){ const k=known.find(x=>x.name===name); if(!k||!confirm('Remove '+name+'?')) return; if(k.builtin){ const del=JSON.parse(localStorage.getItem(LS_DEL)||'[]'); del.push(name); localStorage.setItem(LS_DEL,JSON.stringify(del)); } known=known.filter(x=>x!==k); saveFaces(); renderRefs(); }
function renderRefs(){
  $('refGal').innerHTML = known.length ? known.map(k=>`<div class="ref-card">${k.thumb?`<img src="${k.thumb}">`:`<div style="height:120px;display:flex;align-items:center;justify-content:center;font-size:48px;color:#ff5555;border:1px solid #ff0000;border-radius:6px">${k.name[0]}</div>`}<p>${k.name}</p><p style="color:#888;cursor:pointer" onclick="delFace('${k.name.replace(/'/g,'')}')">✕ remove</p></div>`).join('') : '<p style="color:#555">No targets in database.</p>';
}

// ---------- colour / direction ----------
function domColor(c, box){
  const x1=Math.max(0,box[0]|0), y1=Math.max(0,box[1]|0), w=Math.max(1,(box[2]-box[0])|0), h=Math.max(1,(box[3]-box[1])|0);
  const sx=x1+w*0.3, sy=y1+h*0.3, sw=Math.max(1,w*0.4), sh=Math.max(1,h*0.4);
  const t=document.createElement('canvas'); t.width=t.height=12; const x=t.getContext('2d'); x.drawImage(c,sx,sy,sw,sh,0,0,12,12);
  const d=x.getImageData(0,0,12,12).data, H=[],S=[],V=[];
  for(let i=0;i<d.length;i+=4){ const r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255, mx=Math.max(r,g,b), mn=Math.min(r,g,b), df=mx-mn; let hh=0; if(df){ if(mx===r) hh=((g-b)/df)%6; else if(mx===g) hh=(b-r)/df+2; else hh=(r-g)/df+4; hh*=30; if(hh<0) hh+=180; } H.push(hh); S.push(mx?df/mx*255:0); V.push(mx*255); }
  const med=a=>a.sort((p,q)=>p-q)[a.length>>1]; const h0=med(H),s0=med(S),v0=med(V);
  if(v0<40) return 'Black'; if(v0>200&&s0<40) return 'White'; if(s0<40) return 'Gray'; if(h0>=10&&h0<=30&&v0<120) return 'Brown';
  if(h0<10||h0>=165) return 'Red'; if(h0<22) return 'Orange'; if(h0<35) return 'Yellow'; if(h0<85) return 'Green'; if(h0<105) return 'Cyan'; if(h0<135) return 'Blue'; if(h0<155) return 'Purple'; return 'Pink';
}
function direction(p){ if(p.length<5) return 'Stationary'; const dx=p[p.length-1][0]-p[0][0], dy=p[p.length-1][1]-p[0][1]; if(Math.abs(dx)<15&&Math.abs(dy)<15) return 'Stationary'; return Math.abs(dx)>Math.abs(dy)?(dx>0?'Right':'Left'):(dy>0?'Down':'Up'); }
const cat = n => n==='person'?'PERSON':VEHICLES.has(n)?'VEHICLE':ANIMALS.has(n)?'ANIMAL':'OBJECT';

// ---------- tracking ----------
let tracks=[], nextId=1; const objectDetails={}; const newDetections=[]; let evidence=[], eviCount=0, lastSave=0;
function updateTracks(dets){
  for(const t of tracks) t.matched=false;
  for(const d of dets.sort((a,b)=>b.score-a.score)){
    let best=null,bi=0.25; for(const t of tracks){ if(t.matched||t.cls!==d.cls) continue; const v=iou(t.box,d.box); if(v>bi){bi=v;best=t;} }
    if(best){ best.box=d.box; best.score=d.score; best.matched=true; best.miss=0; best.d=d; }
    else { const t={id:nextId++,cls:d.cls,box:d.box,score:d.score,miss:0,matched:true,trail:[],name:COCO[d.cls].toUpperCase(),raw:COCO[d.cls],cat:cat(COCO[d.cls]),color:'Gray',crim:false,face:'N/A',age:0,isNew:true}; tracks.push(t); }
  }
  for(const t of tracks) if(!t.matched) t.miss++;
  tracks=tracks.filter(t=>t.miss<6);
}

// ---------- camera + loop ----------
const video=document.createElement('video'); video.setAttribute('playsinline',''); video.muted=true; video.style.display='none'; document.body.appendChild(video);
const cap=document.createElement('canvas'), capx=cap.getContext('2d',{willReadFrequently:true});
const view=$('view'), vx=view.getContext('2d');
let stream=null, facing='environment', running=false, frameNo=0, fcount=0, ftime=performance.now(), lastFaces=[];
async function startCam(){
  if(!modelsReady){ alert('AI is still loading, wait a few seconds'); return; }
  try{
    if(stream) stream.getTracks().forEach(t=>t.stop());
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:facing,width:{ideal:640},height:{ideal:480}},audio:false});
    video.srcObject=stream; await video.play();
    $('camMsg').style.display='none'; $('camBtn').innerHTML=IC('cam')+' CAMERA ON'; AC&&AC.resume(); beep(880,0.08,0.2);
    if(!running){ running=true; loop(); }
  }catch(e){ $('camMsg').style.display='flex'; $('camMsgT').innerHTML='Camera blocked<br>Allow camera permission in the browser (lock icon near the address bar) and tap START again.<br><small>'+e.message+'</small>'; }
}
function flipCam(){ facing = facing==='user'?'environment':'user'; if(running) startCam(); }

let aiCount=0, aiTime=performance.now(), renderCount=0, renderTime=performance.now(), vw=0, vh=0;
function renderLoop(){
  if(!running) return;
  if(video.readyState>=2 && video.videoWidth){
    const w=video.videoWidth, h=video.videoHeight; if(view.width!==w){view.width=w;view.height=h;}
    vw=w; vh=h; draw(w,h);
    renderCount++; const n=performance.now(); if(n-renderTime>=1000){ $('hfps').textContent=renderCount+' | AI '+aiCount; renderCount=0; aiCount=0; renderTime=n; }
  }
  requestAnimationFrame(renderLoop);
}
async function aiLoop(){
  while(running){
    try{ if(video.readyState>=2 && video.videoWidth){
      const w=video.videoWidth, h=video.videoHeight; if(cap.width!==w){cap.width=w;cap.height=h;}
      capx.drawImage(video,0,0,w,h); frameNo++;
      if(frameNo%2===0){ const dets=await runYolo(cap,w,h); updateTracks(dets); }
      let fresh=false;
      if(frameNo%4===1){ const L=letterbox(cap,w,h); lastFaces=await runYunet(L.data,L.s); fresh=true; }
      await matchFaces(lastFaces, fresh);
      postProcess(w,h); aiCount++;
    }}catch(e){ console.error(e); }
    await new Promise(r=>setTimeout(r,0));
  }
}
function loop(){ renderLoop(); aiLoop(); }
async function matchFaces(faces, fresh){
  if(!fresh||!known.length) return; let done=0;
  for(const f of faces){
    if(done>=2) break;
    const cx=(f.box[0]+f.box[2])/2, cy=(f.box[1]+f.box[3])/2; let tr=null, ba=1e12;
    for(const t of tracks){ if(t.cls!==0||t.miss) continue; if(cx>=t.box[0]&&cx<=t.box[2]&&cy>=t.box[1]&&cy<=t.box[3]){ const a=(t.box[2]-t.box[0])*(t.box[3]-t.box[1]); if(a<ba){ba=a;tr=t;} } }
    if(!tr) continue;
    if(tr.lastEmb && frameNo-tr.lastEmb < (tr.crim?30:6)) continue;
    tr.lastEmb=frameNo; done++;
    const e=await embed(cap,f.lm); let bs=0,bn=null; for(const k of known){ const s=dot(k.emb,e); if(s>bs){bs=s;bn=k.name;} }
    if(bs>=MATCH_MIN){ tr.crimNew = !tr.crim || tr.name!==bn; tr.crim=true; tr.name=bn; tr.face=Math.min(99.9,bs*100+30).toFixed(1)+'%'; tr.rawScore=bs; }
  }
}
function postProcess(w,h){
  const now=Date.now(); let anyCrim=false; const counts={PERSON:0,VEHICLE:0,ANIMAL:0,OBJECT:0};
  for(const t of tracks){
    if(t.miss) continue; counts[t.cat]++; if(t.crim) anyCrim=true;
    t.trail.push([(t.box[0]+t.box[2])/2,(t.box[1]+t.box[3])/2]); if(t.trail.length>30) t.trail.shift();
    t.dir=direction(t.trail);
    if(t.age%15===0){ t.color=domColor(cap,t.box); }
    t.age++;
    const obj={track_id:String(t.id),class:t.name,category:t.cat,confidence:+(t.score*100).toFixed(1),dominant_color:t.color,color_hex:COLOR_HEX[t.color]||'#888',direction:t.dir,is_criminal:t.crim,face_match:t.face,time:new Date().toLocaleTimeString()};
    objectDetails[obj.track_id]=obj;
    if(t.isNew){ t.isNew=false; newDetections.unshift(obj); if(newDetections.length>60) newDetections.pop(); if(!t.crimNew) addDetCard(obj); if(t.cat==='PERSON'&&!t.crim) beep(880,0.08,0.2); }
    if(t.crimNew){ t.crimNew=false; objectDetails[obj.track_id]=obj; addDetCard(obj); beep(1500,0.2,0.8); { const nm=t.name.charAt(0)+t.name.slice(1).toLowerCase(); speak(nm+' detected. Target locked: '+nm+'.',true); }; saveEvidence(t,obj,true); }
  }
  $('sys-dot').classList.toggle('alert',anyCrim);
  const act=tracks.filter(t=>!t.miss), tot=act.length;
  for(const [a,b] of [['sp','bp'],['sv','bv'],['sa','ba']]){} 
  $('sp').textContent=$('bp').textContent=counts.PERSON; $('sv').textContent=$('bv').textContent=counts.VEHICLE; $('sa').textContent=$('ba').textContent=counts.ANIMAL; $('bo').textContent=counts.OBJECT; $('st2').textContent=tot;
  if(frameNo%5===0) $('olist').innerHTML = act.length? act.map(o=>`<div onclick="showObj('${o.id}')" style="${o.crim?'color:red;font-weight:bold':''}"><span class="color-dot" style="background:${COLOR_HEX[o.color]||'#888'};border:none"></span> ${o.name} #${o.id} <span style="color:#555;margin-left:auto">${(o.score*100).toFixed(0)}%</span></div>`).join('') : 'Scanning...';
}
function addDetCard(det){
  const feed=$('dfeed'); if(feed.innerHTML.includes('Waiting')) feed.innerHTML='';
  const colors={PERSON:'#00ffa3',VEHICLE:'#00d2ff',ANIMAL:'#ffaa00',OBJECT:'#aaa'}, c=det.is_criminal?'#ff0000':(colors[det.category]||'#aaa');
  feed.innerHTML=`<div class="dc ${det.is_criminal?'crim':''}" onclick="showObj('${det.track_id}')" style="border-left-color:${c}"><div class="dh"><span class="dn" style="color:${c}">${det.class} #${det.track_id}</span><span class="dp">${det.confidence}%</span></div><div class="di"><span class="color-dot" style="background:${det.color_hex}"></span>${det.dominant_color} · ${det.direction}${det.is_criminal?' · FACE '+det.face_match:''} · ${det.time}</div></div>`+feed.innerHTML;
  while(feed.children.length>40) feed.lastChild.remove();
}
function saveEvidence(t,obj,crim){
  const b=t.box, x1=Math.max(0,b[0]|0), y1=Math.max(0,b[1]|0), w=Math.max(8,Math.min(cap.width-x1,(b[2]-b[0])|0)), h=Math.max(8,Math.min(cap.height-y1,(b[3]-b[1])|0));
  const c=document.createElement('canvas'); c.width=w; c.height=h; c.getContext('2d').drawImage(cap,x1,y1,w,h,0,0,w,h);
  const f=document.createElement('canvas'); f.width=480; f.height=Math.round(480*cap.height/cap.width); f.getContext('2d').drawImage(cap,0,0,f.width,f.height);
  eviCount++; evidence.unshift({id:'EVT-'+String(eviCount).padStart(5,'0'),class:t.name,conf:obj.confidence,color:t.color,crim:t.crim,face:t.face,time:new Date().toLocaleString(),crop:c.toDataURL('image/jpeg',0.8),frame:f.toDataURL('image/jpeg',0.7)});
  if(evidence.length>50) evidence.pop(); renderEviMini();
}
function renderEviMini(){ $('efeed2').innerHTML=evidence.slice(0,12).map(e=>`<div class="dc ${e.crim?'crim':''}"><div class="dh"><span class="dn">${e.id}</span><span class="dp">${e.conf}%</span></div><img src="${e.crop}" style="width:100%;border-radius:4px;margin:4px 0"><div class="di">${e.class} · ${e.time}</div></div>`).join('')||'<p style="color:#444;text-align:center">No evidence yet.</p>'; }
function renderEviFull(){ $('eviGrid').innerHTML=evidence.map(e=>`<div class="ec ${e.crim?'crim':''}"><img src="${e.frame}"><div class="en">${e.id} · ${e.class}</div><div class="ei">Confidence: ${e.conf}%<br>Color: ${e.color}<br>${e.crim?'FACE MATCH: '+e.face+'<br>':''}${e.time}<br><a style="color:#00d2ff" download="${e.id}-crop.jpg" href="${e.crop}">⬇ download crop</a> · <a style="color:#00d2ff" download="${e.id}-frame.jpg" href="${e.frame}">⬇ frame</a></div></div>`).join('')||'<p style="color:#555">No evidence captured yet.</p>'; }

function draw(w,h){
  vx.drawImage(video,0,0,w,h); const lw=Math.max(2,w/320);
  for(const t of tracks){ if(t.miss) continue; const [x1,y1,x2,y2]=t.box;
    const col=t.crim?'#ff0000':t.cat==='PERSON'?'#00ffb4':t.cat==='VEHICLE'?'#00b4ff':t.cat==='ANIMAL'?'#ffb400':'#b4b4b4';
    vx.strokeStyle=col; vx.fillStyle=col; vx.lineWidth=lw;
    if(t.trail.length>1){ vx.beginPath(); t.trail.forEach((p,i)=>i?vx.lineTo(p[0],p[1]):vx.moveTo(p[0],p[1])); vx.stroke(); }
    const L=Math.min(28,(x2-x1)/3,(y2-y1)/3); vx.beginPath();
    vx.moveTo(x1,y1+L);vx.lineTo(x1,y1);vx.lineTo(x1+L,y1); vx.moveTo(x2-L,y1);vx.lineTo(x2,y1);vx.lineTo(x2,y1+L); vx.moveTo(x1,y2-L);vx.lineTo(x1,y2);vx.lineTo(x1+L,y2); vx.moveTo(x2-L,y2);vx.lineTo(x2,y2);vx.lineTo(x2,y2-L); vx.stroke();
    const fs=Math.max(13,w/30); vx.font='bold '+fs+'px sans-serif';
    let lab=`${t.name} #${t.id} ${(t.score*100).toFixed(0)}% ${t.dir||''}`; if(t.crim) lab+=` | ACC:${t.face}`;
    const tw=vx.measureText(lab).width; vx.fillStyle='rgba(0,0,0,.6)'; vx.fillRect(x1,Math.max(0,y1-fs-4),tw+6,fs+4); vx.fillStyle=col; vx.fillText(lab,x1+3,Math.max(fs,y1-5));
    if(t.crim){ vx.fillStyle='#f00'; vx.font='bold '+(fs*1.2)+'px sans-serif'; vx.fillText(t.name+' LOCKED',x1,y2+fs*1.3); }
    else if(t.cat==='PERSON'||t.cat==='VEHICLE'){ vx.fillText(t.color,x1,y2+fs); }
  }
  vx.font='bold '+Math.max(11,w/55)+'px sans-serif'; vx.fillStyle=tracks.some(t=>t.crim&&!t.miss)?'#f00':'#00ffc8'; vx.fillText(`INSIGHTX | ${known.length} IDs | FPS:${$('hfps').textContent}`,8,h-8);
}

// ---------- UI helpers ----------
function goPage(id,btn){ document.querySelectorAll('.page').forEach(p=>p.classList.remove('on')); document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('on')); $('pg-'+id).classList.add('on'); btn.classList.add('on'); if(id==='evi') renderEviFull(); if(id==='ref') renderRefs(); }
function stab(n,el){ document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on')); document.querySelectorAll('.tp').forEach(t=>t.classList.remove('on')); el.classList.add('on'); $('tp-'+n).classList.add('on'); if(n==='evi2') renderEviMini(); }
function doSearch(){ const q=$('searchInput').value.toLowerCase(), el=$('searchRes'); const f=Object.values(objectDetails).filter(o=>o.class.toLowerCase().includes(q)||o.category.toLowerCase().includes(q)||o.dominant_color.toLowerCase().includes(q)); el.innerHTML=f.length?f.reverse().map(o=>`<div class="sr ${o.is_criminal?'crim':''}" onclick="showObj('${o.track_id}')"><div class="sn">${o.class} #${o.track_id}</div><div class="sd">${o.category} · ${o.dominant_color} · ${o.direction} · ${o.confidence}%</div></div>`).join(''):'<p style="color:#555">No results.</p>'; }
function showObj(id){ const o=objectDetails[id]; if(!o) return; beep(500,0.05,0.15); $('mbody').innerHTML=[['Object',o.class],['Category',o.category],['Confidence',o.confidence+'%'],['Color',`<span class="color-dot" style="background:${o.color_hex}"></span>${o.dominant_color}`],['Direction',o.direction],['Face match',o.is_criminal?o.face_match:'—'],['Seen',o.time]].map(r=>`<div class="rw"><span class="k">${r[0]}</span><span class="v">${r[1]}</span></div>`).join(''); $('modal').classList.add('show'); }
window.addEventListener('load', loadModels);

// ---------- fullscreen video ----------
(function(){
  const c=document.querySelector('.center');
  function openFS(){ if(c.classList.contains('fs')) return; c.classList.add('fs'); document.body.classList.add('fs-on'); try{ const f=c.requestFullscreen||c.webkitRequestFullscreen; if(f){ const r=f.call(c); if(r&&r.catch) r.catch(()=>{}); } }catch(e){} }
  window.closeFS=function(){ if(!c.classList.contains('fs')) return false; c.classList.remove('fs'); document.body.classList.remove('fs-on'); try{ if(document.fullscreenElement||document.webkitFullscreenElement){ (document.exitFullscreen||document.webkitExitFullscreen).call(document); } }catch(e){} return true; };
  c.addEventListener('click',function(e){
    if(e.target.closest('#fsBtn')){ if(c.classList.contains('fs')) closeFS(); else openFS(); return; }
    if(e.target.closest('#camMsg button')) return;
    if(c.classList.contains('fs')){ closeFS(); return; }
    if(running) openFS();
  });
  document.addEventListener('fullscreenchange',function(){ if(!document.fullscreenElement) closeFS(); });
  document.addEventListener('keydown',function(e){ if(e.key==='Escape') closeFS(); });
})();
const DEVS=['HARSHI','JEEVA','SANJEEV_PKT']; let devI=0;
function nextDev(){ devI=(devI+1)%DEVS.length; const b=document.getElementById('devname'); const n=b.cloneNode(false); n.textContent=DEVS[devI]; b.replaceWith(n); }
// ---------- intro video ----------
(function(){
  const box=document.getElementById('intro'), v=document.getElementById('introVid'), snd=document.getElementById('introSnd');
  let done=false;
  function reveal(){ if(done) return; done=true; box.classList.add('out'); setTimeout(function(){ box.remove(); try{URL.revokeObjectURL(v.src)}catch(x){} },600); }
  function finish(){ if(done||waiting) return; try{v.pause();}catch(x){} snd.style.display='none'; if(modelsReady){ reveal(); return; } waiting=true; box.classList.add('wait'); const t0=Date.now(); const t=setInterval(function(){ const er=(document.getElementById('pwr-mode')||{}).textContent||''; if(modelsReady||er.indexOf('ERROR')>=0||Date.now()-t0>90000){ clearInterval(t); reveal(); } },150); }
  let waiting=false;
  window.skipIntro=function(e){ if(e&&e.stopPropagation) e.stopPropagation(); finish(); };
  box.addEventListener('click',function(){ if(v.muted){ v.muted=false; snd.style.display='none'; try{v.play();}catch(x){} } else finish(); });
  v.addEventListener('ended',function(){ finish(); });
  v.addEventListener('error',function(){ finish(); });
  fetch('intro.mp4').then(function(r){ if(!r.ok) throw 0; return r.blob(); }).then(function(b){
    v.src=URL.createObjectURL(b); v.muted=false;
    const pr=v.play();
    if(pr&&pr.catch) pr.catch(function(){ v.muted=true; v.play().then(function(){ snd.style.display='block'; }).catch(finish); });
  }).catch(function(){ finish(); });
  setTimeout(function(){ if(!done && !v.src) finish(); },6000);
})();
