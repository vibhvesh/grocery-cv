import * as V from './vision.js?v=11';
const APP_VERSION='11',MODEL_TAG='dinov2-small-q8';
const TJS='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';
// ---------- IndexedDB ----------
let db;function openDB(){return new Promise((res,rej)=>{const r=indexedDB.open('visionbill',1);r.onupgradeneeded=()=>{const d=r.result;d.createObjectStore('items',{keyPath:'id'});d.createObjectStore('kv')};r.onsuccess=()=>{db=r.result;res()};r.onerror=()=>rej(r.error)})}
const tx=(s,m='readonly')=>db.transaction(s,m).objectStore(s),req=r=>new Promise((a,b)=>{r.onsuccess=()=>a(r.result);r.onerror=()=>b(r.error)});
const getItems=()=>req(tx('items').getAll()),putItem0=i=>req(tx('items','readwrite').put(i)),delItem0=id=>req(tx('items','readwrite').delete(id));
const kvGet=k=>req(tx('kv').get(k)),kvSet0=(k,v)=>req(tx('kv','readwrite').put(v,k));
const putItem=async i=>{await putItem0(i);scheduleSave()},delItem=async id=>{await delItem0(id);scheduleSave()},kvSet=async(k,v)=>{await kvSet0(k,v);if(k==='negs')scheduleSave()};
// ---------- state / utils ----------
const $=s=>document.querySelector(s);let singles=[];let T,det,fe,items=[],negs=[],bank=[],pendingPhotos=[],bill=[],scans=[],busy=Promise.resolve();
window.VB={get items(){return items},get scans(){return scans},get singles(){return singles},get bill(){return bill},get ready(){return !!(det&&fe)}};
const EMB_DIM=384,inr=n=>'₹'+(+n).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
const status=t=>$('#status').textContent=t,esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const MARGIN=0.02,thr=()=>+$('#thr').value,NIDB='Not in database',nameOf=id=>id==='unknown'?NIDB:(items.find(i=>i.id==id)||{}).name||'?';
const queue=fn=>busy=busy.then(fn).catch(e=>{console.error(e);status('Error: '+e.message)});
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,.tab').forEach(e=>e.classList.remove('active'));b.classList.add('active');$('#'+b.dataset.tab).classList.add('active')});
const loadImg=src=>new Promise((a,b)=>{const i=new Image();i.onload=()=>a(i);i.onerror=b;i.src=src});
function toCanvas(src,max=1280){const W=src.naturalWidth||src.videoWidth||src.width,H=src.naturalHeight||src.videoHeight||src.height,s=Math.min(1,max/Math.max(W,H));const c=document.createElement('canvas');c.width=Math.round(W*s);c.height=Math.round(H*s);c.getContext('2d').drawImage(src,0,0,c.width,c.height);return c}
function cropURL(cv,[x,y,w,h],pad=0.1,size=224){const s=Math.max(w,h)*(1+2*pad),cx=x+w/2,cy=y+h/2,c=document.createElement('canvas');c.width=c.height=size;const g=c.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,size,size);const k=size/s;g.drawImage(cv,(cx-s/2),(cy-s/2),s,s,0,0,size,size);return c.toDataURL('image/jpeg',.9)}
const raw=cv=>T.RawImage.fromCanvas(cv);
// ---------- models ----------
async function loadModels(){const hi=localStorage.getItem('vb_hi')==='1';$('#hiAcc').checked=hi;
 T=await import(TJS);T.env.allowLocalModels=false;try{if(localStorage.getItem('vb_proxy')!=='0')T.env.backends.onnx.wasm.proxy=true}catch{}// run WASM inference in a worker so the camera/capture UI never freezes

 const files={},progress=p=>{if(p.status==='progress'&&p.total){files[p.file]=[p.loaded,p.total];const L=Object.values(files).reduce((a,b)=>a+b[0],0),Tt=Object.values(files).reduce((a,b)=>a+b[1],0);status(`Downloading models ${(L/1e6).toFixed(0)}/${(Tt/1e6).toFixed(0)} MB`)}};
 const detId=hi?'Xenova/owlv2-base-patch16-ensemble':'Xenova/owlvit-base-patch32';
 const tryLoad=async device=>{const gpu=device==='webgpu';
  const d=await T.pipeline('zero-shot-object-detection',detId,{device,dtype:gpu?'q4f16':'q8',progress_callback:progress});
  const f=await T.pipeline('image-feature-extraction','onnx-community/dinov2-small',{device,dtype:gpu?'fp16':'q8',progress_callback:progress});
  const c=document.createElement('canvas');c.width=c.height=64;c.getContext('2d').fillRect(8,8,30,30);await d(raw(c),['an object']);await f(raw(c)); // smoke test
  return[d,f]};
 const t0=performance.now();let device='wasm';
 let adapter=null;try{adapter=navigator.gpu&&await navigator.gpu.requestAdapter()}catch{}
 if(adapter&&localStorage.getItem('vb_gpu')!=='0'){try{[det,fe]=await tryLoad('webgpu');device='webgpu'}catch(e){console.warn('WebGPU failed, falling back to WASM',e);localStorage.setItem('vb_gpu','0');location.reload();return new Promise(()=>{})}}
 if(!det)[det,fe]=await tryLoad('wasm');
 $('#engineInfo').textContent=`Detector: ${detId.split('/')[1]} · Embeddings: dinov2-small · Backend: ${device} · loaded in ${((performance.now()-t0)/1000).toFixed(1)}s`;window.VB.device=device}
// ---------- training data (incremental) ----------
let cols={};function rebuildBank(){bank=[];cols={};for(const it of items)cols[it.id]=it.photos.map(p=>p.col).filter(Boolean);for(const it of items)for(const p of it.photos)for(const e of p.emb||[])bank.push({l:it.id,e});for(const e of negs)bank.push({l:V.BG,e})}
async function embedPhoto(it,p){const im=await loadImg(p.src),cv=toCanvas(im);
 if(p.kind==='crop'){p.emb=await V.embedCrop(fe,raw(cv),[0,0,cv.width,cv.height],true);p.model=MODEL_TAG;p.thumb=p.src;return}
 const r=await V.trainPhoto(det,fe,raw(cv),{name:it.name});p.emb=r.pos;p.model=MODEL_TAG;p.box=r.box;p.thumb=cropURL(cv,r.box,0.1,128);
 // negatives: scene objects that don't look like any known product
 const kept=r.neg.filter(e=>V.classify(bank.filter(b=>b.l!==V.BG),e,0).conf<0.8);negs.push(...kept);if(negs.length>400)negs=negs.slice(-400);await kvSet('negs',negs)}
async function trainItem(it,force=false){let n=0;for(const p of it.photos)if(force||!p.emb||(p.emb[0]&&p.emb[0].length!==EMB_DIM)||(p.model&&p.model!==MODEL_TAG)){$('#trainInfo').textContent=`Learning ${it.name}: photo ${++n}…`;await embedPhoto(it,p)}
 if(n){await putItem(it);rebuildBank();reclassifyScans()}return n}
async function ensureColours(){let n=0;for(const it of items){let ch=false;for(const p of it.photos)if(!p.col&&p.src){try{p.col=await colOfSrc(p.kind==='crop'?p.src:(p.thumb||p.src));ch=true;n++}catch{}}if(ch)await putItem0(it)}if(n){scheduleSave();rebuildBank()}return n}
async function colOfSrc(src){const im=await loadImg(src),c=document.createElement('canvas');c.width=c.height=224;const g=c.getContext('2d');g.drawImage(im,0,0,224,224);return colOfCanvas(c)}
function colOfCanvas(c){const d=c.getContext('2d').getImageData(0,0,224,224);return V.colourHist(d.data,224,224,4,V.INNER224)}
async function trainAll(force=false){if(!det)return;await ensureColours();if(force){negs=[];await kvSet('negs',negs)}let n=0,k=0;const todo=items.filter(it=>force||it.photos.some(p=>!p.emb||(p.emb[0]&&p.emb[0].length!==EMB_DIM)||(p.model&&p.model!==MODEL_TAG)));
 for(const it of todo){status(`Learning ${k+1}/${todo.length}…`);$('#trainInfo').textContent=`Learning product ${++k}/${todo.length}: ${it.name}…`;n+=await trainItem(it,force);await refresh()}status('Ready ✔');if(todo.length&&/Added/.test($('#samplesInfo').textContent))$('#samplesInfo').textContent='✔ Sample products learned – open Billing and scan a photo.';
 $('#trainInfo').textContent=`Model: ${bank.filter(b=>b.l!==V.BG).length} product embeddings from ${items.reduce((s,i)=>s+i.photos.length,0)} photos · ${negs.length} background examples. Updates automatically.`}
$('#samplesBtn').onclick=async()=>{const btn=$('#samplesBtn'),info=$('#samplesInfo');btn.disabled=true;
 try{info.textContent='Fetching sample list…';const man=await (await fetch('samples/manifest.json?v='+APP_VERSION,{cache:'no-store'})).json();const embs=await fetch('samples/embeddings.json?v='+APP_VERSION).then(r=>r.ok?r.json():null).catch(()=>null);if(embs&&embs.model===MODEL_TAG)man.forEach(m=>{m.model=MODEL_TAG;m.emb=embs.items[m.name]});
  const toURL=async u=>{const b=await (await fetch(u+'?v='+APP_VERSION)).blob();return new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result);f.readAsDataURL(b)})};
  let n=0,k=0;for(const m of man){k++;if(items.some(i=>i.name===m.name))continue;info.textContent=`Adding ${k}/${man.length}: ${m.name}…`;
   const pre=m.model===MODEL_TAG&&m.emb;const photos=await Promise.all(m.photos.map(async(u,j)=>({src:await toURL(u),kind:'crop',thumb:null,...(pre&&m.emb[j]?{emb:m.emb[j],model:MODEL_TAG}:{})})));photos.forEach(p=>p.thumb=p.src);
   await putItem({id:crypto.randomUUID(),name:m.name,price:m.price,category:m.category,unit:m.unit,notes:m.notes,photos});n++;await refresh()}
  if(embs&&embs.negs&&embs.model===MODEL_TAG&&localStorage.getItem('vb_sampleNegs')!==embs.created){negs=[...negs,...embs.negs];await kvSet('negs',negs);localStorage.setItem('vb_sampleNegs',embs.created);rebuildBank();reclassifyScans()}
  info.textContent=n?`✔ Added ${n} sample products${man[0].emb?' with pre-learned embeddings (no relearning needed)':''}. ${man[0].emb?'':det?'Learning them now…':'They will be learned as soon as the AI models finish downloading.'}`:'✔ Sample products are already loaded.';
  queue(()=>trainAll())}catch(e){console.error(e);info.textContent='Could not load samples: '+e.message}finally{btn.disabled=false}};
$('#trainBtn').onclick=()=>queue(()=>trainAll(true));
// ---------- products page ----------
async function fileToDataURL(f,max=1024){const i=await loadImg(URL.createObjectURL(f));return toCanvas(i,max).toDataURL('image/jpeg',.88)}
$('#iphotos').onchange=async e=>{for(const f of e.target.files)pendingPhotos.push({src:await fileToDataURL(f),kind:'photo'});e.target.value='';renderPending()};
function renderPending(){$('#previewPhotos').innerHTML=pendingPhotos.map((p,i)=>`<img src="${p.thumb||p.src}" title="click to remove" data-i="${i}">`).join('');$('#previewPhotos').querySelectorAll('img').forEach(im=>im.onclick=()=>{pendingPhotos.splice(+im.dataset.i,1);renderPending()})}
function resetForm(){$('#itemForm').reset();$('#itemId').value='';pendingPhotos=[];renderPending()}
$('#resetForm').onclick=resetForm;
$('#itemForm').onsubmit=async e=>{e.preventDefault();const id=$('#itemId').value||crypto.randomUUID();
 const it={id,name:$('#iname').value.trim(),price:+$('#iprice').value,category:$('#icat').value,unit:$('#iunit').value,notes:$('#inotes').value,photos:pendingPhotos};
 await putItem(it);resetForm();await refresh();queue(async()=>{await trainItem(items.find(i=>i.id==id));await trainAll()})};
async function refresh(){items=(await getItems()).map(i=>({...i,photos:(i.photos||[]).map(p=>typeof p==='string'?{src:p,kind:'photo'}:p)}));items.sort((a,b)=>a.name.localeCompare(b.name));rebuildBank();
 $('#itemList').innerHTML=items.length?items.map(i=>`<div class="item"><div class="thumbs">${i.photos.slice(0,4).map(p=>`<img src="${p.thumb||p.src}">`).join('')}</div><div class="info"><b>${esc(i.name)}</b> – ${inr(i.price)}<div class="muted">${esc(i.category)} · ${esc(i.unit)} · ${i.photos.length} photos${i.photos.some(p=>!p.emb)?' (learning…)':''} ${i.notes?'· '+esc(i.notes):''}</div></div><button data-e="${i.id}">Edit</button><button data-d="${i.id}">🗑</button></div>`).join(''):'<p class="muted">No products yet. You can add them here, or scan a photo on the Billing tab and tap “➕ New product” on each crop.</p>';
 $('#itemList').querySelectorAll('[data-e]').forEach(b=>b.onclick=()=>{const i=items.find(x=>x.id==b.dataset.e);$('#itemId').value=i.id;$('#iname').value=i.name;$('#iprice').value=i.price;$('#icat').value=i.category;$('#iunit').value=i.unit;$('#inotes').value=i.notes;pendingPhotos=[...i.photos];renderPending();scrollTo(0,0)});
 $('#itemList').querySelectorAll('[data-d]').forEach(b=>b.onclick=async()=>{if(confirm('Delete product?')){await delItem(b.dataset.d);await refresh();reclassifyScans()}});
 $('#manualItem').innerHTML=items.map(i=>`<option value="${i.id}">${esc(i.name)} (${inr(i.price)})</option>`).join('');renderBill()}
// ---------- active learning from crops ----------
async function addCropTo(id,d,kind='crop'){const it=items.find(i=>i.id==id);if(!it)return;it.photos.push({col:d.col||null,words:wordsOf(d.text),src:d.crop,kind:'crop',emb:d.es?d.es.slice(0,12):await V.embedCrop(fe,raw(toCanvas(await loadImg(d.crop))),[0,0,224,224],true),model:MODEL_TAG,thumb:d.crop,from:kind});
 await putItem(it);await refresh();reclassifyScans()}
// every confirmation or correction is saved as a training crop (embedding + colour + read words)
function setLabel(d,id,render=true){const was=d.guess;d.id=id;d.manual=true;d.open=false;if(render)renderScans();if(id!=='unknown')queue(()=>addCropTo(id,d,id===was?'confirm':'correction'))}
async function newProductFromCrop(d,name,price){if(!name){name=prompt('New product name:');if(!name)return;price=+prompt(`Price of ${name} in ₹:`,'0')||0}
 const it={id:crypto.randomUUID(),name:name.trim(),price,category:'',unit:'',notes:'',photos:[]};await putItem(it);items.push(it);d.id=it.id;d.manual=true;d.open=false;await addCropTo(it.id,d,'new');renderScans()}
function reclassifyScans(){for(const sc of scans)for(const d of sc.dets){const r=V.decideFused(V.scoreViews(bank,d.es),cols,d.col,d.ocr||{},thr(),MARGIN);if(d.manual&&items.find(i=>i.id==d.id)){d.top3=r.top3;continue}Object.assign(d,r)}renderScans()}
// ---------- scanning ----------
async function scanCanvas(cv,full,tDecode=0){full=full||cv;const T0=performance.now(),tm={decode:Math.round(tDecode)};status('Step 1/2: finding items…');const R=raw(cv);
 const boxes=await V.segment(det,R,{extra:items.map(i=>i.name)});tm.detect=Math.round(performance.now()-T0);
 const sc={cv,full,dets:[],sel:-1,pending:boxes.length,tm};scans.push(sc);
 // pass 1 (fast): one padded full-res crop per item, all embedded in one batched call
 const s=full.width/cv.width;let t=performance.now();const v1=boxes.map(b=>viewCanvases(full,b.box,s,'fast')[0]);
 status(`Step 2/2: recognising ${boxes.length} items…`);const e1=await V.embedBatch(fe,v1.map(raw));tm.pass1=Math.round(performance.now()-t);
 let ds=boxes.map((b,i)=>{const es=[e1[i]];const col=colOfCanvas(v1[i]);const r=V.decideFused(V.scoreViews(bank,es),cols,col,{},thr(),MARGIN);return{box:b.box,col,big:b.big,es,crop:cropURL(full,b.box.map(v=>v*s)),...r}});
 sc.dets=V.mergeSame(V.dedupe(ds.filter(d=>!(d.big&&d.id==='unknown'))));sc.pending=0;renderScans();
 // pass 2 (closer look) only for uncertain items: +6 views (3 rotations, flip, tight crop, centre zoom)
 t=performance.now();const thorough=false,nx=6;const unsure=!$('#closer').checked?[]:sc.dets.filter(d=>d.conf<0.7||d.margin<0.05);
 if(unsure.length){status(`Closer look at ${unsure.length} uncertain item(s)…`);const vs=unsure.flatMap(d=>viewCanvases(full,d.box,s,thorough?'thorough':'extra'));const e2=await V.embedBatch(fe,vs.map(raw));
  unsure.forEach((d,i)=>{d.es=[...d.es,...e2.slice(i*nx,i*nx+nx)];const r=V.decideFused(V.scoreViews(bank,d.es),cols,d.col,d.ocr||{},thr(),MARGIN);if(!d.manual)Object.assign(d,r);else d.top3=r.top3;d.closer=true});sc.dets=V.mergeSame(V.dedupe(sc.dets))}
 if($('#ocr').checked){const still=sc.dets.filter(d=>!d.manual&&(d.id==='unknown'||d.margin<0.05));if(still.length){t=performance.now();status(`Reading packet text on ${still.length} item(s)…`);await ocrDets(sc,still);tm.ocr=Math.round(performance.now()-t)}}
 tm.pass2=Math.round(performance.now()-t);tm.unsure=unsure.length;sc.ms=Math.round(performance.now()-T0+tDecode);renderScans();status('Ready ✔')}
// optional printed-text cue (tesseract.js, loaded only when enabled): words from product names + words read on confirmed crops
let ocrW=null;async function ocrWorker(){if(ocrW)return ocrW;status('Loading text reader (≈4 MB, once)…');const Ts=(await import('https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.esm.min.js')).default;ocrW=await Ts.createWorker('eng');await ocrW.setParameters({tessedit_pageseg_mode:'11'});return ocrW}
const GENERIC=new Set(['BLACK','SALT','PACK','PREMIUM','PACKET','POUCH','WITH','FROM','ONLY','NEW']);
function vocab(){const v={};for(const it of items){const w=new Set(it.name.toUpperCase().replace(/[^A-Z ]/g,' ').split(/\s+/).filter(x=>x.length>=4&&!GENERIC.has(x)));for(const p of it.photos)for(const x of p.words||[])w.add(x);v[it.id]=[...w]}return v}
async function ocrText(full,box,s){const [x,y,w,h]=box.map(v=>v*s),W=await ocrWorker();const out=[];for(const r of [0,1,2,3]){const k=Math.min(1,640/Math.max(w,h)),c=document.createElement('canvas'),sw=r%2?h*k:w*k,sh=r%2?w*k:h*k;c.width=sw;c.height=sh;const g=c.getContext('2d');g.filter='grayscale(1) contrast(1.4)';g.translate(sw/2,sh/2);g.rotate(r*Math.PI/2);g.drawImage(full,x,y,w,h,-w*k/2,-h*k/2,w*k,h*k);const {data}=await W.recognize(c);out.push((data.words||[]).filter(q=>q.confidence>50).map(q=>q.text).join(' '))}return out.join(' ')}
async function ocrDets(sc,ds){const s=sc.full.width/sc.cv.width,voc=vocab();for(const d of ds){try{d.text=await ocrText(sc.full,d.box,s);d.ocr=V.ocrHits(d.text,voc);if(!d.manual)Object.assign(d,V.decideFused(V.scoreViews(bank,d.es),cols,d.col,d.ocr,thr(),MARGIN))}catch(e){console.warn('ocr',e)}}renderScans()}
const wordsOf=t=>[...new Set((String(t||'').toUpperCase().match(/[A-Z]{4,}/g)||[]).filter(x=>!GENERIC.has(x)))].slice(0,20);
// views of one crop from the ORIGINAL (full-res) image
function viewCanvases(full,box,s,mode='fast'){const [x,y,w,h]=box.map(v=>v*s),out=[];
 const sq=(pad,zoom=1)=>{const S=Math.max(w,h)*(1+2*pad)*zoom,cx=x+w/2,cy=y+h/2,c=document.createElement('canvas');c.width=c.height=224;const g=c.getContext('2d');g.fillStyle='#d7d4d0';g.fillRect(0,0,224,224);const k=224/S,m=0.03;g.save();g.beginPath();g.rect((x-m*w-(cx-S/2))*k,(y-m*h-(cy-S/2))*k,w*(1+2*m)*k,h*(1+2*m)*k);g.clip();g.drawImage(full,cx-S/2,cy-S/2,S,S,0,0,224,224);g.restore();return c};
 const tf=(src,rot,flip)=>{const c=document.createElement('canvas');c.width=c.height=224;const g=c.getContext('2d');g.translate(112,112);g.rotate(rot*Math.PI/2);if(flip)g.scale(-1,1);g.drawImage(src,-112,-112);return c};
 if(mode==='fast')return[sq(0.1)];if(mode==='thorough'){const o=[];for(const pad of [0.1,0.02]){const c=sq(pad);for(let r=0;r<4;r++)if(pad!==0.1||r)o.push(tf(c,r,false));o.push(tf(c,0,true))}o.push(sq(0.1,0.8));return o}const c=sq(0.1),t=sq(0.02);return[tf(c,1,false),tf(c,2,false),tf(c,3,false),tf(c,0,true),t,sq(0.1,0.8)]}
async function makeDet(sc,box){const s=sc.full.width/sc.cv.width;const vs=[...viewCanvases(sc.full,box,s,'fast'),...viewCanvases(sc.full,box,s,'extra')];const es=await V.embedBatch(fe,vs.map(raw));
 const col=colOfCanvas(vs[0]);const r=V.decideFused(V.scoreViews(bank,es),cols,col,{},thr(),MARGIN);return{box,es,col,crop:cropURL(sc.full,box.map(v=>v*s)),...r}}
const COLORS=['#2563eb','#16a34a','#d97706','#7c3aed','#0891b2','#db2777','#65a30d','#0f766e'];
function renderScans(){const root=$('#scanResults');root.innerHTML='';const showU=$('#showUnknown').checked;
 scans.forEach((sc,si)=>{const wrap=document.createElement('div');wrap.className='scanimg';const W=sc.cv.width,H=sc.cv.height;
  const cv=document.createElement('canvas');cv.className='prev';cv.width=W;cv.height=H;const g=cv.getContext('2d');g.drawImage(sc.cv,0,0);const lw=Math.max(2,W/300),fs=Math.max(14,W/40);
  sc.dets.forEach((d,i)=>{if(d.id==='unknown'&&!showU)return;const col=d.id==='unknown'?'#dc2626':COLORS[i%COLORS.length];g.lineWidth=i==sc.sel?lw*2.5:lw;g.strokeStyle=col;g.strokeRect(...d.box);g.font=`bold ${fs}px sans-serif`;const t=`${i+1}. ${d.id==='unknown'?'?':nameOf(d.id)}`;g.fillStyle=col;g.fillRect(d.box[0],d.box[1],g.measureText(t).width+10,fs*1.3);g.fillStyle='#fff';g.fillText(t,d.box[0]+5,d.box[1]+fs)});
  let start=null;const pt=e=>{const r=cv.getBoundingClientRect();return[(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height]};
  cv.onpointerdown=e=>{start=pt(e);cv.setPointerCapture(e.pointerId)};
  cv.onpointerup=e=>{if(!start)return;const p=pt(e),b=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])];const s0=start;start=null;
   if(b[2]<W/40||b[3]<H/40){sc.sel=sc.dets.findIndex(d=>s0[0]>=d.box[0]&&s0[0]<=d.box[0]+d.box[2]&&s0[1]>=d.box[1]&&s0[1]<=d.box[1]+d.box[3]);return renderScans()}
   queue(async()=>{const d=await makeDet(sc,b);d.manual=false;sc.dets.push(d);renderScans()})};
  wrap.append(cv);let hidden=0;
  sc.dets.forEach((d,i)=>{if(d.id==='unknown'&&!showU){hidden++;return}const unk=d.id==='unknown';const row=document.createElement('div');row.className='det'+(i==sc.sel?' sel':'')+(unk?' unk':'');
   const pct=v=>Math.round(v*100)+'%',chips=(d.top3||[]).map(t=>`<button class="chip pick ${t.id===d.id?'on':''}" data-id="${t.id}">${esc(nameOf(t.id))} ${pct(t.s)}</button>`).join('');
   const sugg=d.manual?d.id:(d.id!=='unknown'?d.id:d.guess),sname=sugg&&sugg!=='unknown'?esc(nameOf(sugg)):NIDB;
   row.innerHTML=`<div class="dhead"><span class="n">${i+1}</span><img src="${d.crop}" class="dthumb"><div class="dmain">${d.manual?`<b>Item ${i+1}: ${d.id==='unknown'?'❓ '+NIDB:esc(nameOf(d.id))}</b> <span class="tag">✔ confirmed</span>`:`<b>Item ${i+1}: ${unk?'not sure – maybe':'I think this is'} ${sname}?</b> <span class="conf ${unk?'low':''}">${pct(d.conf)}</span> <span class="tag ask">needs your OK</span>`}
    <div class="muted small">Top guesses (tap the right one): ${chips||'—'}${d.text?` · text read: “${esc(d.text.slice(0,40))}”`:''}</div></div><input type="number" min="1" value="${d.qty||1}" class="qty" title="quantity"></div>
    <div class="acts">${!d.manual&&sugg&&sugg!=='unknown'?`<button data-a="yes" class="primary">✔ Yes, ${sname}</button>`:''}<button data-a="label">${d.manual?'✏️ Change':'✏️ Something else'}</button><button data-a="del" title="delete box">🗑</button></div>
    <div class="picker" ${d.open?'':'hidden'}><input class="psearch" placeholder="🔍 Search products…"><div class="plist">${items.map(it=>`<button class="chip pick" data-id="${it.id}" data-n="${esc(it.name.toLowerCase())}">${esc(it.name)} · ${inr(it.price)}</button>`).join('')}</div>
     <div class="row"><input class="pnew" placeholder="Type new label (new product)"><input class="pprice" type="number" min="0" step="0.5" placeholder="₹ price" style="width:90px"><button class="primary" data-a="create">Save</button></div>
     <div class="row"><button data-a="setunk">Mark as ${NIDB}</button><button data-a="close">Close</button></div></div>`;
   row.querySelectorAll('.pick').forEach(b=>b.onclick=e=>{e.stopPropagation();setLabel(d,b.dataset.id)});
   row.querySelector('.qty').onchange=e=>{d.qty=Math.max(1,+e.target.value||1);renderSummary()};
   row.querySelector('[data-a=label]').onclick=()=>{d.open=!d.open;renderScans()};
   row.querySelector('[data-a=close]').onclick=()=>{d.open=false;renderScans()};
   row.querySelector('[data-a=setunk]').onclick=()=>{d.id='unknown';d.manual=true;d.open=false;renderScans()};
   row.querySelector('.psearch').oninput=e=>{const q=e.target.value.toLowerCase();row.querySelectorAll('.plist .pick').forEach(b=>b.hidden=!b.dataset.n.includes(q))};
   row.querySelector('[data-a=create]').onclick=()=>{const name=row.querySelector('.pnew').value.trim();if(!name)return row.querySelector('.pnew').focus();let price=row.querySelector('.pprice').value;
    if(price===''){price=prompt(`Price of ${name} in ₹:`,'0');if(price===null)return}queue(()=>newProductFromCrop(d,name,+price||0))};
   const yb=row.querySelector('[data-a=yes]');if(yb)yb.onclick=()=>setLabel(d,sugg);
   row.querySelector('[data-a=del]').onclick=()=>{sc.dets.splice(i,1);sc.sel=-1;renderScans()};
   row.onclick=e=>{if(e.target===row||e.target.className=='n'){sc.sel=i;renderScans()}};wrap.append(row)});
  const ctl=document.createElement('div');ctl.className='row';ctl.innerHTML=`<span class="muted">${sc.dets.length} item(s) found${hidden?`, ${hidden} not-in-database hidden`:''} · ${sc.ms} ms${sc.tm?` (decode ${sc.tm.decode}, find ${sc.tm.detect}, recognise ${sc.tm.pass1}, closer look ${sc.tm.pass2} ms on ${sc.tm.unsure})`:''}</span><button>Remove photo</button>`;ctl.querySelector('button').onclick=()=>{scans.splice(si,1);renderScans()};wrap.append(ctl);root.append(wrap)});
 renderSummary()}
function counts(){const c={};scans.forEach(s=>s.dets.forEach(d=>{if(d.manual)c[d.id]=(c[d.id]||0)+(d.qty||1)}));return c}
const unconfirmed=()=>scans.flatMap(s=>s.dets.filter(d=>!d.manual));
function renderSummary(){const c=counts();$('#summaryBox').hidden=!scans.length;const uc=unconfirmed();$('#confirmAll').hidden=!uc.some(d=>(d.id!=='unknown'?d.id:d.guess));$('#confirmAll').textContent=`✔ Confirm all ${uc.length} suggestion(s)`;$('#summary').innerHTML=(uc.length?`<p class="muted">${uc.length} item(s) still waiting for your confirmation – they are not counted yet.</p>`:'')+Object.entries(c).sort((a,b)=>(a[0]==='unknown')-(b[0]==='unknown')).map(([id,n])=>`<span class="chip ${id==='unknown'?'unk':''}">${esc(nameOf(id))} × ${n}</span>`).join('')||'<span class="muted">Nothing detected</span>'}
$('#confirmAll').onclick=()=>{for(const d of unconfirmed()){const g=d.id!=='unknown'?d.id:d.guess;if(g)setLabel(d,g,false)}renderScans()};
$('#addAll').onclick=()=>{const uc=unconfirmed();if(uc.length&&!confirm(`${uc.length} item(s) are not confirmed yet and will be left out. Continue?`))return;const c=counts();for(const [id,n] of Object.entries(c))if(id!=='unknown')addToBill(id,n);scans=[];renderScans()};
$('#clearScans').onclick=()=>{scans=[];renderScans()};$('#ocr').checked=localStorage.getItem('vb_ocr')==='1';$('#ocr').onchange=()=>localStorage.setItem('vb_ocr',$('#ocr').checked?1:0);$('#closer').checked=localStorage.getItem('vb_closer')!=='0';$('#closer').onchange=()=>localStorage.setItem('vb_closer',$('#closer').checked?1:0);$('#showUnknown').onchange=()=>{localStorage.setItem('vb_showU',$('#showUnknown').checked?1:0);renderScans()};
$('#thr').oninput=()=>{$('#thrVal').textContent=(+$('#thr').value).toFixed(2);localStorage.setItem('vb_thr2',$('#thr').value);reclassifyScans()};
$('#hiAcc').onchange=()=>{localStorage.setItem('vb_hi',$('#hiAcc').checked?1:0);if(confirm('Reload now to switch detector?'))location.reload()};
const needModels=()=>{if(!det){alert('Models are still loading…');return true}};
$('#billPhotos').onchange=async e=>{if(needModels())return;const fs=[...e.target.files];e.target.value='';for(const f of fs){const t=performance.now();status('Decoding photo…');const {cv,full}=await decodePhoto(f);const td=performance.now()-t;queue(()=>scanCanvas(cv,full,td))}};
let stream;$('#camBtn').onclick=async()=>{const v=$('#cam');if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;v.hidden=$('#snapBtn').hidden=true;return}
 try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280}}});v.srcObject=stream;await v.play();v.hidden=$('#snapBtn').hidden=false}catch(err){alert('Camera unavailable: '+err.message)}};
$('#snapBtn').onclick=()=>{if(needModels())return;const cv=toCanvas($('#cam')),full=toCanvas($('#cam'),2048);queue(()=>scanCanvas(cv,full))};
async function decodePhoto(f){let bmp;try{bmp=await createImageBitmap(f,{imageOrientation:'from-image'})}catch{bmp=await loadImg(URL.createObjectURL(f))}
 const W=bmp.width||bmp.naturalWidth,H=bmp.height||bmp.naturalHeight;let full=bmp;
 if(Math.max(W,H)>3072){try{full=await createImageBitmap(bmp,{resizeWidth:Math.round(W*3072/Math.max(W,H)),resizeHeight:Math.round(H*3072/Math.max(W,H)),resizeQuality:'high'})}catch{full=toCanvas(bmp,3072)}}
 return{full:toCanvas(full,3072),cv:toCanvas(full,1280)}}
// ---------- bill ----------
function addToBill(id,q){const l=bill.find(b=>b.id==id);l?l.qty+=q:bill.push({id,qty:q});saveBill()}
function saveBill(){localStorage.setItem('vb_bill',JSON.stringify(bill));localStorage.setItem('vb_gst',$('#gst').value);renderBill()}
function totals(){const lines=bill.map(b=>{const i=items.find(x=>x.id==b.id);return i&&{...b,name:i.name,unit:i.unit,price:i.price,total:i.price*b.qty}}).filter(Boolean);const sub=lines.reduce((s,l)=>s+l.total,0),g=sub*(+$('#gst').value||0)/100;return{lines,sub,g,grand:sub+g}}
function renderBill(){const t=totals();$('#billDate').textContent=new Date().toLocaleString('en-IN');
 $('#billTable tbody').innerHTML=t.lines.map(l=>`<tr><td>${esc(l.name)}<div class="muted">${esc(l.unit)}</div></td><td><input type="number" min="0" step="any" value="${l.qty}" data-q="${l.id}"></td><td>${inr(l.price)}</td><td>${inr(l.total)}</td><td class="np"><button data-r="${l.id}">✕</button></td></tr>`).join('')||'<tr><td colspan="5" class="muted">No items yet</td></tr>';
 document.querySelectorAll('[data-q]').forEach(e=>e.onchange=()=>{const b=bill.find(x=>x.id==e.dataset.q);b.qty=+e.value;if(b.qty<=0)bill=bill.filter(x=>x!==b);saveBill()});
 document.querySelectorAll('[data-r]').forEach(e=>e.onclick=()=>{bill=bill.filter(x=>x.id!=e.dataset.r);saveBill()});
 $('#subtotal').textContent=inr(t.sub);$('#gstAmt').textContent=inr(t.g);$('#grand').textContent=inr(t.grand)}
$('#gst').oninput=saveBill;$('#addManual').onclick=()=>$('#manualItem').value&&addToBill($('#manualItem').value,1);$('#clearBill').onclick=()=>{bill=[];saveBill()};
$('#pdfBtn').onclick=()=>{const {jsPDF}=window.jspdf,d=new jsPDF(),t=totals(),R=n=>'Rs. '+n.toFixed(2);
 d.setFontSize(16);d.text('VisionBill - Grocery Invoice',14,16);d.setFontSize(10);d.text(new Date().toLocaleString('en-IN'),14,23);
 d.autoTable({startY:28,head:[['#','Item','Qty','Unit price','Total']],body:t.lines.map((l,i)=>[i+1,l.name+(l.unit?` (${l.unit})`:''),l.qty,R(l.price),R(l.total)]),
  foot:[['','','','Subtotal',R(t.sub)],['','','',`GST ${$('#gst').value||0}%`,R(t.g)],['','','','Grand total',R(t.grand)]],theme:'grid',headStyles:{fillColor:[22,163,74]},footStyles:{fillColor:[240,240,240],textColor:20}});
 d.save(`bill-${Date.now()}.pdf`)};
// ---------- persistence: folder (File System Access API) + export/import ----------
let dir=null,saveT=null,loadingSnap=false;const hasFS='showDirectoryPicker' in window;const r4=v=>Math.round(v*1e4)/1e4;
function settingsObj(){const s={};for(const k of ['vb_thr2','vb_showU','vb_hi','vb_gst'])s[k]=localStorage.getItem(k);return s}
async function snapshot(){const its=(await getItems());return{format:'visionbill-db',version:1,app:APP_VERSION,model:MODEL_TAG,saved:new Date().toISOString(),items:its,negs:(await kvGet('negs'))||[],settings:settingsObj()}}
async function applySnapshot(s){if(s.format!=='visionbill-db')throw new Error('Not a VisionBill database');loadingSnap=true;
 const st=tx('items','readwrite');await req(st.clear());for(const it of s.items)await putItem0(it);await kvSet0('negs',s.negs||[]);negs=s.negs||[];
 for(const [k,v] of Object.entries(s.settings||{}))if(v!=null)localStorage.setItem(k,v);loadingSnap=false;await refresh();if(det)queue(()=>trainAll())}
const dataURLtoBlob=u=>fetch(u).then(r=>r.blob());const blobToDataURL=b=>new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result);f.readAsDataURL(b)});
async function writeFile(d,name,data){const fh=await d.getFileHandle(name,{create:true});const w=await fh.createWritable();await w.write(data);await w.close()}
async function exists(d,name){try{await d.getFileHandle(name);return true}catch{return false}}
async function saveToFolder(){if(!dir||loadingSnap)return;const s=await snapshot();const ph=await dir.getDirectoryHandle('photos',{create:true});const emb={model:MODEL_TAG,items:{}};
 const products=s.items.map(it=>({id:it.id,name:it.name,price:it.price,category:it.category,unit:it.unit,notes:it.notes,photos:it.photos.map((p,i)=>{const f=`${it.id}_${i}.jpg`,t=`${it.id}_${i}_thumb.jpg`;return{file:'photos/'+f,thumb:p.thumb&&p.thumb!==p.src?'photos/'+t:null,kind:p.kind,box:p.box||null,from:p.from||null}})}));
 for(const it of s.items){emb.items[it.id]=it.photos.map(p=>(p.emb||[]).map(v=>v.map(r4)));for(const [i,p] of it.photos.entries()){const f=`${it.id}_${i}.jpg`,t=`${it.id}_${i}_thumb.jpg`;
   if(!await exists(ph,f))await writeFile(ph,f,await dataURLtoBlob(p.src));if(p.thumb&&p.thumb!==p.src&&!await exists(ph,t))await writeFile(ph,t,await dataURLtoBlob(p.thumb))}}
 emb.negs=s.negs.map(v=>v.map(r4));
 await writeFile(dir,'products.json',JSON.stringify({format:'visionbill-db',version:1,app:APP_VERSION,model:MODEL_TAG,saved:s.saved,products},null,1));
 await writeFile(dir,'embeddings.json',JSON.stringify(emb));await writeFile(dir,'settings.json',JSON.stringify(s.settings,null,1));
 $('#folderInfo').textContent=`💾 Saved to folder “${dir.name}” at ${new Date().toLocaleTimeString('en-IN')} (${s.items.length} products)`}
async function loadFromFolder(){if(!await exists(dir,'products.json')){await saveToFolder();return false}
 const rd=async n=>JSON.parse(await (await (await dir.getFileHandle(n)).getFile()).text());const P=await rd('products.json'),E=await rd('embeddings.json').catch(()=>({items:{},negs:[]})),S=await rd('settings.json').catch(()=>({}));
 const ph=await dir.getDirectoryHandle('photos',{create:true});const file=async path=>blobToDataURL(await (await ph.getFileHandle(path.replace('photos/',''))).getFile());
 const its=[];for(const p of P.products){const photos=[];for(const [i,q] of p.photos.entries()){const src=await file(q.file);const e=E.model===MODEL_TAG?(E.items[p.id]||[])[i]:null;photos.push({src,thumb:q.thumb?await file(q.thumb):src,kind:q.kind,box:q.box,from:q.from,...(e&&e.length?{emb:e,model:MODEL_TAG}:{})})}
  its.push({id:p.id,name:p.name,price:p.price,category:p.category,unit:p.unit,notes:p.notes,photos})}
 await applySnapshot({format:'visionbill-db',items:its,negs:E.model===MODEL_TAG?E.negs||[]:[],settings:S});$('#folderInfo').textContent=`📂 Loaded ${its.length} products from folder “${dir.name}”. Autosave is on.`;return true}
function scheduleSave(){if(!dir||loadingSnap)return;clearTimeout(saveT);saveT=setTimeout(()=>saveToFolder().catch(e=>{console.error(e);$('#folderInfo').textContent='Folder save failed: '+e.message}),1500)}
async function connectFolder(){try{dir=await window.showDirectoryPicker({id:'visionbill',mode:'readwrite'})}catch(e){return}await kvSet0('dirHandle',dir);
 const has=await exists(dir,'products.json');if(has&&items.length&&!confirm(`Folder “${dir.name}” already has a saved database. Load it (replaces products in this browser)? Cancel = overwrite the folder with this browser's products.`)){await saveToFolder()}else await loadFromFolder();updateFolderUI()}
function updateFolderUI(){$('#folderBtn').hidden=!hasFS;$('#folderBtn').textContent=dir?'📂 Change folder':'📂 Connect folder';if(!hasFS)$('#folderInfo').textContent='This browser can’t save to a folder (Chrome/Edge only) – use Export / Import.'}
async function initFolder(){if(!hasFS)return updateFolderUI();const h=await kvGet('dirHandle');if(!h)return updateFolderUI();
 const ok=async()=>{dir=h;await loadFromFolder();updateFolderUI()};
 if(await h.queryPermission({mode:'readwrite'})==='granted')return ok();
 $('#folderInfo').innerHTML='';const b=document.createElement('button');b.className='primary';b.textContent=`🔓 Reconnect folder “${h.name}”`;b.onclick=async()=>{if(await h.requestPermission({mode:'readwrite'})==='granted'){b.remove();await ok()}};$('#folderInfo').append(b);updateFolderUI()}
$('#folderBtn').onclick=connectFolder;
$('#resetBtn').onclick=async()=>{if(!confirm('⚠️ Reset database?\n\nThis permanently deletes ALL products, prices, training photos, learned embeddings, background examples, settings and the current bill from this browser.'))return;
 if(!confirm('Are you sure? This cannot be undone (unless you exported a backup).'))return;
 let clearFolder=false;if(dir)clearFolder=confirm(`A folder “${dir.name}” is connected. Also delete the saved database files in that folder?\n\nOK = delete them too · Cancel = keep the files and disconnect the folder`);
 loadingSnap=true;clearTimeout(saveT);
 try{if(dir&&clearFolder){for(const n of ['products.json','embeddings.json','settings.json'])await dir.removeEntry(n).catch(()=>{});await dir.removeEntry('photos',{recursive:true}).catch(()=>{})}
  await req(tx('items','readwrite').clear());await req(tx('kv','readwrite').clear());dir=null;
  Object.keys(localStorage).filter(k=>k.startsWith('vb_')).forEach(k=>localStorage.removeItem(k))}finally{loadingSnap=false}
 items=[];negs=[];bill=[];scans=[];rebuildBank();renderScans();await refresh();updateFolderUI();$('#thr').value=0.5;$('#thrVal').textContent='0.50';
 $('#folderInfo').textContent='🗑️ Database reset. Everything was cleared'+(clearFolder?' (including the folder files).':'.');$('#samplesInfo').innerHTML='';
 const b=document.createElement('button');b.className='primary';b.textContent='📦 Load fresh sample set';b.onclick=()=>{b.remove();$('#samplesBtn').click()};$('#samplesInfo').append(b);
 if(confirm('Database cleared. Load the fresh sample set now?')){b.remove();$('#samplesBtn').click()}};window.VB.useDir=async h=>{dir=h;await kvSet0('dirHandle',h);await saveToFolder();updateFolderUI()};
$('#exportBtn').onclick=async()=>{const s=await snapshot();s.items.forEach(it=>it.photos.forEach(p=>{if(p.emb)p.emb=p.emb.map(v=>v.map(r4))}));s.negs=s.negs.map(v=>v.map(r4));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(s)],{type:'application/json'}));a.download=`visionbill-db-${new Date().toISOString().slice(0,10)}.json`;document.body.append(a);a.click();a.remove()};
$('#importBtn').onclick=()=>$('#importFile').click();
$('#importFile').onchange=async e=>{const f=e.target.files[0];e.target.value='';if(!f)return;try{const s=JSON.parse(await f.text());if(items.length&&!confirm(`Replace ${items.length} products in this browser with ${s.items.length} from the file?`))return;await applySnapshot(s);scheduleSave();$('#folderInfo').textContent=`✔ Imported ${s.items.length} products from ${f.name}`}catch(err){alert('Import failed: '+err.message)}};
// ---------- init ----------
(async()=>{await openDB();bill=JSON.parse(localStorage.getItem('vb_bill')||'[]');$('#gst').value=localStorage.getItem('vb_gst')||0;
 $('#thr').value=localStorage.getItem('vb_thr2')||0.5;$('#thrVal').textContent=(+$('#thr').value).toFixed(2);$('#showUnknown').checked=localStorage.getItem('vb_showU')!=='0';
 negs=(await kvGet('negs'))||[];if(negs[0]&&negs[0].length!==EMB_DIM)negs=[];await refresh();await initFolder().catch(e=>console.warn('folder',e));
 try{status('Loading AI models…');await loadModels();status('Ready ✔');queue(()=>trainAll())}catch(e){console.error(e);status('Model load failed: '+e.message)}})();

// ---------- v11: One-at-a-time mode ----------
const setMode=m=>{localStorage.setItem('vb_mode',m);$('#singleMode').hidden=m!=='single';$('#multiMode').hidden=m==='single';$('#modeSingle').checked=m==='single';$('#modeMulti').checked=m!=='single'};
document.querySelectorAll('input[name=mode]').forEach(r=>r.onchange=()=>setMode(r.value));setMode(localStorage.getItem('vb_mode')||'single');
let sstream=null,sN=0;const sv=$('#scam');
async function startSingleCam(){if(sstream){sstream.getTracks().forEach(t=>t.stop());sstream=null;sv.srcObject=null;$('#bigSnap').disabled=true;$('#camOff').hidden=false;$('#scamBtn').textContent='🎥 Start camera';return}
 try{sstream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false});sv.srcObject=sstream;await sv.play();$('#bigSnap').disabled=false;$('#camOff').hidden=true;$('#scamBtn').textContent='⏹ Stop camera';
  setTimeout(()=>{if(!emptyRef)learnEmpty(true)},1200)}catch(err){alert('Camera unavailable: '+err.message+'\nYou can still use “Photo(s)”.')}}
$('#scamBtn').onclick=startSingleCam;window.VB.startSingleCam=startSingleCam;
function addSingle(cv,full){const s={n:++sN,thumb:toCanvas(cv,240).toDataURL('image/jpeg',.8),status:'queued',det:null};singles.push(s);renderSingles();queue(()=>processSingle(s,cv,full));return s}
function captureSingle(how='button'){if(!sv.videoWidth)return;if(needModels())return;if(how==='button'&&emptyRef&&sdiff(frameSig(),emptyRef)<SAME){status('Table looks empty – place a product first');badge('Table looks empty');return}const full=toCanvas(sv,2048),cv=toCanvas(sv,1280);const s=addSingle(cv,full);s.how=how;
 const b=$('#bigSnap');b.classList.add('flash');setTimeout(()=>b.classList.remove('flash'),250);lastSig=frameSig();}
$('#bigSnap').onclick=()=>captureSingle('button');
$('#singlePhotos').onchange=async e=>{if(needModels())return;const fs=[...e.target.files];e.target.value='';for(const f of fs){const {cv,full}=await decodePhoto(f);addSingle(cv,full)}};
// only one product per photo: take the biggest, most central detection; fall back to a centre crop
async function processSingle(s,cv,full){s.status='processing';renderSingles();const t=performance.now();
 let boxes=[];try{boxes=await V.segment(det,raw(cv),{extra:items.map(i=>i.name)})}catch(e){console.warn(e)}
 const W=cv.width,H=cv.height,A=W*H;const cand=boxes.filter(b=>!b.big&&b.box[2]*b.box[3]>A*0.01);
 const rank=b=>{const [x,y,w,h]=b.box,dx=(x+w/2)/W-0.5,dy=(y+h/2)/H-0.5;return Math.sqrt(w*h/A)*(1-Math.min(0.9,Math.hypot(dx,dy)*1.4))};
 let box;if(cand.length){box=cand.sort((a,b)=>rank(b)-rank(a))[0].box;s.fallback=false}else{box=[W*0.2,H*0.2,W*0.6,H*0.6];s.fallback=true}
 const sc0=full.width/cv.width;let vs=viewCanvases(full,box,sc0,'fast');let es=await V.embedBatch(fe,vs.map(raw));const col=colOfCanvas(vs[0]);let r=V.decideFused(V.scoreViews(bank,es),cols,col,{},thr(),MARGIN);
 if(r.conf<0.7||r.margin<0.05){const ex=viewCanvases(full,box,sc0,'extra');es=[...es,...await V.embedBatch(fe,ex.map(raw))];r=V.decideFused(V.scoreViews(bank,es),cols,col,{},thr(),MARGIN);s.closer=true}
 const d={box,es,col,crop:cropURL(full,box.map(v=>v*sc0)),...r};d.manual=false;s.det=d;s.status='done';s.ms=Math.round(performance.now()-t);renderSingles()}
function renderSingles(){const root=$('#singleList');root.innerHTML='';const pct=v=>Math.round(v*100)+'%';
 singles.forEach((s,i)=>{const d=s.det,row=document.createElement('div');row.className='det srow'+(d&&d.manual?' ok':'');
  if(!d){row.innerHTML=`<div class="dhead"><span class="n">${s.n}</span><img src="${s.thumb}" class="dthumb"><div class="dmain"><b>Item ${s.n}</b> <span class="tag">${s.status==='processing'?'⏳ recognising…':'🕒 queued'}</span></div><button data-a="del">🗑</button></div>`}
  else{const sugg=d.manual?d.id:(d.id!=='unknown'?d.id:d.guess),sname=sugg&&sugg!=='unknown'?esc(nameOf(sugg)):NIDB,unk=d.id==='unknown';
   const chips=(d.top3||[]).map(t=>`<button class="chip pick ${t.id===d.id&&d.manual?'on':''}" data-id="${t.id}">${esc(nameOf(t.id))} ${pct(t.s)}</button>`).join('');
   row.innerHTML=`<div class="dhead"><span class="n">${s.n}</span><img src="${d.crop}" class="dthumb"><div class="dmain">${d.manual?`<b>Item ${s.n}: ${d.id==='unknown'?'❓ '+NIDB:esc(nameOf(d.id))}</b> <span class="tag okt">✔ confirmed</span>`:`<b>Item ${s.n}: ${unk?'not sure – maybe':'I think this is'} ${sname}?</b> <span class="conf ${unk?'low':''}">${pct(d.conf)}</span> <span class="tag ask">needs your OK</span>`}${s.fallback?' <span class="muted small">(no clear item found – used centre of photo)</span>':''}
    <div class="muted small">Top guesses: ${chips||'—'}</div></div></div>
    <div class="acts">${!d.manual&&sugg&&sugg!=='unknown'?`<button data-a="yes" class="primary">✔ Yes, ${sname}</button>`:''}<select class="spick"><option value="">✏️ ${d.manual?'Change':'Something else'}…</option>${items.map(it=>`<option value="${it.id}">${esc(it.name)} · ${inr(it.price)}</option>`).join('')}<option value="__new">➕ New product…</option><option value="unknown">Not in database</option></select><button data-a="del">🗑</button></div>`;
   row.querySelectorAll('.pick').forEach(b=>b.onclick=()=>{setLabel(d,b.dataset.id,false);renderSingles()});
   const yb=row.querySelector('[data-a=yes]');if(yb)yb.onclick=()=>{setLabel(d,sugg,false);renderSingles()};
   row.querySelector('.spick').onchange=async e=>{const v=e.target.value;if(!v)return;if(v==='__new'){await newProductFromCrop(d);renderSingles();return}if(v==='unknown'){d.id='unknown';d.manual=true;renderSingles();return}setLabel(d,v,false);renderSingles()}}
  row.querySelector('[data-a=del]').onclick=()=>{singles.splice(i,1);renderSingles()};root.append(row)});
 const pend=singles.filter(s=>!s.det).length,unc=singles.filter(s=>s.det&&!s.det.manual).length,ok=singles.filter(s=>s.det&&s.det.manual&&s.det.id!=='unknown').length;
 $('#singleInfo').textContent=singles.length?`${singles.length} captured · ${pend} processing · ${unc} waiting for your OK · ${ok} confirmed`:'Nothing captured yet.'}
$('#sConfirmAll').onclick=()=>{for(const s of singles){const d=s.det;if(!d||d.manual)continue;const g=d.id!=='unknown'?d.id:d.guess;if(g)setLabel(d,g,false)}renderSingles()};
$('#sClear').onclick=()=>{if(singles.length&&!confirm('Clear the captured list?'))return;singles=[];sN=0;renderSingles()};
$('#createBill').onclick=async()=>{const b=$('#createBill');if(singles.some(s=>!s.det)){b.disabled=true;b.textContent='⏳ Waiting for processing…';await busy;b.disabled=false;b.textContent='🧾 Create bill'}
 const unc=singles.filter(s=>s.det&&!s.det.manual);if(unc.length&&!confirm(`${unc.length} item(s) are not confirmed yet and will be LEFT OUT of the bill.\nOK = create bill without them · Cancel = go back and confirm`))return;
 const ok=singles.filter(s=>s.det&&s.det.manual&&s.det.id!=='unknown');if(!ok.length){alert('No confirmed items yet.');return}
 for(const s of ok)addToBill(s.det.id,1);singles=singles.filter(s=>!ok.includes(s));renderSingles();$('#billTable').scrollIntoView({behavior:'smooth'});status(`Bill created: ${ok.length} item(s) added`)};
// auto-capture: frame differencing on a tiny grey thumbnail. motion -> still for N s -> not the empty table -> not the same as last capture
const sigC=document.createElement('canvas');sigC.width=32;sigC.height=24;let prevSig=null,lastSig=null,emptyRef=null,stillSince=0,armed=true;
function frameSig(){if(!sv.videoWidth)return null;const g=sigC.getContext('2d',{willReadFrequently:true});g.drawImage(sv,0,0,32,24);const d=g.getImageData(0,0,32,24).data,o=new Float32Array(768);let m=0;for(let i=0;i<768;i++){o[i]=(d[i*4]*0.3+d[i*4+1]*0.59+d[i*4+2]*0.11);m+=o[i]}m/=768;for(let i=0;i<768;i++)o[i]-=m;return o}// brightness-normalised
const sdiff=(a,b)=>{if(!a||!b)return 999;let s=0;for(let i=0;i<a.length;i++)s+=Math.abs(a[i]-b[i]);return s/a.length};
function learnEmpty(silent){emptyRef=frameSig();lastSig=null;if(!silent)status('Empty table learned ✔')}
$('#emptyBtn').onclick=()=>learnEmpty(false);
const badge=t=>{const b=$('#autoBadge');b.hidden=!t;b.textContent=t||''};
$('#autoCap').checked=localStorage.getItem('vb_auto')==='1';$('#autoDelay').value=localStorage.getItem('vb_autoDelay')||2;
$('#autoCap').onchange=()=>{localStorage.setItem('vb_auto',$('#autoCap').checked?1:0);armed=true;stillSince=0;if(!$('#autoCap').checked)badge('')};$('#autoDelay').onchange=()=>localStorage.setItem('vb_autoDelay',$('#autoDelay').value);
const MOVE=6,SAME=9;window.VB.auto={get emptyRef(){return emptyRef},MOVE,SAME};
setInterval(()=>{if(!sstream||!$('#autoCap').checked||$('#singleMode').hidden||!det)return;const f=frameSig();if(!f)return;const mv=sdiff(f,prevSig);prevSig=f;const now=performance.now(),need=(+$('#autoDelay').value||2)*1000;
 const isEmpty=emptyRef&&sdiff(f,emptyRef)<SAME;if(isEmpty)lastSig=null;// table cleared: the same product may be placed again (2nd pack)
 if(mv>MOVE){stillSince=0;armed=true;badge('👋 Movement…');return}
 if(!stillSince)stillSince=now;if(isEmpty){badge('Empty table – place a product');return}
 if(lastSig&&sdiff(f,lastSig)<SAME){badge('✔ Already captured – place the next product');return}
 if(!armed){badge('Waiting for a change…');return}
 const left=need-(now-stillSince);if(left>0){badge(`Hold still… ${(left/1000).toFixed(1)} s`);return}
 captureSingle('auto');armed=false;stillSince=0;badge(`📸 Captured item ${sN}`)},200);
