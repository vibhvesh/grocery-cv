import * as V from './vision.js';
const TJS='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';
// ---------- IndexedDB ----------
let db;function openDB(){return new Promise((res,rej)=>{const r=indexedDB.open('visionbill',1);r.onupgradeneeded=()=>{const d=r.result;d.createObjectStore('items',{keyPath:'id'});d.createObjectStore('kv')};r.onsuccess=()=>{db=r.result;res()};r.onerror=()=>rej(r.error)})}
const tx=(s,m='readonly')=>db.transaction(s,m).objectStore(s),req=r=>new Promise((a,b)=>{r.onsuccess=()=>a(r.result);r.onerror=()=>b(r.error)});
const getItems=()=>req(tx('items').getAll()),putItem=i=>req(tx('items','readwrite').put(i)),delItem=id=>req(tx('items','readwrite').delete(id));
const kvGet=k=>req(tx('kv').get(k)),kvSet=(k,v)=>req(tx('kv','readwrite').put(v,k));
// ---------- state / utils ----------
const $=s=>document.querySelector(s);let T,det,fe,items=[],negs=[],bank=[],pendingPhotos=[],bill=[],scans=[],busy=Promise.resolve();
window.VB={get items(){return items},get scans(){return scans},get ready(){return !!(det&&fe)}};
const EMB_DIM=384,inr=n=>'₹'+(+n).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
const status=t=>$('#status').textContent=t,esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const thr=()=>+$('#thr').value,NIDB='Not in database',nameOf=id=>id==='unknown'?NIDB:(items.find(i=>i.id==id)||{}).name||'?';
const queue=fn=>busy=busy.then(fn).catch(e=>{console.error(e);status('Error: '+e.message)});
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,.tab').forEach(e=>e.classList.remove('active'));b.classList.add('active');$('#'+b.dataset.tab).classList.add('active')});
const loadImg=src=>new Promise((a,b)=>{const i=new Image();i.onload=()=>a(i);i.onerror=b;i.src=src});
function toCanvas(src,max=1280){const W=src.naturalWidth||src.videoWidth||src.width,H=src.naturalHeight||src.videoHeight||src.height,s=Math.min(1,max/Math.max(W,H));const c=document.createElement('canvas');c.width=Math.round(W*s);c.height=Math.round(H*s);c.getContext('2d').drawImage(src,0,0,c.width,c.height);return c}
function cropURL(cv,[x,y,w,h],pad=0.1,size=224){const s=Math.max(w,h)*(1+2*pad),cx=x+w/2,cy=y+h/2,c=document.createElement('canvas');c.width=c.height=size;const g=c.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,size,size);const k=size/s;g.drawImage(cv,(cx-s/2),(cy-s/2),s,s,0,0,size,size);return c.toDataURL('image/jpeg',.9)}
const raw=cv=>T.RawImage.fromCanvas(cv);
// ---------- models ----------
async function loadModels(){const hi=localStorage.getItem('vb_hi')==='1';$('#hiAcc').checked=hi;
 T=await import(TJS);T.env.allowLocalModels=false;
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
function rebuildBank(){bank=[];for(const it of items)for(const p of it.photos)for(const e of p.emb||[])bank.push({l:it.id,e});for(const e of negs)bank.push({l:V.BG,e})}
async function embedPhoto(it,p){const im=await loadImg(p.src),cv=toCanvas(im);
 if(p.kind==='crop'){p.emb=await V.embedCrop(fe,raw(cv),[0,0,cv.width,cv.height],true);p.thumb=p.src;return}
 const r=await V.trainPhoto(det,fe,raw(cv),{name:it.name});p.emb=r.pos;p.box=r.box;p.thumb=cropURL(cv,r.box,0.1,128);
 // negatives: scene objects that don't look like any known product
 const kept=r.neg.filter(e=>V.classify(bank.filter(b=>b.l!==V.BG),e,0).conf<0.8);negs.push(...kept);if(negs.length>400)negs=negs.slice(-400);await kvSet('negs',negs)}
async function trainItem(it,force=false){let n=0;for(const p of it.photos)if(force||!p.emb||(p.emb[0]&&p.emb[0].length!==EMB_DIM)){status(`Learning ${it.name} (${++n})…`);await embedPhoto(it,p)}
 if(n){await putItem(it);rebuildBank();reclassifyScans()}return n}
async function trainAll(force=false){if(!det)return;if(force){negs=[];await kvSet('negs',negs)}let n=0;for(const it of items)n+=await trainItem(it,force);status('Ready ✔');
 $('#trainInfo').textContent=`Model: ${bank.filter(b=>b.l!==V.BG).length} product embeddings from ${items.reduce((s,i)=>s+i.photos.length,0)} photos · ${negs.length} background examples. Updates automatically.`}
$('#samplesBtn').onclick=async()=>{const btn=$('#samplesBtn');btn.disabled=true;status('Loading samples…');
 const man=await (await fetch('samples/manifest.json')).json();const toURL=async u=>{const b=await (await fetch(u)).blob();return new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result);f.readAsDataURL(b)})};
 let n=0;for(const m of man){if(items.some(i=>i.name===m.name))continue;const photos=[];for(const u of m.photos)photos.push({src:await toURL(u),kind:'crop'});
  await putItem({id:crypto.randomUUID(),name:m.name,price:m.price,category:m.category,unit:m.unit,notes:m.notes,photos});n++}
 await refresh();btn.disabled=false;status(n?`Added ${n} sample products – learning…`:'Samples already loaded');queue(()=>trainAll())};
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
async function addCropTo(id,d,kind='crop'){const it=items.find(i=>i.id==id);if(!it)return;it.photos.push({src:d.crop,kind:'crop',emb:await V.embedCrop(fe,raw(toCanvas(await loadImg(d.crop))),[0,0,224,224],true),thumb:d.crop,from:kind});
 await putItem(it);await refresh();reclassifyScans()}
async function newProductFromCrop(d){const name=prompt('New product name:');if(!name)return;const price=+prompt(`Price of ${name} in ₹:`,'0')||0;
 const it={id:crypto.randomUUID(),name:name.trim(),price,category:'',unit:'',notes:'',photos:[]};await putItem(it);items.push(it);await addCropTo(it.id,d,'new');d.id=it.id;d.manual=true;renderScans()}
function reclassifyScans(){for(const sc of scans)for(const d of sc.dets){if(d.manual&&items.find(i=>i.id==d.id))continue;Object.assign(d,V.classify(bank,d.e,thr()))}renderScans()}
// ---------- scanning ----------
async function scanCanvas(cv){const t0=performance.now();status('Step 1/2: finding items…');const R=raw(cv);
 const boxes=await V.segment(det,R,{extra:items.map(i=>i.name)});const sc={cv,dets:[],sel:-1};
 status(`Step 2/2: recognising ${boxes.length} crops…`);
 for(const b of boxes){const d=await makeDet(cv,R,b.box);if(b.big&&d.id==='unknown')continue;sc.dets.push(d)}
 sc.dets=V.dedupe(sc.dets);sc.ms=Math.round(performance.now()-t0);scans.push(sc);renderScans();status('Ready ✔')}
async function makeDet(cv,R,box){const [e]=await V.embedCrop(fe,R,box);return{box,e,crop:cropURL(cv,box),...V.classify(bank,e,thr())}}
const COLORS=['#2563eb','#16a34a','#d97706','#7c3aed','#0891b2','#db2777','#65a30d','#0f766e'];
function renderScans(){const root=$('#scanResults');root.innerHTML='';const showU=$('#showUnknown').checked;
 scans.forEach((sc,si)=>{const wrap=document.createElement('div');wrap.className='scanimg';const W=sc.cv.width,H=sc.cv.height;
  const cv=document.createElement('canvas');cv.className='prev';cv.width=W;cv.height=H;const g=cv.getContext('2d');g.drawImage(sc.cv,0,0);const lw=Math.max(2,W/300),fs=Math.max(14,W/40);
  sc.dets.forEach((d,i)=>{if(d.id==='unknown'&&!showU)return;const col=d.id==='unknown'?'#dc2626':COLORS[i%COLORS.length];g.lineWidth=i==sc.sel?lw*2.5:lw;g.strokeStyle=col;g.strokeRect(...d.box);g.font=`bold ${fs}px sans-serif`;const t=`${i+1}. ${d.id==='unknown'?'?':nameOf(d.id)}`;g.fillStyle=col;g.fillRect(d.box[0],d.box[1],g.measureText(t).width+10,fs*1.3);g.fillStyle='#fff';g.fillText(t,d.box[0]+5,d.box[1]+fs)});
  let start=null;const pt=e=>{const r=cv.getBoundingClientRect();return[(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height]};
  cv.onpointerdown=e=>{start=pt(e);cv.setPointerCapture(e.pointerId)};
  cv.onpointerup=e=>{if(!start)return;const p=pt(e),b=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])];const s0=start;start=null;
   if(b[2]<W/40||b[3]<H/40){sc.sel=sc.dets.findIndex(d=>s0[0]>=d.box[0]&&s0[0]<=d.box[0]+d.box[2]&&s0[1]>=d.box[1]&&s0[1]<=d.box[1]+d.box[3]);return renderScans()}
   queue(async()=>{sc.dets.push(await makeDet(sc.cv,raw(sc.cv),b));renderScans()})};
  wrap.append(cv);let hidden=0;
  sc.dets.forEach((d,i)=>{if(d.id==='unknown'&&!showU){hidden++;return}const unk=d.id==='unknown';const row=document.createElement('div');row.className='det'+(i==sc.sel?' sel':'')+(unk?' unk':'');
   row.innerHTML=`<span class="n">${i+1}</span><img src="${d.crop}" width="56" height="56" style="border-radius:6px"><select><option value="unknown">❓ ${NIDB}</option>${items.map(it=>`<option value="${it.id}">${esc(it.name)}</option>`).join('')}</select><span class="conf ${unk?'low':''}" title="similarity to nearest training example${d.guess?' (closest: '+esc(nameOf(d.guess))+')':''}">${Math.round(d.conf*100)}%</span><input type="number" min="1" value="${d.qty||1}" style="width:52px"><div class="acts"><button data-a="new">➕ New product</button>${unk?'':`<button data-a="teach" title="save this crop as a training photo of ${esc(nameOf(d.id))}">💾 Add photo to ${esc(nameOf(d.id))}</button>`}<button data-a="del" title="delete box">🗑</button></div>${unk&&d.guess?`<span class="tag">closest: ${esc(nameOf(d.guess))}</span>`:''}`;
   const sel=row.querySelector('select');sel.value=d.id;
   sel.onchange=()=>{d.id=sel.value;d.manual=true;if(d.id!=='unknown')queue(()=>addCropTo(d.id,d,'correction'));renderScans()};
   row.querySelector('input').onchange=e=>{d.qty=Math.max(1,+e.target.value||1);renderSummary()};
   row.querySelector('[data-a=new]').onclick=()=>queue(()=>newProductFromCrop(d));
   const tb=row.querySelector('[data-a=teach]');if(tb)tb.onclick=()=>{tb.disabled=true;tb.textContent='✔ Saved';queue(()=>addCropTo(d.id,d,'confirm'))};
   row.querySelector('[data-a=del]').onclick=()=>{sc.dets.splice(i,1);sc.sel=-1;renderScans()};
   row.onclick=e=>{if(e.target===row||e.target.className=='n'){sc.sel=i;renderScans()}};wrap.append(row)});
  const ctl=document.createElement('div');ctl.className='row';ctl.innerHTML=`<span class="muted">${sc.dets.length} item(s) found${hidden?`, ${hidden} not-in-database hidden`:''} · ${sc.ms} ms</span><button>Remove photo</button>`;ctl.querySelector('button').onclick=()=>{scans.splice(si,1);renderScans()};wrap.append(ctl);root.append(wrap)});
 renderSummary()}
function counts(){const c={};scans.forEach(s=>s.dets.forEach(d=>c[d.id]=(c[d.id]||0)+(d.qty||1)));return c}
function renderSummary(){const c=counts();$('#summaryBox').hidden=!scans.length;$('#summary').innerHTML=Object.entries(c).sort((a,b)=>(a[0]==='unknown')-(b[0]==='unknown')).map(([id,n])=>`<span class="chip ${id==='unknown'?'unk':''}">${esc(nameOf(id))} × ${n}</span>`).join('')||'<span class="muted">Nothing detected</span>'}
$('#addAll').onclick=()=>{const c=counts();for(const [id,n] of Object.entries(c))if(id!=='unknown')addToBill(id,n);scans=[];renderScans()};
$('#clearScans').onclick=()=>{scans=[];renderScans()};$('#showUnknown').onchange=()=>{localStorage.setItem('vb_showU',$('#showUnknown').checked?1:0);renderScans()};
$('#thr').oninput=()=>{$('#thrVal').textContent=(+$('#thr').value).toFixed(2);localStorage.setItem('vb_thr2',$('#thr').value);reclassifyScans()};
$('#hiAcc').onchange=()=>{localStorage.setItem('vb_hi',$('#hiAcc').checked?1:0);if(confirm('Reload now to switch detector?'))location.reload()};
const needModels=()=>{if(!det){alert('Models are still loading…');return true}};
$('#billPhotos').onchange=async e=>{if(needModels())return;const fs=[...e.target.files];e.target.value='';for(const f of fs){const cv=toCanvas(await loadImg(URL.createObjectURL(f)));queue(()=>scanCanvas(cv))}};
let stream;$('#camBtn').onclick=async()=>{const v=$('#cam');if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;v.hidden=$('#snapBtn').hidden=true;return}
 try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280}}});v.srcObject=stream;await v.play();v.hidden=$('#snapBtn').hidden=false}catch(err){alert('Camera unavailable: '+err.message)}};
$('#snapBtn').onclick=()=>{if(needModels())return;const cv=toCanvas($('#cam'));queue(()=>scanCanvas(cv))};
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
// ---------- init ----------
(async()=>{await openDB();bill=JSON.parse(localStorage.getItem('vb_bill')||'[]');$('#gst').value=localStorage.getItem('vb_gst')||0;
 $('#thr').value=localStorage.getItem('vb_thr2')||0.55;$('#thrVal').textContent=(+$('#thr').value).toFixed(2);$('#showUnknown').checked=localStorage.getItem('vb_showU')!=='0';
 negs=(await kvGet('negs'))||[];if(negs[0]&&negs[0].length!==EMB_DIM)negs=[];await refresh();
 try{status('Loading AI models…');await loadModels();status('Ready ✔');queue(()=>trainAll())}catch(e){console.error(e);status('Model load failed: '+e.message)}})();
