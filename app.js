// ---------- IndexedDB ----------
const DB='visionbill';let db;
function openDB(){return new Promise((res,rej)=>{const r=indexedDB.open(DB,1);r.onupgradeneeded=()=>{const d=r.result;d.createObjectStore('items',{keyPath:'id'});d.createObjectStore('kv')};r.onsuccess=()=>{db=r.result;res()};r.onerror=()=>rej(r.error)})}
const tx=(s,m='readonly')=>db.transaction(s,m).objectStore(s);
const req=r=>new Promise((a,b)=>{r.onsuccess=()=>a(r.result);r.onerror=()=>b(r.error)});
const getItems=()=>req(tx('items').getAll());const putItem=i=>req(tx('items','readwrite').put(i));const delItem=id=>req(tx('items','readwrite').delete(id));
const kvGet=k=>req(tx('kv').get(k));const kvSet=(k,v)=>req(tx('kv','readwrite').put(v,k));
// ---------- state ----------
const $=s=>document.querySelector(s);let items=[],net,knn,coco,pendingPhotos=[],bill=[];
const inr=n=>'₹'+(+n).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
const status=t=>$('#status').textContent=t;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,.tab').forEach(e=>e.classList.remove('active'));b.classList.add('active');$('#'+b.dataset.tab).classList.add('active')});
function loadImg(src){return new Promise((a,b)=>{const i=new Image();i.onload=()=>a(i);i.onerror=b;i.src=src})}
function fileToDataURL(f,max=320){return new Promise(a=>{const r=new FileReader();r.onload=async()=>{const i=await loadImg(r.result);const s=Math.min(1,max/Math.max(i.width,i.height));const c=document.createElement('canvas');c.width=i.width*s;c.height=i.height*s;c.getContext('2d').drawImage(i,0,0,c.width,c.height);a(c.toDataURL('image/jpeg',.85))};r.readAsDataURL(f)})}
// ---------- products ----------
$('#iphotos').onchange=async e=>{for(const f of e.target.files)pendingPhotos.push(await fileToDataURL(f));e.target.value='';renderPending()};
function renderPending(){$('#previewPhotos').innerHTML=pendingPhotos.map((p,i)=>`<img src="${p}" title="click to remove" data-i="${i}">`).join('');$('#previewPhotos').querySelectorAll('img').forEach(im=>im.onclick=()=>{pendingPhotos.splice(+im.dataset.i,1);renderPending()})}
function resetForm(){$('#itemForm').reset();$('#itemId').value='';pendingPhotos=[];renderPending()}
$('#resetForm').onclick=resetForm;
$('#itemForm').onsubmit=async e=>{e.preventDefault();const id=$('#itemId').value||crypto.randomUUID();
 await putItem({id,name:$('#iname').value.trim(),price:+$('#iprice').value,category:$('#icat').value,unit:$('#iunit').value,notes:$('#inotes').value,photos:pendingPhotos});
 resetForm();await refresh();$('#trainInfo').textContent='Products changed – click Train model to update.'};
async function refresh(){items=await getItems();items.sort((a,b)=>a.name.localeCompare(b.name));
 $('#itemList').innerHTML=items.length?items.map(i=>`<div class="item"><div class="thumbs">${i.photos.slice(0,3).map(p=>`<img src="${p}">`).join('')}</div><div class="info"><b>${esc(i.name)}</b> – ${inr(i.price)}<div class="muted">${esc(i.category)} · ${esc(i.unit)} · ${i.photos.length} photos ${i.notes?'· '+esc(i.notes):''}</div></div><button data-e="${i.id}">Edit</button><button data-d="${i.id}">🗑</button></div>`).join(''):'<p class="muted">No products yet. Add at least 2 products with 3–10 photos each.</p>';
 $('#itemList').querySelectorAll('[data-e]').forEach(b=>b.onclick=()=>{const i=items.find(x=>x.id==b.dataset.e);$('#itemId').value=i.id;$('#iname').value=i.name;$('#iprice').value=i.price;$('#icat').value=i.category;$('#iunit').value=i.unit;$('#inotes').value=i.notes;pendingPhotos=[...i.photos];renderPending();scrollTo(0,0)});
 $('#itemList').querySelectorAll('[data-d]').forEach(b=>b.onclick=async()=>{if(confirm('Delete product?')){await delItem(b.dataset.d);refresh()}});
 $('#manualItem').innerHTML=items.map(i=>`<option value="${i.id}">${esc(i.name)} (${inr(i.price)})</option>`).join('');renderBill()}
// ---------- model ----------
const embed=el=>tf.tidy(()=>{const e=net.infer(el,true);return e.div(e.norm())});
async function augmentedEmbeds(img){const out=[];const c=document.createElement('canvas');c.width=c.height=224;const g=c.getContext('2d');
 for(const [flip,zoom] of [[0,1],[1,1],[0,.8]]){g.save();g.clearRect(0,0,224,224);if(flip){g.translate(224,0);g.scale(-1,1)}const w=img.width*zoom,h=img.height*zoom;g.drawImage(img,(img.width-w)/2,(img.height-h)/2,w,h,0,0,224,224);g.restore();out.push(embed(c))}return out}
$('#trainBtn').onclick=async()=>{if(!net)return alert('Model still loading');const ok=items.filter(i=>i.photos.length);if(!ok.length)return alert('Add products with photos first');
 $('#trainBtn').disabled=true;knn.clearAllClasses();const saved=[];let n=0,tot=ok.reduce((s,i)=>s+i.photos.length,0);
 for(const it of ok)for(const p of it.photos){const im=await loadImg(p);for(const e of await augmentedEmbeds(im)){knn.addExample(e,it.id);saved.push({l:it.id,e:Array.from(await e.data())});e.dispose()}status(`Training ${++n}/${tot}`);await tf.nextFrame()}
 await kvSet('embeddings',saved);$('#trainBtn').disabled=false;status('Model ready ✔');$('#trainInfo').textContent=`Trained on ${tot} photos (${saved.length} embeddings incl. augmentation) for ${ok.length} products. Saved.`};
async function restoreModel(){const s=await kvGet('embeddings');if(!s)return;for(const {l,e} of s){if(!items.find(i=>i.id==l))continue;const t=tf.tensor1d(e);knn.addExample(t,l);t.dispose()}$('#trainInfo').textContent=`Restored trained model (${s.length} embeddings).`}
async function classify(el){if(!knn.getNumClasses())return null;const e=embed(el);const k=Math.min(5,Object.values(knn.getClassExampleCount()).reduce((a,b)=>a+b,0));const r=await knn.predictClass(e,k);e.dispose();return{id:r.label,conf:r.confidences[r.label]}}
// ---------- billing ----------
async function scanImage(img){const res=[];const crops=[];
 if($('#multiDetect').checked&&coco){const dets=(await coco.detect(img,10,.4)).filter(d=>d.bbox[2]*d.bbox[3]<img.width*img.height*.95&&d.class!=='person'&&d.class!=='dining table');
  for(const d of dets){let [x,y,w,h]=d.bbox;const p=.08;x=Math.max(0,x-w*p);y=Math.max(0,y-h*p);w=Math.min(img.width-x,w*(1+2*p));h=Math.min(img.height-y,h*(1+2*p));crops.push([x,y,w,h])}}
 if(!crops.length)crops.push([0,0,img.width,img.height]);
 for(const [x,y,w,h] of crops){const c=document.createElement('canvas');c.width=c.height=224;c.getContext('2d').drawImage(img,x,y,w,h,0,0,224,224);const r=await classify(c);res.push({canvas:c,...r})}
 // merge same predicted item into quantity
 return res}
async function handleImages(imgs){if(!knn||!knn.getNumClasses())return alert('Train the model first (Products tab).');status('Scanning…');
 for(const img of imgs)for(const r of await scanImage(img)){const row=document.createElement('div');row.className='scan';const low=r.conf<.6;
  row.innerHTML=`<span></span><span class="conf ${low?'low':''}">${Math.round(r.conf*100)}%</span><select>${items.map(i=>`<option value="${i.id}" ${i.id==r.id?'selected':''}>${esc(i.name)}</option>`).join('')}</select><input type="number" min="1" value="1" style="width:60px"><button class="primary">Add</button><button>✕</button>`;
  row.firstChild.replaceWith(r.canvas);const [sel]=row.getElementsByTagName('select'),[q]=row.getElementsByTagName('input'),[add,rm]=row.getElementsByTagName('button');
  add.onclick=()=>{addToBill(sel.value,+q.value||1);row.remove()};rm.onclick=()=>row.remove();$('#scanResults').prepend(row)}
 status('Model ready ✔')}
$('#billPhotos').onchange=async e=>{const imgs=[];for(const f of e.target.files)imgs.push(await loadImg(URL.createObjectURL(f)));e.target.value='';handleImages(imgs)};
let stream;$('#camBtn').onclick=async()=>{const v=$('#cam');if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;v.hidden=$('#snapBtn').hidden=true;return}
 try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});v.srcObject=stream;await v.play();v.hidden=$('#snapBtn').hidden=false}catch(err){alert('Camera unavailable: '+err.message)}};
$('#snapBtn').onclick=()=>{const v=$('#cam'),c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);handleImages([c])};
function addToBill(id,q){const l=bill.find(b=>b.id==id);l?l.qty+=q:bill.push({id,qty:q});saveBill()}
function saveBill(){localStorage.setItem('vb_bill',JSON.stringify(bill));localStorage.setItem('vb_gst',$('#gst').value);renderBill()}
function totals(){const lines=bill.map(b=>{const i=items.find(x=>x.id==b.id);return i&&{...b,name:i.name,unit:i.unit,price:i.price,total:i.price*b.qty}}).filter(Boolean);const sub=lines.reduce((s,l)=>s+l.total,0),g=sub*(+$('#gst').value||0)/100;return{lines,sub,g,grand:sub+g}}
function renderBill(){const t=totals();$('#billDate').textContent=new Date().toLocaleString('en-IN');
 $('#billTable tbody').innerHTML=t.lines.map((l,i)=>`<tr><td>${esc(l.name)}<div class="muted">${esc(l.unit)}</div></td><td><input type="number" min="0" step="any" value="${l.qty}" data-q="${l.id}"></td><td>${inr(l.price)}</td><td>${inr(l.total)}</td><td class="np"><button data-r="${l.id}">✕</button></td></tr>`).join('')||'<tr><td colspan="5" class="muted">No items yet</td></tr>';
 document.querySelectorAll('[data-q]').forEach(e=>e.onchange=()=>{const b=bill.find(x=>x.id==e.dataset.q);b.qty=+e.value;if(b.qty<=0)bill=bill.filter(x=>x!==b);saveBill()});
 document.querySelectorAll('[data-r]').forEach(e=>e.onclick=()=>{bill=bill.filter(x=>x.id!=e.dataset.r);saveBill()});
 $('#subtotal').textContent=inr(t.sub);$('#gstAmt').textContent=inr(t.g);$('#grand').textContent=inr(t.grand)}
$('#gst').oninput=saveBill;$('#addManual').onclick=()=>$('#manualItem').value&&addToBill($('#manualItem').value,1);$('#clearBill').onclick=()=>{bill=[];saveBill();$('#scanResults').innerHTML=''};
$('#pdfBtn').onclick=()=>{const {jsPDF}=window.jspdf,d=new jsPDF(),t=totals(),R=n=>'Rs. '+n.toFixed(2);
 d.setFontSize(16);d.text('VisionBill – Grocery Invoice',14,16);d.setFontSize(10);d.text(new Date().toLocaleString('en-IN'),14,23);
 d.autoTable({startY:28,head:[['#','Item','Qty','Unit price','Total']],body:t.lines.map((l,i)=>[i+1,l.name+(l.unit?` (${l.unit})`:''),l.qty,R(l.price),R(l.total)]),
  foot:[['','','','Subtotal',R(t.sub)],['','','',`GST ${$('#gst').value||0}%`,R(t.g)],['','','','Grand total',R(t.grand)]],theme:'grid',headStyles:{fillColor:[22,163,74]},footStyles:{fillColor:[240,240,240],textColor:20}});
 d.save(`bill-${Date.now()}.pdf`)};
// ---------- init ----------
(async()=>{await openDB();bill=JSON.parse(localStorage.getItem('vb_bill')||'[]');$('#gst').value=localStorage.getItem('vb_gst')||0;await refresh();
 try{await tf.ready();net=await mobilenet.load({version:2,alpha:1});knn=knnClassifier.create();await restoreModel();status(knn.getNumClasses()?'Model ready ✔':'Feature extractor ready – train model');
  cocoSsd.load({base:'lite_mobilenet_v2'}).then(m=>coco=m).catch(e=>console.warn('COCO-SSD failed',e))}catch(e){status('Model load failed');console.error(e)}})();
