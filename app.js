// ---------- IndexedDB ----------
const DB='visionbill';let db;
function openDB(){return new Promise((res,rej)=>{const r=indexedDB.open(DB,1);r.onupgradeneeded=()=>{const d=r.result;d.createObjectStore('items',{keyPath:'id'});d.createObjectStore('kv')};r.onsuccess=()=>{db=r.result;res()};r.onerror=()=>rej(r.error)})}
const tx=(s,m='readonly')=>db.transaction(s,m).objectStore(s);
const req=r=>new Promise((a,b)=>{r.onsuccess=()=>a(r.result);r.onerror=()=>b(r.error)});
const getItems=()=>req(tx('items').getAll());const putItem=i=>req(tx('items','readwrite').put(i));const delItem=id=>req(tx('items','readwrite').delete(id));
const kvGet=k=>req(tx('kv').get(k));const kvSet=(k,v)=>req(tx('kv','readwrite').put(v,k));
// ---------- state ----------
const $=s=>document.querySelector(s);let items=[],net,coco,pendingPhotos=[],bill=[];
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
 resetForm();await refresh();bank&&buildBank((await kvGet('embeddings'))||[]);$('#trainInfo').textContent='Products changed – click Train model to update.'};
async function refresh(){items=await getItems();items.sort((a,b)=>a.name.localeCompare(b.name));
 $('#itemList').innerHTML=items.length?items.map(i=>`<div class="item"><div class="thumbs">${i.photos.slice(0,3).map(p=>`<img src="${p}">`).join('')}</div><div class="info"><b>${esc(i.name)}</b> – ${inr(i.price)}<div class="muted">${esc(i.category)} · ${esc(i.unit)} · ${i.photos.length} photos ${i.notes?'· '+esc(i.notes):''}</div></div><button data-e="${i.id}">Edit</button><button data-d="${i.id}">🗑</button></div>`).join(''):'<p class="muted">No products yet. Add at least 2 products with 3–10 photos each.</p>';
 $('#itemList').querySelectorAll('[data-e]').forEach(b=>b.onclick=()=>{const i=items.find(x=>x.id==b.dataset.e);$('#itemId').value=i.id;$('#iname').value=i.name;$('#iprice').value=i.price;$('#icat').value=i.category;$('#iunit').value=i.unit;$('#inotes').value=i.notes;pendingPhotos=[...i.photos];renderPending();scrollTo(0,0)});
 $('#itemList').querySelectorAll('[data-d]').forEach(b=>b.onclick=async()=>{if(confirm('Delete product?')){await delItem(b.dataset.d);refresh()}});
 $('#manualItem').innerHTML=items.map(i=>`<option value="${i.id}">${esc(i.name)} (${inr(i.price)})</option>`).join('');renderBill()}
// ---------- class-agnostic object finder ----------
function boxIoU(a,b){const x=Math.max(a[0],b[0]),y=Math.max(a[1],b[1]),r=Math.min(a[0]+a[2],b[0]+b[2]),t=Math.min(a[1]+a[3],b[1]+b[3]);const i=Math.max(0,r-x)*Math.max(0,t-y);return{iou:i/(a[2]*a[3]+b[2]*b[3]-i),cont:i/Math.min(a[2]*a[3],b[2]*b[3]),inA:i/(b[2]*b[3])}}
function union(a,b){const x=Math.min(a[0],b[0]),y=Math.min(a[1],b[1]);return[x,y,Math.max(a[0]+a[2],b[0]+b[2])-x,Math.max(a[1]+a[3],b[1]+b[3])-y]}
function contourBoxes(img){const W0=img.width||img.videoWidth,H0=img.height||img.videoHeight,sc=Math.min(1,256/Math.max(W0,H0)),W=Math.round(W0*sc),H=Math.round(H0*sc),N=W*H;
 const c=document.createElement('canvas');c.width=W;c.height=H;const g=c.getContext('2d');g.filter='blur(1px)';g.drawImage(img,0,0,W,H);const d=g.getImageData(0,0,W,H).data;
 // background = median of border pixels
 const bor=[];for(let x=0;x<W;x++)for(const y of[0,1,H-2,H-1])bor.push(y*W+x);for(let y=0;y<H;y++)for(const x of[0,1,W-2,W-1])bor.push(y*W+x);
 const med=ch=>{const v=bor.map(i=>d[i*4+ch]).sort((a,b)=>a-b);return v[v.length>>1]};const bg=[med(0),med(1),med(2)];
 const diff=new Float32Array(N),gray=new Float32Array(N);for(let i=0;i<N;i++){diff[i]=Math.max(Math.abs(d[i*4]-bg[0]),Math.abs(d[i*4+1]-bg[1]),Math.abs(d[i*4+2]-bg[2]));gray[i]=.3*d[i*4]+.59*d[i*4+1]+.11*d[i*4+2]}
 const bd=bor.map(i=>diff[i]).sort((a,b)=>a-b),T=Math.max(28,bd[Math.floor(bd.length*.95)]+18);
 let m=new Uint8Array(N);for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){const i=y*W+x,gx=gray[i+1]-gray[i-1],gy=gray[i+W]-gray[i-W];if(diff[i]>T||Math.hypot(gx,gy)>70)m[i]=1}
 const morph=(src,r,dil)=>{const o=new Uint8Array(N);for(let y=0;y<H;y++)for(let x=0;x<W;x++){let v=dil?0:1;for(let dy=-r;dy<=r&&(dil?!v:v);dy++)for(let dx=-r;dx<=r;dx++){const yy=y+dy,xx=x+dx;const p=(yy<0||xx<0||yy>=H||xx>=W)?0:src[yy*W+xx];if(dil&&p){v=1;break}if(!dil&&!p){v=0;break}}o[y*W+x]=v}return o};
 m=morph(morph(m,1,false),1,true);m=morph(morph(m,3,true),3,false); // open (denoise) then close (fill)
 const lab=new Int32Array(N),boxes=[],st=new Int32Array(N);let L=0;
 for(let s=0;s<N;s++){if(!m[s]||lab[s])continue;L++;let sp=0,cnt=0,x0=W,y0=H,x1=0,y1=0;st[sp++]=s;lab[s]=L;
  while(sp){const i=st[--sp],x=i%W,y=(i/W)|0;cnt++;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;
   for(const j of[i-1,i+1,i-W,i+W,i-W-1,i-W+1,i+W-1,i+W+1]){if(j<0||j>=N||!m[j]||lab[j])continue;if(Math.abs(j%W-x)>1)continue;lab[j]=L;st[sp++]=j}}
  const bw=x1-x0+1,bh=y1-y0+1;if(cnt>N*.004&&bw*bh<N*.9&&bw>4&&bh>4)boxes.push([x0/sc,y0/sc,bw/sc,bh/sc])}
 // merge fragments of the same object
 let merged=true;while(merged){merged=false;for(let i=0;i<boxes.length&&!merged;i++)for(let j=i+1;j<boxes.length;j++)if(boxIoU(boxes[i],boxes[j]).cont>.6){boxes[i]=union(boxes[i],boxes[j]);boxes.splice(j,1);merged=true;break}}
 return boxes}
async function findObjects(img){const W=img.width||img.videoWidth,H=img.height||img.videoHeight;let boxes=contourBoxes(img);
 if(coco){const dets=(await coco.detect(img,20,.2)).map(d=>d.bbox).filter(b=>b[2]*b[3]<W*H*.8&&b[2]*b[3]>W*H*.004);
  // split a contour blob that covers >=2 COCO objects (touching items)
  boxes=boxes.flatMap(b=>{const inside=dets.filter(c=>boxIoU(b,c).inA>.8);return inside.length>=2?inside:[b]});
  for(const c of dets)if(!boxes.some(b=>{const o=boxIoU(b,c);return o.iou>.3||o.cont>.6}))boxes.push(c)}
 // NMS-style dedupe
 boxes.sort((a,b)=>b[2]*b[3]-a[2]*a[3]);const out=[];for(const b of boxes)if(!out.some(o=>boxIoU(o,b).iou>.5))out.push(b);
 return out.sort((a,b)=>a[1]-b[1]||a[0]-b[0])}
function cropCanvas(img,[x,y,w,h],pad=.12){const W=img.width||img.videoWidth,H=img.height||img.videoHeight,s=Math.max(w,h)*(1+2*pad),cx=x+w/2,cy=y+h/2;
 const c=document.createElement('canvas');c.width=c.height=224;const g=c.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,224,224);
 const sx=Math.max(0,cx-s/2),sy=Math.max(0,cy-s/2),ex=Math.min(W,cx+s/2),ey=Math.min(H,cy+s/2),k=224/s;g.drawImage(img,sx,sy,ex-sx,ey-sy,(sx-(cx-s/2))*k,(sy-(cy-s/2))*k,(ex-sx)*k,(ey-sy)*k);return c}
// ---------- model: MobileNet embeddings + cosine kNN with Unknown ----------
let bank=null,bankLabels=[];const thr=()=>+$('#thr').value;
$('#thr').oninput=()=>{$('#thrVal').textContent=(+$('#thr').value).toFixed(2);localStorage.setItem('vb_thr',$('#thr').value)};
const embed=el=>tf.tidy(()=>{const e=net.infer(el,true).squeeze();return e.div(e.norm())});
function buildBank(saved){if(bank)bank.dispose();const ok=saved.filter(s=>items.find(i=>i.id==s.l));bankLabels=ok.map(s=>s.l);bank=ok.length?tf.tensor2d(ok.map(s=>s.e)):null}
function augs(src){const out=[];for(const [flip,zoom,rot] of [[0,1,0],[1,1,0],[0,.85,0],[0,1,.2]]){const c=document.createElement('canvas');c.width=c.height=224;const g=c.getContext('2d');g.fillStyle='#fff';g.fillRect(0,0,224,224);g.translate(112,112);g.rotate(rot);if(flip)g.scale(-1,1);const z=224/zoom;g.drawImage(src,-z/2,-z/2,z,z);out.push(c)}return out}
$('#trainBtn').onclick=async()=>{if(!net)return alert('Model still loading');const ok=items.filter(i=>i.photos.length);if(ok.length<1)return alert('Add products with photos first');
 $('#trainBtn').disabled=true;const saved=[];let n=0,tot=ok.reduce((s,i)=>s+i.photos.length,0);
 for(const it of ok)for(const p of it.photos){const im=await loadImg(p);const bx=contourBoxes(im).sort((a,b)=>b[2]*b[3]-a[2]*a[3])[0];
  const srcs=[...(bx?augs(cropCanvas(im,bx)):[]),cropCanvas(im,[0,0,im.width,im.height],0)];
  for(const c of srcs){const e=embed(c);saved.push({l:it.id,e:Array.from(await e.data())});e.dispose()}status(`Training ${++n}/${tot}`);await tf.nextFrame()}
 await kvSet('embeddings',saved);buildBank(saved);$('#trainBtn').disabled=false;status('Model ready ✔');$('#trainInfo').textContent=`Trained on ${tot} photos (auto-cropped to the object, ${saved.length} embeddings) for ${ok.length} products. Saved.`};
async function restoreModel(){const s=await kvGet('embeddings');if(!s)return;buildBank(s);if(bank)$('#trainInfo').textContent=`Restored trained model (${s.length} embeddings).`}
async function classify(c){if(!bank)return{id:'unknown',conf:0};const e=embed(c);const sims=await tf.tidy(()=>bank.matMul(e.expandDims(1)).squeeze()).array();e.dispose();
 const sims1=[].concat(sims);const best={};sims1.forEach((s,i)=>{const l=bankLabels[i];if(!(l in best)||s>best[l])best[l]=s});
 const [id,sim]=Object.entries(best).sort((a,b)=>b[1]-a[1])[0];return{id:sim>=thr()?id:'unknown',conf:sim,guess:id}}
// ---------- billing: scans ----------
let scans=[];const nameOf=id=>id=='unknown'?'Unknown':(items.find(i=>i.id==id)||{}).name||'?';
const COLORS=['#e11d48','#2563eb','#16a34a','#d97706','#7c3aed','#0891b2','#db2777','#65a30d'];
async function handleImages(imgs){if(!bank)return alert('Train the model first (Products tab).');status('Scanning…');
 for(const img of imgs){const boxes=$('#multiDetect').checked?await findObjects(img):[];if(!boxes.length)boxes.push([0,0,img.width||img.videoWidth,img.height||img.videoHeight]);
  const sc={img,dets:[],sel:-1};for(const b of boxes)sc.dets.push(await makeDet(img,b));scans.push(sc)}
 renderScans();status('Model ready ✔')}
async function makeDet(img,box){const crop=cropCanvas(img,box);const r=await classify(crop);return{box,crop,...r,qty:1}}
function renderScans(){const root=$('#scanResults');root.innerHTML='';
 scans.forEach((sc,si)=>{const wrap=document.createElement('div');wrap.className='scanimg';const W=sc.img.width,H=sc.img.height;
  const cv=document.createElement('canvas');cv.className='prev';cv.width=W;cv.height=H;const g=cv.getContext('2d');g.drawImage(sc.img,0,0);const lw=Math.max(2,W/250),fs=Math.max(14,W/35);
  sc.dets.forEach((d,i)=>{const col=d.id=='unknown'?'#dc2626':COLORS[i%COLORS.length];g.lineWidth=i==sc.sel?lw*2.5:lw;g.strokeStyle=col;g.strokeRect(...d.box);g.font=`bold ${fs}px sans-serif`;const t=`${i+1}. ${nameOf(d.id)}`;g.fillStyle=col;g.fillRect(d.box[0],d.box[1],g.measureText(t).width+10,fs*1.3);g.fillStyle='#fff';g.fillText(t,d.box[0]+5,d.box[1]+fs)});
  let start=null;const pt=e=>{const r=cv.getBoundingClientRect();return[(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height]};
  cv.onpointerdown=e=>{start=pt(e);cv.setPointerCapture(e.pointerId)};
  cv.onpointerup=async e=>{if(!start)return;const p=pt(e),b=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])];const s0=start;start=null;
   if(b[2]<W/40||b[3]<H/40){sc.sel=sc.dets.findIndex(d=>s0[0]>=d.box[0]&&s0[0]<=d.box[0]+d.box[2]&&s0[1]>=d.box[1]&&s0[1]<=d.box[1]+d.box[3]);return renderScans()}
   sc.dets.push(await makeDet(sc.img,b));renderScans()};
  wrap.append(cv);
  sc.dets.forEach((d,i)=>{const row=document.createElement('div');row.className='det'+(i==sc.sel?' sel':'');
   row.innerHTML=`<span class="n">${i+1}</span><span></span><select><option value="unknown">❓ Unknown</option>${items.map(it=>`<option value="${it.id}">${esc(it.name)}</option>`).join('')}</select><span class="conf ${d.id=='unknown'?'low':''}" title="cosine similarity to nearest training photo (best guess: ${esc(nameOf(d.guess))})">${Math.round(d.conf*100)}%</span><input type="number" min="1" value="${d.qty}" style="width:56px"><button title="delete box">🗑</button>`;
   row.children[1].replaceWith(d.crop);const sel=row.querySelector('select');sel.value=d.id;sel.onchange=()=>{d.id=sel.value;renderScans()};
   row.querySelector('input').onchange=e=>{d.qty=Math.max(1,+e.target.value||1);renderSummary()};row.querySelector('button').onclick=()=>{sc.dets.splice(i,1);sc.sel=-1;renderScans()};
   row.onclick=e=>{if(e.target.tagName=='DIV'||e.target.tagName=='SPAN'){sc.sel=i;renderScans()}};wrap.append(row)});
  const ctl=document.createElement('div');ctl.className='row';ctl.innerHTML=`<span class="muted">${sc.dets.length} object(s) found</span><button>Remove photo</button>`;ctl.querySelector('button').onclick=()=>{scans.splice(si,1);renderScans()};wrap.append(ctl);root.append(wrap)});
 renderSummary()}
function counts(){const c={};scans.forEach(s=>s.dets.forEach(d=>c[d.id]=(c[d.id]||0)+d.qty));return c}
function renderSummary(){const c=counts();$('#summaryBox').hidden=!scans.length;$('#summary').innerHTML=Object.entries(c).map(([id,n])=>`<span class="chip ${id=='unknown'?'unk':''}">${esc(nameOf(id))} × ${n}</span>`).join('')||'<span class="muted">Nothing detected</span>'}
$('#addAll').onclick=()=>{const c=counts();if(c.unknown&&!confirm(`${c.unknown} Unknown item(s) will be skipped. Fix labels first? Press OK to skip them.`))return;
 for(const [id,n] of Object.entries(c))if(id!='unknown')addToBill(id,n);scans=[];renderScans()};
$('#clearScans').onclick=()=>{scans=[];renderScans()};
function addToBill(id,q){const l=bill.find(b=>b.id==id);l?l.qty+=q:bill.push({id,qty:q});saveBill()}
function saveBill(){localStorage.setItem('vb_bill',JSON.stringify(bill));localStorage.setItem('vb_gst',$('#gst').value);renderBill()}
function totals(){const lines=bill.map(b=>{const i=items.find(x=>x.id==b.id);return i&&{...b,name:i.name,unit:i.unit,price:i.price,total:i.price*b.qty}}).filter(Boolean);const sub=lines.reduce((s,l)=>s+l.total,0),g=sub*(+$('#gst').value||0)/100;return{lines,sub,g,grand:sub+g}}
function renderBill(){const t=totals();$('#billDate').textContent=new Date().toLocaleString('en-IN');
 $('#billTable tbody').innerHTML=t.lines.map((l,i)=>`<tr><td>${esc(l.name)}<div class="muted">${esc(l.unit)}</div></td><td><input type="number" min="0" step="any" value="${l.qty}" data-q="${l.id}"></td><td>${inr(l.price)}</td><td>${inr(l.total)}</td><td class="np"><button data-r="${l.id}">✕</button></td></tr>`).join('')||'<tr><td colspan="5" class="muted">No items yet</td></tr>';
 document.querySelectorAll('[data-q]').forEach(e=>e.onchange=()=>{const b=bill.find(x=>x.id==e.dataset.q);b.qty=+e.value;if(b.qty<=0)bill=bill.filter(x=>x!==b);saveBill()});
 document.querySelectorAll('[data-r]').forEach(e=>e.onclick=()=>{bill=bill.filter(x=>x.id!=e.dataset.r);saveBill()});
 $('#subtotal').textContent=inr(t.sub);$('#gstAmt').textContent=inr(t.g);$('#grand').textContent=inr(t.grand)}
$('#gst').oninput=saveBill;$('#addManual').onclick=()=>$('#manualItem').value&&addToBill($('#manualItem').value,1);$('#clearBill').onclick=()=>{bill=[];saveBill()};
$('#pdfBtn').onclick=()=>{const {jsPDF}=window.jspdf,d=new jsPDF(),t=totals(),R=n=>'Rs. '+n.toFixed(2);
 d.setFontSize(16);d.text('VisionBill – Grocery Invoice',14,16);d.setFontSize(10);d.text(new Date().toLocaleString('en-IN'),14,23);
 d.autoTable({startY:28,head:[['#','Item','Qty','Unit price','Total']],body:t.lines.map((l,i)=>[i+1,l.name+(l.unit?` (${l.unit})`:''),l.qty,R(l.price),R(l.total)]),
  foot:[['','','','Subtotal',R(t.sub)],['','','',`GST ${$('#gst').value||0}%`,R(t.g)],['','','','Grand total',R(t.grand)]],theme:'grid',headStyles:{fillColor:[22,163,74]},footStyles:{fillColor:[240,240,240],textColor:20}});
 d.save(`bill-${Date.now()}.pdf`)};
// ---------- init ----------
(async()=>{await openDB();bill=JSON.parse(localStorage.getItem('vb_bill')||'[]');$('#gst').value=localStorage.getItem('vb_gst')||0;await refresh();
 try{await tf.ready();$('#thr').value=localStorage.getItem('vb_thr')||0.6;$('#thr').oninput();net=await mobilenet.load({version:2,alpha:1});await restoreModel();status(bank?'Model ready ✔':'Feature extractor ready – train model');
  cocoSsd.load({base:'mobilenet_v2'}).then(m=>coco=m).catch(e=>console.warn('COCO-SSD failed',e))}catch(e){status('Model load failed');console.error(e)}})();
// ---------- inputs ----------
$('#billPhotos').onchange=async e=>{const imgs=[];for(const f of e.target.files)imgs.push(await loadImg(URL.createObjectURL(f)));e.target.value='';handleImages(imgs)};
let stream;$('#camBtn').onclick=async()=>{const v=$('#cam');if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;v.hidden=$('#snapBtn').hidden=true;return}
 try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});v.srcObject=stream;await v.play();v.hidden=$('#snapBtn').hidden=false}catch(err){alert('Camera unavailable: '+err.message)}};
$('#snapBtn').onclick=()=>{const v=$('#cam'),c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);handleImages([c])};
