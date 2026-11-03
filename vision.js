// Shared vision core (browser + node tests). Works on transformers.js RawImage.
export const PROMPTS=['a product','an object','a package','a box','a tool','a bottle','a cable','a can','a packet','a jar'];
export function iou(a,b){const x=Math.max(a[0],b[0]),y=Math.max(a[1],b[1]),r=Math.min(a[0]+a[2],b[0]+b[2]),t=Math.min(a[1]+a[3],b[1]+b[3]);const i=Math.max(0,r-x)*Math.max(0,t-y);return{iou:i/(a[2]*a[3]+b[2]*b[3]-i),cont:i/Math.min(a[2]*a[3],b[2]*b[3]),inA:i/(b[2]*b[3])}}
// class-agnostic open-vocabulary detection with NMS + size filtering
export async function detect(det,img,{minScore=0.08,minArea=0.002,maxArea=0.6,max=30,extra=[]}={}){
 const W=img.width,H=img.height,A=W*H;
 const raw=await det(img,[...PROMPTS,...extra],{threshold:0.04,percentage:false});
 let c=raw.map(o=>({box:[o.box.xmin,o.box.ymin,o.box.xmax-o.box.xmin,o.box.ymax-o.box.ymin],score:o.score,label:o.label}));
 c=c.filter(o=>o.box[2]*o.box[3]>=A*minArea&&o.score>=minScore).sort((a,b)=>b.score-a.score);
 const keep=[];for(const o of c){if(keep.some(k=>{const v=iou(k.box,o.box);return v.iou>0.45||(v.cont>0.85&&Math.abs(Math.log((k.box[2]*k.box[3])/(o.box[2]*o.box[3])))<0.7)}))continue;keep.push(o)}
 // mark huge boxes and "container" boxes (hold >=2 other detections) — e.g. the laptop/table
 for(const o of keep)o.big=o.box[2]*o.box[3]>A*maxArea
 return keep.slice(0,max)}
export async function squareCrop(img,[x,y,w,h],pad=0.1){const s=Math.max(w,h)*(1+2*pad),cx=x+w/2,cy=y+h/2;
 const x0=Math.max(0,Math.round(cx-s/2)),y0=Math.max(0,Math.round(cy-s/2)),x1=Math.min(img.width-1,Math.round(cx+s/2)),y1=Math.min(img.height-1,Math.round(cy+s/2));return await img.crop([x0,y0,x1,y1])}
export async function embed(fe,img){const o=await fe(img);const d=Array.from(o.data.slice(0,o.dims[2]));const n=Math.hypot(...d)||1;return d.map(v=>v/n)}
// bank: [{l,e}] individual views (never averaged). k-NN vote + top-1 score + top1-vs-top2 margin rejection.
export function classify(bank,e,thr,{k=5,minMargin=0.03}={}){const sims=bank.map(({l,e:b})=>{let s=0;for(let i=0;i<b.length;i++)s+=b[i]*e[i];return{l,s}}).sort((a,b)=>b.s-a.s);
 if(!sims.length)return{id:'unknown',conf:0,guess:null,margin:0};
 const best={};for(const {l,s} of sims)if(!(l in best))best[l]=s;const bg=best[BG]??-1;delete best[BG];
 const vote={};for(const {l,s} of sims.slice(0,k))vote[l]=(vote[l]||0)+s;
 const ranked=Object.keys(best).sort((a,b)=>((vote[b]||0)-(vote[a]||0))||(best[b]-best[a]));if(!ranked.length)return{id:'unknown',conf:0,guess:null,margin:0};
 const id=ranked[0],sim=best[id],second=Math.max(-1,...Object.entries(best).filter(([l])=>l!==id).map(([,v])=>v)),margin=sim-(second>-1?second:0);
 const ok=sim>=thr&&sim>bg+0.03&&(margin>=minMargin||second<0)&&(vote[id]||0)>=(vote[BG]||0);
 return{id:ok?id:'unknown',conf:sim,guess:id,margin,bg}}

export async function embedCrop(fe,img,box,aug=false){const c=await squareCrop(img,box);const out=[await embed(fe,c)];
 if(aug){const s=c.width;out.push(await embed(fe,await squareCrop(img,box,0.0)));out.push(await embed(fe,await squareCrop(img,box,0.25)))}return out}
// full pipeline: detect -> crop -> classify -> merge same-label fragments -> drop parts/huge unknowns
export async function analyze(det,fe,img,bank,{thr=0.5,extra=[],debug=null}={}){
 const boxes=await detect(det,img,{extra});const res=[];
 for(const b of boxes){const [e]=await embedCrop(fe,img,b.box);const c=classify(bank,e,thr);res.push({...b,...c})}
 if(debug)debug.push(...res.map(o=>({...o})));
 const area=o=>o.box[2]*o.box[3];
 // a box that contains >=2 boxes with different known labels is a container (laptop/table/tray): drop it
 let r=res.filter(o=>{if(o.big&&o.id==='unknown')return false;const labs=new Set(res.filter(k=>k!==o&&k.id!=='unknown'&&k.id!==o.id&&iou(o.box,k.box).inA>0.7).map(k=>k.id));return labs.size<1||(labs.size<2&&!o.big)});
 // merge overlapping fragments with the same known label, if the union stays compact
 r.sort((a,b)=>area(b)-area(a));
 let changed=true;while(changed){changed=false;for(let i=0;i<r.length&&!changed;i++)for(let j=i+1;j<r.length;j++){const a=r[i],b=r[j];if(a.id==='unknown'||a.id!==b.id||iou(a.box,b.box).cont<=0.25)continue;
  const x=Math.min(a.box[0],b.box[0]),y=Math.min(a.box[1],b.box[1]),u=[x,y,Math.max(a.box[0]+a.box[2],b.box[0]+b.box[2])-x,Math.max(a.box[1]+a.box[3],b.box[1]+b.box[3])-y];
  if(u[2]*u[3]>1.8*area(a))continue;a.box=u;a.conf=Math.max(a.conf,b.conf);a.score=Math.max(a.score,b.score);r.splice(j,1);changed=true;break}}
 // drop parts: boxes mostly inside a bigger box that is known
 r=r.filter(o=>!r.some(k=>k!==o&&k.id!=='unknown'&&area(k)>area(o)&&iou(k.box,o.box).inA>0.7&&(o.id==='unknown'||o.id===k.id)));
 // drop unknowns that contain a known item
 r=r.filter(o=>!(o.id==='unknown'&&r.some(k=>k.id!=='unknown'&&iou(o.box,k.box).inA>0.7)));
 return r.sort((a,b)=>a.box[1]-b.box[1]||a.box[0]-b.box[0])}
export const BG='__bg';
// one training photo -> positive embeddings for the product crop + background negatives from the rest of the scene
export async function trainPhoto(det,fe,img,{box=null,name='',exclude=[]}={}){
 const dets=await detect(det,img,{extra:name?[name]:[]});
 if(!box){const c=dets.filter(d=>!d.big);const byName=c.filter(d=>d.label===name).sort((a,b)=>b.score-a.score)[0];const best=byName||c[0];box=best?best.box:[0,0,img.width,img.height]}
 const pos=await embedCrop(fe,img,box,true);const neg=[];
 for(const d of dets){const o=iou(d.box,box);if(o.cont>0.3||[...exclude].some(x=>iou(d.box,x).cont>0.3))continue;neg.push((await embedCrop(fe,img,d.box))[0])}
 return{box,pos,neg}}
// STEP 1: class-agnostic segmentation into item boxes (no training needed)
export async function segment(det,img,{extra=[],minScore=0.1,maxArea=0.6}={}){
 const A=img.width*img.height,area=o=>o.box[2]*o.box[3];
 let b=await detect(det,img,{extra,minScore,maxArea});
 // containers: a box holding >=3 other detections (laptop, tray, table)
 b=b.filter(o=>b.filter(k=>k!==o&&iou(o.box,k.box).inA>0.75).length<3);
 // parts: a box mostly inside a not-much-larger box (sub-part of the same item)
 b=b.filter(o=>!b.some(k=>k!==o&&area(k)>area(o)&&iou(k.box,o.box).inA>0.75&&area(k)<4*area(o)));
 return b.sort((x,y)=>x.box[1]-y.box[1]||x.box[0]-y.box[0])}
// post-classification cleanup: merge fragments/parts of one item, keep separate copies of the same product
export function dedupe(ds){const area=d=>d.box[2]*d.box[3];let r=[...ds].sort((a,b)=>area(b)-area(a)),changed=true;
 while(changed){changed=false;outer:for(let i=0;i<r.length;i++)for(let j=i+1;j<r.length;j++){const a=r[i],b=r[j],o=iou(a.box,b.box);
  const nested=o.inA>0.75,sameish=a.id===b.id||a.id==='unknown'||b.id==='unknown'||o.iou>0.5||(nested&&o.inA>0.85);
  if((nested&&sameish)||(a.id===b.id&&o.cont>0.4)||(sameish&&o.iou>0.5)){
   if(a.id==='unknown'&&b.id!=='unknown'){a.id=b.id;a.conf=b.conf;a.guess=b.guess}
   if(a.id===b.id&&!nested){const x=Math.min(a.box[0],b.box[0]),y=Math.min(a.box[1],b.box[1]);a.box=[x,y,Math.max(a.box[0]+a.box[2],b.box[0]+b.box[2])-x,Math.max(a.box[1]+a.box[3],b.box[1]+b.box[3])-y]}
   r.splice(j,1);changed=true;break outer}}}
 return r.sort((a,b)=>a.box[1]-b.box[1]||a.box[0]-b.box[0])}
// v7 "closer look": average per-class best similarity over many views of one crop
export function scoreViews(bank,es){const acc={};for(const e of es){const best={};for(const {l,e:b} of bank){let s=0;for(let i=0;i<b.length;i++)s+=b[i]*e[i];if(!(l in best)||s>best[l])best[l]=s}for(const [l,s] of Object.entries(best))acc[l]=(acc[l]||0)+s/es.length}
 const bg=acc[BG]??-1;delete acc[BG];return{ranked:Object.entries(acc).sort((a,b)=>b[1]-a[1]).map(([id,s])=>({id,s})),bg}}
export function decide({ranked,bg},thr=0.55,minMargin=0.02){if(!ranked.length)return{id:'unknown',conf:0,guess:null,margin:0,top3:[]};const [a,b]=ranked,margin=a.s-(b?b.s:0);
 const ok=a.s>=thr&&margin>=minMargin&&a.s>bg+0.02;return{id:ok?a.id:'unknown',conf:a.s,guess:a.id,margin,top3:ranked.slice(0,3)}}
// batched embeddings: one model call for many crops
export async function embedBatch(fe,imgs,bs=8){const out=[];for(let i=0;i<imgs.length;i+=bs){const o=await fe(imgs.slice(i,i+bs));const [n,t,D]=o.dims;for(let k=0;k<n;k++){const d=Array.from(o.data.slice(k*t*D,k*t*D+D));const m=Math.hypot(...d)||1;out.push(d.map(v=>v/m))}}return out}
// ---------- v10: multi-cue fusion (DINOv2 shape/texture + colour histogram + optional OCR text) ----------
// grey-world normalised hue/sat histogram (12x3) + 4 achromatic brightness bins, over the inner region
export function colourHist(data,W,H,ch,inner=[0,0,W,H]){const HB=12,SB=3,VB=4,h=new Float32Array(HB*SB+VB);let n=0;const [x0,y0,w,hh]=inner.map(Math.round);
 let mr=0,mg=0,mb=0,m=0;for(let y=y0;y<y0+hh;y+=2)for(let x=x0;x<x0+w;x+=2){const i=(y*W+x)*ch;mr+=data[i];mg+=data[i+1];mb+=data[i+2];m++}const g=(mr+mg+mb)/3/m,kr=g/(mr/m||1),kg=g/(mg/m||1),kb=g/(mb/m||1);
 for(let y=y0;y<y0+hh;y+=2)for(let x=x0;x<x0+w;x+=2){const i=(y*W+x)*ch;const r=Math.min(255,data[i]*kr)/255,gg=Math.min(255,data[i+1]*kg)/255,b=Math.min(255,data[i+2]*kb)/255;
  const mx=Math.max(r,gg,b),mn=Math.min(r,gg,b),d=mx-mn,s=mx?d/mx:0;let bin;
  if(s<0.25||mx<0.15)bin=HB*SB+Math.min(VB-1,Math.floor(mx*VB));else{let hue=mx===r?((gg-b)/d+6)%6:mx===gg?(b-r)/d+2:(r-gg)/d+4;bin=Math.min(HB-1,Math.floor(hue/6*HB))*SB+Math.min(SB-1,Math.floor((s-0.25)/0.75*SB))}h[bin]++;n++}
 for(let i=0;i<h.length;i++)h[i]/=n;return Array.from(h,v=>Math.round(v*1e4)/1e4)}
export const histSim=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=Math.min(a[i],b[i]);return s};
export const INNER224=[224*0.15,224*0.15,224*0.7,224*0.7];
const ed=(a,b)=>{const d=Array.from({length:a.length+1},(_,i)=>[i]);for(let j=1;j<=b.length;j++)d[0][j]=j;for(let i=1;i<=a.length;i++)for(let j=1;j<=b.length;j++)d[i][j]=Math.min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!==b[j-1]));return d[a.length][b.length]};
// vocab: {productId:[WORDS]} ; returns {id:1} only when the read text points to exactly one product
export function ocrHits(text,vocab){const ws=(String(text).toUpperCase().match(/[A-Z]{4,}/g)||[]);const hit={};for(const [id,v] of Object.entries(vocab))for(const w of ws)for(const t of v)if(w===t||(t.length>=5&&ed(w,t)<=1))hit[id]=1;return Object.keys(hit).length===1?hit:{}}
// score-level fusion: rank by embedding + wc*(colour-0.5) + wo*ocr; accept only if the embedding itself is confident
export const WC=0.1,WO=0.05;
export function decideFused({ranked,bg},cols,col,ocr={},thr=0.5,minMargin=0.02){if(!ranked.length)return{id:'unknown',conf:0,guess:null,margin:0,top3:[]};
 const rk=ranked.map(({id,s})=>{const hs=col&&cols[id]&&cols[id].length?Math.max(...cols[id].map(h=>histSim(h,col))):0.5;return{id,s,c:hs,f:s+WC*(hs-0.5)+WO*(ocr[id]||0)}}).sort((a,b)=>b.f-a.f);
 const [a,b]=rk,margin=a.f-(b?b.f:0),ok=a.s>=thr&&margin>=minMargin&&a.s>bg+0.02;return{id:ok?a.id:'unknown',conf:a.s,guess:a.id,margin,top3:rk.slice(0,3).map(t=>({id:t.id,s:t.f}))}}
// merge two boxes with the same label that overlap >=25% (a tilted pack split in two), unless the union swallows another item
export function mergeSame(r){const A=b=>b[2]*b[3];let ch=true;while(ch){ch=false;for(let i=0;i<r.length&&!ch;i++)for(let j=i+1;j<r.length;j++){const a=r[i],b=r[j];if(a.id==='unknown'||a.id!==b.id)continue;if(iou(a.box,b.box).cont<0.25)continue;
 const x=Math.min(a.box[0],b.box[0]),y=Math.min(a.box[1],b.box[1]),u=[x,y,Math.max(a.box[0]+a.box[2],b.box[0]+b.box[2])-x,Math.max(a.box[1]+a.box[3],b.box[1]+b.box[3])-y];
 if(A(u)>2.3*Math.max(A(a.box),A(b.box))||r.some(k=>k!==a&&k!==b&&iou(u,k.box).inA>0.5))continue;a.box=u;a.conf=Math.max(a.conf,b.conf);r.splice(j,1);ch=true;break}}return r}
