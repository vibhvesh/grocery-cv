// v12 Tray mode: find a printed tray (4 ArUco corner markers, ids 0-3 = TL,TR,BR,BL) or use tapped corners,
// warp it to a top-down view and split it into rows x cols slot crops.
// Marker convention: each marker sits OUTSIDE one corner of the slot grid; its corner nearest the tray centre touches the grid corner.
export const MARKER_IDS=[0,1,2,3];
export function markerBits(AR,id){const d=new AR.Dictionary('ARUCO_MIP_36h12');const c=d.codeList[id];const n=d.markSize-2;const g=[];for(let y=0;y<n;y++){g.push([]);for(let x=0;x<n;x++)g[y].push(c[y*n+x]==='1')}return g}
// solve homography mapping src[i] -> dst[i] (4 points)
export function homography(src,dst){const A=[],b=[];for(let i=0;i<4;i++){const [x,y]=src[i],[u,v]=dst[i];A.push([x,y,1,0,0,0,-u*x,-u*y]);b.push(u);A.push([0,0,0,x,y,1,-v*x,-v*y]);b.push(v)}
 const n=8;for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;[A[c],A[p]]=[A[p],A[c]];[b[c],b[p]]=[b[p],b[c]];for(let r=0;r<n;r++){if(r===c)continue;const f=A[r][c]/A[c][c];for(let k=c;k<n;k++)A[r][k]-=f*A[c][k];b[r]-=f*b[c]}}
 const h=b.map((v,i)=>v/A[i][i]);return[...h,1]}
// warp quad (TL,TR,BR,BL in source pixels) to a W x H top-down image (bilinear)
export function warpQuad(img,quad,W,H){const Hm=homography([[0,0],[W,0],[W,H],[0,H]],quad);const out=new Uint8ClampedArray(W*H*4),{data:s,width:sw,height:sh}=img;
 for(let y=0;y<H;y++)for(let x=0;x<W;x++){const X=x+0.5,Y=y+0.5,w=Hm[6]*X+Hm[7]*Y+1,u=(Hm[0]*X+Hm[1]*Y+Hm[2])/w-0.5,v=(Hm[3]*X+Hm[4]*Y+Hm[5])/w-0.5;const o=(y*W+x)*4;
  if(u<0||v<0||u>=sw-1||v>=sh-1){out[o]=out[o+1]=out[o+2]=200;out[o+3]=255;continue}const x0=u|0,y0=v|0,fx=u-x0,fy=v-y0,i=(y0*sw+x0)*4;
  for(let k=0;k<3;k++)out[o+k]=(s[i+k]*(1-fx)+s[i+4+k]*fx)*(1-fy)+(s[i+sw*4+k]*(1-fx)+s[i+sw*4+4+k]*fx)*fy;out[o+3]=255}
 return{data:out,width:W,height:H}}
const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
// returns {quad:[TL,TR,BR,BL], found:n} in img pixels, or null. img: {data(RGBA),width,height}
export function findTray(AR,img,maxSide=1200){const k=Math.min(1,maxSide/Math.max(img.width,img.height));let im=img;
 if(k<1){const W=Math.round(img.width*k),H=Math.round(img.height*k),d=new Uint8ClampedArray(W*H*4);for(let y=0;y<H;y++)for(let x=0;x<W;x++){const s=((Math.min(img.height-1,(y/k)|0))*img.width+Math.min(img.width-1,(x/k)|0))*4,o=(y*W+x)*4;d[o]=img.data[s];d[o+1]=img.data[s+1];d[o+2]=img.data[s+2];d[o+3]=255}im={data:d,width:W,height:H}}
 const det=new AR.Detector({dictionaryName:'ARUCO_MIP_36h12'});const ms=det.detect(im).filter(m=>MARKER_IDS.includes(m.id));const byId={};for(const m of ms)if(!byId[m.id])byId[m.id]=m;
 const ids=Object.keys(byId).map(Number);if(ids.length<3)return null;
 const centre=m=>[m.corners.reduce((a,c)=>a+c.x,0)/4,m.corners.reduce((a,c)=>a+c.y,0)/4];const cs={};for(const id of ids)cs[id]=centre(byId[id]);
 let C=[0,0];for(const id of ids){C[0]+=cs[id][0]/ids.length;C[1]+=cs[id][1]/ids.length}
 const q={};for(const id of ids){let best=null;for(const c of byId[id].corners){const p=[c.x,c.y];if(!best||dist(p,C)<dist(best,C))best=p}q[id]=best}
 // 3 markers: complete the parallelogram
 if(ids.length===3){const miss=[0,1,2,3].find(i=>!(i in q)),opp=(miss+2)%4,a=q[(miss+1)%4],b=q[(miss+3)%4],o=q[opp];q[miss]=[a[0]+b[0]-o[0],a[1]+b[1]-o[1]]}
 return{quad:[0,1,2,3].map(i=>[q[i][0]/k,q[i][1]/k]),found:ids.length}}
// output size of the warped tray: keep the physical aspect from the quad edge lengths, ~S px per slot
export function trayDims(quad,rows,cols,S=360){const w=(dist(quad[0],quad[1])+dist(quad[3],quad[2]))/2,h=(dist(quad[0],quad[3])+dist(quad[1],quad[2]))/2;const slotAsp=(w/cols)/(h/rows);return{W:Math.round(cols*S*Math.sqrt(slotAsp)),H:Math.round(rows*S/Math.sqrt(slotAsp))}}
export function slotRects(W,H,rows,cols,inset=0.045){const r=[];const sw=W/cols,sh=H/rows;for(let i=0;i<rows;i++)for(let j=0;j<cols;j++)r.push([j*sw+sw*inset,i*sh+sh*inset,sw*(1-2*inset),sh*(1-2*inset)]);return r}
// is there a product in this slot crop, and where? background colour = median of the slot's outer ring
export function slotForeground(img,{inkThr=48,minFrac=0.035}={}){const {data:d,width:W,height:H}=img;const ring=[];const st=Math.max(1,(Math.min(W,H)/60)|0);
 for(let x=0;x<W;x+=st)for(const y of [1,2,H-3,H-2])ring.push(y*W+x);for(let y=0;y<H;y+=st)for(const x of [1,2,W-3,W-2])ring.push(y*W+x);
 const med=k=>{const a=ring.map(i=>d[i*4+k]).sort((p,q)=>p-q);return a[a.length>>1]};const bg=[med(0),med(1),med(2)];
 const colCnt=new Uint32Array(W),rowCnt=new Uint32Array(H);let ink=0,n=0;
 for(let y=0;y<H;y+=2)for(let x=0;x<W;x+=2){const i=(y*W+x)*4;const dd=Math.abs(d[i]-bg[0])+Math.abs(d[i+1]-bg[1])+Math.abs(d[i+2]-bg[2]);n++;if(dd>inkThr){ink++;colCnt[x]++;rowCnt[y]++}}
 const frac=ink/n;if(frac<minFrac)return{empty:true,frac,bg};
 const lim=(cnt,len,other)=>{const t=Math.max(2,other/2*0.04);let a=0,b=len-1;while(a<len&&cnt[a]<t)a++;while(b>a&&cnt[b]<t)b--;return[a,b]};
 const [x0,x1]=lim(colCnt,W,H),[y0,y1]=lim(rowCnt,H,W);if(x1-x0<W*0.12||y1-y0<H*0.12)return{empty:true,frac,bg};
 return{empty:false,frac,bg,box:[x0,y0,x1-x0+1,y1-y0+1]}}
export function slotSig(img){const {data:d,width:W,height:H}=img,o=new Float32Array(256);let m=0;for(let y=0;y<16;y++)for(let x=0;x<16;x++){const i=(((y+0.5)*H/16|0)*W+((x+0.5)*W/16|0))*4;o[y*16+x]=d[i]*0.3+d[i+1]*0.59+d[i+2]*0.11;m+=o[y*16+x]}m/=256;for(let i=0;i<256;i++)o[i]-=m;return o}
// printable template geometry (landscape A-series page, mm). Same proportions for A4 and A3.
export function templateLayout(paper='A4'){const s=paper==='A3'?Math.SQRT2:1,W=297*s,H=210*s,m=6*s,M=20*s;const g=m+M;return{W,H,M,m,grid:[g,g,W-2*g,H-2*g]}}
