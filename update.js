(function(){
var bar=document.createElement('div');
bar.style.cssText='position:fixed;left:50%;top:12px;transform:translateX(-50%);z-index:99999;display:none;align-items:center;gap:10px;padding:10px 14px;border-radius:18px;background:rgba(40,10,80,.55);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid rgba(190,120,255,.5);box-shadow:0 0 18px rgba(160,80,255,.5);color:#fff;font:600 14px sans-serif;max-width:92vw';
var t=document.createElement('span');t.textContent='New update available';
var b=document.createElement('button');b.textContent='Update';b.style.cssText='border:0;border-radius:12px;padding:8px 14px;background:linear-gradient(135deg,#b44cff,#6a2cff);color:#fff;font:700 14px sans-serif';
var x=document.createElement('span');x.textContent='\u2715';x.style.cssText='opacity:.7;padding:4px';
bar.append(t,b,x);document.body.appendChild(bar);
var url=null;
x.onclick=function(){bar.style.display='none';};
b.onclick=function(){ if(window.NativeUpdate){t.textContent='Downloading...';NativeUpdate.install(url);} else {window.open(url,'_blank');t.textContent='Download started. Unzip and replace the old folder';b.style.display='none';} };
window.__upd=function(i){url=i.url;bar.style.display='flex';};
window.__updProg=function(p){ if(p>100){t.textContent='Opening installer...';} else t.textContent='Downloading '+p+'%'; };
window.__updMsg=function(m){t.textContent=m;};
if(window.NativeUpdate){setTimeout(function(){try{NativeUpdate.check();}catch(e){}},4000);}
})();
