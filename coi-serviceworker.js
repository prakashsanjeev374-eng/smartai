/* enables multi-threading on static hosts (adds COOP/COEP headers through a service worker) */
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', e => {
    if (e.request.cache === 'only-if-cached' && e.request.mode !== 'same-origin') return;
    e.respondWith(fetch(e.request).then(r => {
      if (r.status === 0) return r;
      const h = new Headers(r.headers);
      h.set('Cross-Origin-Embedder-Policy', 'require-corp');
      h.set('Cross-Origin-Opener-Policy', 'same-origin');
      return new Response(r.body, {status: r.status, statusText: r.statusText, headers: h});
    }).catch(err => { console.error(err); return Response.error(); }));
  });
} else {
  (async () => {
    if (window.crossOriginIsolated !== false || !window.isSecureContext || !navigator.serviceWorker) return;
    if (sessionStorage.getItem('coi_tried')) return;
    try {
      const reg = await navigator.serviceWorker.register(window.document.currentScript.src);
      sessionStorage.setItem('coi_tried', '1');
      if (reg.active && !navigator.serviceWorker.controller) { location.reload(); return; }
      reg.addEventListener('updatefound', () => { const w = reg.installing; w && w.addEventListener('statechange', () => { if (w.state === 'activated') location.reload(); }); });
      if (!reg.active) { const w = reg.installing || reg.waiting; w && w.addEventListener('statechange', () => { if (w.state === 'activated') location.reload(); }); }
    } catch (e) { console.warn('coi sw failed', e); }
  })();
}
