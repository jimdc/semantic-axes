/* Offline support. Strategy:
 *   - vectors.bin (large, immutable per build): cache-first (don't re-download the 50 MB each load)
 *   - everything else same-origin: network-first with cache fallback (fresh when online, works offline)
 * Bump CACHE to invalidate. */
const CACHE = "semantic-axes-v1";
const SHELL = [
  "./", "index.html", "manifest.json",
  "static/styles.css", "static/embed.js", "static/backend.js", "static/vis.js", "static/app.js",
  "data/vocab.json", "data/axes.json", "data/meta.json", "data/sae_features.json",
];

self.addEventListener("install", e => {
  self.skipWaiting();
  // allSettled so a not-yet-built file (e.g. sae_features.json) doesn't fail the whole install
  e.waitUntil(caches.open(CACHE).then(c => Promise.allSettled(SHELL.map(u => c.add(u)))));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin) return;

  const putCopy = res => { if (res.ok) { const c = res.clone(); caches.open(CACHE).then(x => x.put(req, c)); } return res; };

  if (url.pathname.endsWith(".bin")) {                       // cache-first for the big vector blob
    e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(putCopy)));
  } else {                                                   // network-first for code + JSON data
    e.respondWith(fetch(req).then(putCopy).catch(() =>
      caches.match(req).then(hit => hit || caches.match("index.html"))));
  }
});
