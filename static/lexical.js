/* lexical.js — the "keyword search" baseline: character-trigram Jaccard similarity.
 * This is exactly what fuzzy text search does under the hood (e.g. PostgreSQL pg_trgm):
 * it matches SPELLING, not meaning. Pairing it against vector (cosine) search is the whole
 * point of the demo — "king" and "viking" share trigrams but nothing else.
 *
 * Pure + UMD-exported so the Node unit tests can import the same code the browser runs. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Lexical = factory();
})(typeof self !== "undefined" ? self : this, function () {

  // pg_trgm-style: pad with two leading + one trailing space, then all 3-char windows.
  // "king" -> {"  k", " ki", "kin", "ing", "ng "}
  function trigrams(w) {
    const s = "  " + String(w).toLowerCase() + " ";
    const set = new Set();
    for (let i = 0; i + 3 <= s.length; i++) set.add(s.slice(i, i + 3));
    return set;
  }

  function jaccard(A, B) {
    let inter = 0;
    const [small, big] = A.size < B.size ? [A, B] : [B, A];
    for (const t of small) if (big.has(t)) inter++;
    const union = A.size + B.size - inter;
    return union ? inter / union : 0;
  }

  // Build a one-time inverted index over a corpus (array of words) so a query only scores
  // candidates that share at least one trigram — no full scan per query.
  function buildIndex(corpus) {
    const tg = new Map();        // word -> trigram Set
    const inv = new Map();       // trigram -> [corpus indices]
    corpus.forEach((w, i) => {
      const t = trigrams(w);
      tg.set(w, t);
      for (const g of t) { let a = inv.get(g); if (!a) { a = []; inv.set(g, a); } a.push(i); }
    });
    return { corpus, tg, inv };
  }

  // top-k keyword matches for a query word, ranked by trigram Jaccard
  function lexicalNearest(query, idx, k = 12) {
    const q = trigrams(query), cand = new Set();
    for (const g of q) { const a = idx.inv.get(g); if (a) for (const i of a) cand.add(i); }
    const out = [];
    for (const i of cand) {
      const w = idx.corpus[i];
      if (w === query) continue;
      out.push({ word: w, score: jaccard(q, idx.tg.get(w)) });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, k);
  }

  return { trigrams, jaccard, buildIndex, lexicalNearest };
});
