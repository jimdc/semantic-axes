/* embed.js — load the precomputed embedding + axis bank, and do the vector math.
 *
 * Data contract (emitted by tools/build_vectors.py):
 *   data/meta.json    {model, dim, count, scale, ...}
 *   data/vocab.json   ["the","king",...]            words in row order of vectors.bin
 *   data/vectors.bin  int8 (count x dim) row-major  centered+L2-normalized, quantized
 *   data/axes.json    {axes:[{id,label,pos,neg,poles,vector:[dim floats]}]}
 *
 * We keep the vectors as int8 and compute on them DIRECTLY (never materialize a float copy),
 * so memory == download size and the vocabulary can be large (~170k words). Real value of a
 * stored byte q is q*scale; vectors are unit (centered+L2-normalized), so dot products are
 * cosines in [-1,1]: word-word = similarity, word-axis = position on the axis.
 */
const Embed = (() => {
  let dim = 0, count = 0, scale = 1;
  let words = [];                 // row index -> word
  let index = new Map();          // word -> row index
  let q = null;                   // Int8Array(count*dim) — the raw quantized vectors
  let axes = [];                  // [{id,label,pos,neg,poles, vec:Float32Array(dim)}]  (unit float)
  let meta = {};

  async function load(base = "") {
    const [m, vocab, axj, binBuf] = await Promise.all([
      fetch(base + "data/meta.json").then(r => r.json()),
      fetch(base + "data/vocab.json").then(r => r.json()),
      fetch(base + "data/axes.json").then(r => r.json()),
      fetch(base + "data/vectors.bin").then(r => r.arrayBuffer()),
    ]);
    meta = m; dim = m.dim; count = m.count; scale = m.scale; words = vocab;
    index = new Map(words.map((w, i) => [w, i]));
    q = new Int8Array(binBuf);
    axes = axj.axes.map(a => ({ ...a, vec: Float32Array.from(a.vector) }));
    return { dim, count, nAxes: axes.length, model: m.model };
  }

  const has = w => index.has(w);
  const row = w => index.get(w);
  const wordAt = i => words[i];
  const allAxes = () => axes;
  const axisById = id => axes.find(a => a.id === id);
  const vocabList = () => words;
  const metaInfo = () => meta;

  // dot of word-row i with an arbitrary unit float vector -> cosine in [-1,1]
  function dotRowFloat(i, v) {
    let s = 0, off = i * dim;
    for (let d = 0; d < dim; d++) s += q[off + d] * v[d];
    return s * scale;
  }

  // word's position on every axis -> [{id,label,pos,neg,poles,score}]
  function axisScores(w) {
    const i = row(w);
    if (i === undefined) return [];
    return axes.map(a => ({ id: a.id, label: a.label, pos: a.pos, neg: a.neg,
                            poles: a.poles, score: dotRowFloat(i, a.vec) }));
  }

  // top-k nearest words by cosine. One full scan per word is cached (size-1 LRU) so the
  // several views that all want a neighborhood (chips, scatter, discovery) share it.
  const NBR_CACHE_K = 64;
  let _nc = { word: null, list: null };
  function _scanNearest(i, K) {
    const off = i * dim, sc2 = scale * scale, out = [];
    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      let s = 0, o2 = j * dim;
      for (let d = 0; d < dim; d++) s += q[off + d] * q[o2 + d];
      out.push([j, s]);
    }
    out.sort((a, b) => b[1] - a[1]);
    return out.slice(0, K).map(([j, s]) => ({ word: words[j], score: s * sc2, idx: j }));
  }
  function nearest(w, k = 12) {
    const i = row(w);
    if (i === undefined) return [];
    if (_nc.word !== w) _nc = { word: w, list: _scanNearest(i, NBR_CACHE_K) };
    return _nc.list.slice(0, Math.min(k, NBR_CACHE_K));
  }

  const dotVec = (a, b) => { let s = 0; for (let d = 0; d < dim; d++) s += a[d] * b[d]; return s; };
  // raw (unscaled) projection of word-row idx onto an arbitrary vector — for ranking only
  function projectRowVec(idx, v) { let s = 0, off = idx * dim; for (let d = 0; d < dim; d++) s += q[off + d] * v[d]; return s; }

  // local PCA: top-k variance directions of a word's neighborhood (power iteration + deflation).
  // Operates on the int8 rows directly (scale is a uniform factor, so PCA directions are unchanged).
  // Returns { idxs:[query, ...neighbors], comps:[Float64Array(dim) unit] }.
  function localPCA(word, nbrN = 60, k = 3) {
    const i = row(word);
    if (i === undefined) return null;
    const idxs = [i, ...nearest(word, nbrN).map(n => n.idx)];
    const n = idxs.length, mean = new Float64Array(dim);
    for (const ix of idxs) { const off = ix * dim; for (let d = 0; d < dim; d++) mean[d] += q[off + d]; }
    for (let d = 0; d < dim; d++) mean[d] /= n;
    const X = idxs.map(ix => { const off = ix * dim, r = new Float64Array(dim); for (let d = 0; d < dim; d++) r[d] = q[off + d] - mean[d]; return r; });
    const comps = [];
    for (let c = 0; c < k; c++) {
      let u = new Float64Array(dim);
      for (let d = 0; d < dim; d++) u[d] = Math.sin((d + 1) * (c + 1) * 0.7);   // deterministic seed
      let un = Math.hypot(...u) || 1; for (let d = 0; d < dim; d++) u[d] /= un;
      for (let it = 0; it < 40; it++) {
        const y = new Float64Array(dim);
        for (const r of X) { let dot = 0; for (let d = 0; d < dim; d++) dot += r[d] * u[d]; for (let d = 0; d < dim; d++) y[d] += dot * r[d]; }
        for (const p of comps) { let dot = 0; for (let d = 0; d < dim; d++) dot += y[d] * p[d]; for (let d = 0; d < dim; d++) y[d] -= dot * p[d]; }
        let ny = 0; for (let d = 0; d < dim; d++) ny += y[d] * y[d]; ny = Math.sqrt(ny) || 1;
        for (let d = 0; d < dim; d++) y[d] /= ny;
        u = y;
      }
      comps.push(u);
    }
    return { idxs, comps };
  }

  function projectWord(w, axisVec) {
    const i = row(w);
    return i === undefined ? null : dotRowFloat(i, axisVec);
  }

  // build a unit axis vector live from two anchor words (custom-axis feature).
  // scale is a common positive factor so it cancels under normalization.
  function customAxis(posWord, negWord) {
    const ip = row(posWord), ineg = row(negWord);
    if (ip === undefined || ineg === undefined) return null;
    const v = new Float32Array(dim);
    let n = 0;
    for (let d = 0; d < dim; d++) { v[d] = q[ip * dim + d] - q[ineg * dim + d]; n += v[d] * v[d]; }
    n = Math.sqrt(n) || 1;
    for (let d = 0; d < dim; d++) v[d] /= n;
    return v;
  }

  // spelling fallback for a miss: vocab words sharing a prefix, nearest in length, most frequent.
  function suggest(word, k = 8) {
    word = (word || "").toLowerCase().trim();
    if (!word) return [];
    for (const p of [5, 4, 3, 2]) {
      if (word.length < p) continue;
      const pre = word.slice(0, p), cands = [];
      for (let i = 0; i < count && cands.length < 600; i++)
        if (words[i].startsWith(pre)) cands.push(i);
      if (cands.length) {
        cands.sort((a, b) => Math.abs(words[a].length - word.length) - Math.abs(words[b].length - word.length) || a - b);
        return cands.slice(0, k).map(i => words[i]);
      }
    }
    return [];
  }

  return { load, has, row, wordAt, allAxes, axisById, vocabList, metaInfo,
           axisScores, nearest, projectWord, customAxis, suggest, dotRowFloat,
           localPCA, dotVec, projectRowVec,
           get dim() { return dim; }, get count() { return count; } };
})();
