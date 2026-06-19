/* backend.js — the SemanticBackend seam.
 *
 * THE LOAD-BEARING ABSTRACTION. The front-end (app.js / vis.js) only ever consumes the
 * uniform shapes below. A backend decides HOW to fill them in. Today: StaticEmbeddingBackend
 * (GloVe + axis bank, all in-browser). Later (Phase 4): SAEBackend (Neuronpedia SAE features)
 * implements the SAME interface, so the UI never changes — the SAE swap is a new class + a
 * toggle, not a rewrite.
 *
 * Shapes:
 *   QueryResult = { word, backend, axes:[Axis], neighbors:[{word,score}] }
 *   Axis        = { id, label, score, poles:[pos,neg]|null, confidence:number|null,
 *                   featureId:string|null }
 *
 * Interface every backend implements:
 *   ready()                        -> Promise (resolves when loaded)
 *   has(word)                      -> bool
 *   axisBank()                     -> [{id,label,poles}]            (all known axes)
 *   query(word, {axesK, nbrK})     -> QueryResult                   (salient axes + neighbors)
 *   exploreAlongAxis(word, axisId, k) -> [{word, score}]            (neighborhood sorted on axis)
 *   projectOnAxis(word, axisId)    -> number|null
 */

class StaticEmbeddingBackend {
  constructor() { this.name = "static"; this._ready = null; }

  ready(base = "") { return (this._ready ||= Embed.load(base)); }

  has(word) { return Embed.has(word); }

  axisBank() {
    return Embed.allAxes().map(a => ({ id: a.id, label: a.label, poles: a.poles }));
  }

  query(word, { axesK = 8, nbrK = 14 } = {}) {
    if (!Embed.has(word)) return { word, backend: this.name, axes: [], neighbors: [], missing: true };
    const scored = Embed.axisScores(word)
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
    const axes = scored.slice(0, axesK).map(a => ({
      id: a.id, label: a.label, score: a.score, poles: a.poles,
      confidence: null, featureId: null,
    }));
    return { word, backend: this.name, axes, neighbors: Embed.nearest(word, nbrK),
             allAxisScores: scored };   // full ranking kept for the axis picker
  }

  // a word's neighborhood, laid out along an axis object (the spectrum view)
  _along(word, ax, k = 24) {
    if (!ax || !Embed.has(word)) return [];
    const nbrs = Embed.nearest(word, k);
    const self = { word, score: Embed.projectWord(word, ax.vec), isSelf: true };
    return [self, ...nbrs.map(n => ({ word: n.word, score: Embed.projectWord(n.word, ax.vec) }))]
      .filter(x => x.score !== null)
      .sort((a, b) => a.score - b.score);
  }
  exploreAlongAxis(word, axisId, k = 24) { return this._along(word, Embed.axisById(axisId), k); }

  // a LIVE axis from two anchor words (the custom-axis builder). Same shape as a bank axis,
  // so it drives the spectrum identically. poles=[pos,neg] -> pos is the +/right side.
  customAxis(posW, negW) {
    const vec = Embed.customAxis(posW, negW);
    return vec ? { id: "custom", label: `${negW} ↔ ${posW}`, pos: posW, neg: negW, poles: [posW, negW], vec } : null;
  }
  exploreAlongCustom(word, ax, k = 22) { return this._along(word, ax, k); }
  scoreOn(word, ax) { return Embed.projectWord(word, ax.vec); }

  projectOnAxis(word, axisId) {
    const ax = Embed.axisById(axisId);
    return ax ? Embed.projectWord(word, ax.vec) : null;
  }

  // scatter a word's neighborhood on two named axes (Phase 2)
  scatter(word, axisIdX, axisIdY, k = 30) {
    const ax = Embed.axisById(axisIdX), ay = Embed.axisById(axisIdY);
    if (!ax || !ay || !Embed.has(word)) return [];
    const pts = [{ word, isSelf: true }, ...Embed.nearest(word, k)];
    return pts.map(p => ({ word: p.word, isSelf: !!p.isSelf,
                           x: Embed.projectWord(p.word, ax.vec),
                           y: Embed.projectWord(p.word, ay.vec) }));
  }

  // UNSUPERVISED discovery: the top variance directions of a word's OWN neighborhood (local PCA),
  // i.e. axes nobody pre-authored. Each is tentatively labeled by its nearest curated axis (cosine)
  // and described by its extreme neighborhood words. Model-proposed, not authoritative.
  discoverAxes(word, { nbr = 60, k = 3, poleN = 3 } = {}) {
    const pca = Embed.localPCA(word, nbr, k);
    if (!pca) return [];
    const bank = Embed.allAxes();
    return pca.comps.map(u => {
      let best = null, bestc = 0;
      for (const ax of bank) { const c = Embed.dotVec(u, ax.vec); if (Math.abs(c) > Math.abs(bestc)) { bestc = c; best = ax; } }
      const sign = bestc >= 0 ? 1 : -1;                 // orient toward the labeled axis's + pole
      const proj = pca.idxs.map(ix => ({ word: Embed.wordAt(ix), s: sign * Embed.projectRowVec(ix, u) }))
        .filter(p => p.word !== word && !StaticEmbeddingBackend.STOP.has(p.word))   // keep poles to content words
        .sort((a, b) => b.s - a.s);
      const labeled = Math.abs(bestc) >= 0.35;   // nearest curated axis, only if a real resemblance
      return {
        label: labeled ? best.id : "(unnamed)",
        labelFull: labeled ? best.label : null,
        cos: Math.abs(bestc),
        posWords: proj.slice(0, poleN).map(p => p.word),
        negWords: proj.slice(-poleN).map(p => p.word).reverse(),
      };
    });
  }
}

// function/discourse words to keep OUT of discovered-axis pole labels (they're still in the PCA)
StaticEmbeddingBackend.STOP = new Set((
  "the a an of to and in is was for on with as at by it its his her their my your you he she they we " +
  "but so what even that this these those then than when who which not no all any some such more most very " +
  "just only also here there how why will would can could do does did have has had been being into out up " +
  "down over about after before however finally first second another each both either"
).split(" "));

/* SAEBackend — the substrate swap. Reads SAE features precomputed by tools/build_sae.py
 * (Neuronpedia search-all) and fills the SAME QueryResult shape as the static backend:
 *   each top SAE feature -> Axis {id, label:explanation, score:activation, featureId}.
 * Neighbors = other words that fire on the same features. The SAE substrate has no continuous
 * geometry, so the static-only geometric views (spectrum, scatter, custom axis, PCA discovery)
 * don't apply — the front-end hides them in this mode. */
class SAEBackend {
  constructor() { this.name = "sae"; this._ready = null; this.data = null; }

  ready(base = "") {
    return (this._ready ||= fetch(base + "data/sae_features.json")
      .then(r => r.ok ? r.json() : Promise.reject(new Error("sae_features.json missing — run tools/build_sae.py")))
      .then(d => { this.data = d; return { nAxes: `${d.count} words`, model: `${d.model}/${d.sourceSet} SAE` }; }));
  }

  has(word) { return !!(this.data && this.data.words[word]); }
  vocab() { return this.data ? Object.keys(this.data.words) : []; }
  model() { return this.data ? this.data.model : ""; }

  query(word, { axesK = 8 } = {}) {
    const w = this.data && this.data.words[word];
    if (!w) return { word, backend: this.name, axes: [], neighbors: [], missing: true };
    const maxAct = Math.max(...w.map(f => f.act)) || 1;
    const axes = w.slice(0, axesK).map(f => ({
      id: f.id, label: f.label, score: f.act / maxAct, act: f.act,
      featureId: f.id, poles: null, confidence: null,
    }));
    return { word, backend: this.name, axes, neighbors: this.neighbors(word) };
  }

  // neighbors = words sharing the most top-features (simple overlap)
  neighbors(word, k = 14) {
    const w = this.data && this.data.words[word];
    if (!w) return [];
    const mine = new Set(w.map(f => f.id)), out = [];
    for (const [other, feats] of Object.entries(this.data.words)) {
      if (other === word) continue;
      let shared = 0; for (const f of feats) if (mine.has(f.id)) shared++;
      if (shared > 0) out.push({ word: other, score: shared });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, k);
  }
}
// a feature's page on Neuronpedia, e.g. gpt2-small/11-res-jb/12786
SAEBackend.featureUrl = (model, id) => `https://www.neuronpedia.org/${model}/${id}`;
