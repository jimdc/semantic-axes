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

/* SAEBackend — the substrate swap, as a navigable word<->feature graph.
 *
 * Reads SAE features precomputed by tools/build_sae.py (Neuronpedia search-all) and fills the SAME
 * QueryResult shape as the static backend — but the SAE substrate is READ, not steered (its sibling
 * intrusive-thought owns steering). The lesson here is structural: a word is a BUNDLE of features
 * firing at once; a feature is SHARED across many words; "similar" means "shares features". Two moves
 * make that legible instead of noisy:
 *   - distinctiveness re-rank. On a bare word at layer 20 the top-by-activation features are mostly
 *     boilerplate (start-of-document, proper-nouns) that fire on ~every word and so say nothing about
 *     THIS one. We measure each feature's document frequency across the set and split a word's bundle
 *     into distinctive (high act*idf — where the meaning is) and generic (fires on >=GENERIC of the
 *     set — demoted). The de-noising IS the intuition pump.
 *   - the receptive field. wordsFiringOn(feature) returns the words that light a feature up, so the UI
 *     can pivot word -> feature -> words -> feature ... — the SAE counterpart to the static side's
 *     click-to-re-center wander.
 * The continuous-geometry views (spectrum, scatter, custom axis, PCA) don't apply; the UI hides them. */
class SAEBackend {
  constructor() { this.name = "sae"; this._ready = null; this.data = null; this.df = null; this.N = 0; }

  ready(base = "") {
    return (this._ready ||= fetch(base + "data/sae_features.json")
      .then(r => r.ok ? r.json() : Promise.reject(new Error("sae_features.json missing — run tools/build_sae.py")))
      .then(d => { this.data = d; this._index(); return { nAxes: `${d.count} words`, model: `${d.model}/${d.sourceSet} SAE` }; }));
  }

  // document frequency per feature across the whole set — the measure of how generic a feature is
  _index() {
    this.N = Object.keys(this.data.words).length;
    this.df = new Map(); this.labelOf = new Map();
    for (const feats of Object.values(this.data.words))
      for (const f of feats) {
        this.df.set(f.id, (this.df.get(f.id) || 0) + 1);
        if (!this.labelOf.has(f.id)) this.labelOf.set(f.id, f.label);
      }
  }

  has(word) { return !!(this.data && this.data.words[word]); }
  vocab() { return this.data ? Object.keys(this.data.words) : []; }
  model() { return this.data ? this.data.model : ""; }

  _fires(id) { return this.df.get(id) || 0; }
  _generic(id) { return this._fires(id) / this.N >= SAEBackend.GENERIC; }
  _idf(id) { return Math.log(this.N / (this._fires(id) || 1)); }

  // a word's bundle, split into the distinctive features (high act*idf, where the meaning lives) and
  // the generic ones (fire on most text). distinctive ranked by distinctiveness, generic by activation.
  query(word, { axesK = 8 } = {}) {
    const w = this.data && this.data.words[word];
    if (!w) return { word, backend: this.name, axes: [], neighbors: [], missing: true };
    const rows = w.map(f => ({
      id: f.id, label: f.label, act: f.act, featureId: f.id, poles: null, confidence: null,
      fires: this._fires(f.id), generic: this._generic(f.id), distinct: f.act * this._idf(f.id),
    }));
    const distinctive = rows.filter(r => !r.generic).sort((a, b) => b.distinct - a.distinct).slice(0, axesK);
    const generic = rows.filter(r => r.generic).sort((a, b) => b.act - a.act);
    const maxAct = Math.max(1, ...distinctive.map(r => r.act));
    distinctive.forEach(r => r.score = r.act / maxAct);
    return { word, backend: this.name, axes: distinctive, generic, neighbors: this.neighbors(word) };
  }

  // the receptive field of a feature: the words in the set that fire it, strongest first.
  wordsFiringOn(featureId, k = 40) {
    const out = [];
    for (const [word, feats] of Object.entries(this.data.words)) {
      const f = feats.find(x => x.id === featureId);
      if (f) out.push({ word, act: f.act });
    }
    out.sort((a, b) => b.act - a.act);
    return { label: this.labelOf.get(featureId) || featureId, fires: out.length, words: out.slice(0, k) };
  }

  // neighbors = words sharing DISTINCTIVE features, ranked by summed idf (a rare shared feature is
  // strong evidence; a semi-common one is weak). Counting generic features would make every word the
  // neighbor of every other. Each neighbor carries WHY: the shared feature labels, most distinctive first.
  neighbors(word, k = 14) {
    const w = this.data && this.data.words[word];
    if (!w) return [];
    const mine = new Map(w.filter(f => !this._generic(f.id)).map(f => [f.id, f.label]));
    const out = [];
    for (const [other, feats] of Object.entries(this.data.words)) {
      if (other === word) continue;
      let score = 0; const shared = [];
      for (const f of feats) if (mine.has(f.id)) { score += this._idf(f.id); shared.push({ label: mine.get(f.id), idf: this._idf(f.id) }); }
      if (shared.length) { shared.sort((a, b) => b.idf - a.idf); out.push({ word: other, score, shared: shared.map(s => s.label) }); }
    }
    return out.sort((a, b) => b.score - a.score).slice(0, k);
  }
}
SAEBackend.GENERIC = 0.30;   // a feature firing on >=30% of the set is "generic" (low signal for any one word)
// a feature's page on Neuronpedia, e.g. gpt2-small/11-res-jb/12786
SAEBackend.featureUrl = (model, id) => `https://www.neuronpedia.org/${model}/${id}`;
