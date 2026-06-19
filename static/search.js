/* search.js — controller for the "why vector search?" page.
 * One query drives two searches over the SAME corpus (top-N frequent words): keyword (trigram
 * Jaccard, static/lexical.js) vs vector (cosine, Embed.nearestIn). The contrast is the lesson. */
(() => {
  const CORPUS = 8000;          // shared search corpus = the N most frequent words
  const K = 12;                 // results shown per column
  const state = { word: "king" };
  let idx = null;               // lexical inverted index over the corpus
  let _mapRaf = 0;              // deferred meaning-map render handle
  let graph = null;             // {nodes:[{word}], edges:[{a,b,w}], expanded:Set}

  // --- leaf concept graph model ---
  function seedGraph(w) {
    SearchVis.resetGraphLayout();
    graph = { nodes: [{ word: w }], edges: [], expanded: new Set() };
    expandNode(w, 6);
  }
  function expandNode(w, k = 5) {
    if (graph.expanded.has(w)) return;
    graph.expanded.add(w);
    for (const n of Embed.nearestIn(w, CORPUS, k)) {
      if (!graph.nodes.some(nd => nd.word === n.word)) graph.nodes.push({ word: n.word });
      if (!graph.edges.some(e => (e.a === w && e.b === n.word) || (e.a === n.word && e.b === w)))
        graph.edges.push({ a: w, b: n.word, w: n.score });
    }
  }
  function renderGraph() {
    SearchVis.drawConceptGraph($("graph"), graph, onGraphClick);
    window.__graphSize = graph.nodes.length;   // test hook: observe expansion
  }
  function onGraphClick(word) {
    if (!graph.expanded.has(word) && graph.nodes.length < 36) { expandNode(word); renderGraph(); }
    else setWord(word);   // re-center on an already-grown node
  }

  const $ = id => document.getElementById(id);
  const el = (t, c, x) => { const e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; };

  function setWord(w) {
    w = (w || "").trim().toLowerCase();
    if (!w) return;
    state.word = w; $("qInput").value = w;
    render(); history.replaceState(null, "", "#q=" + encodeURIComponent(w));
  }

  function resultItem(r, overlapSet, scoreFmt) {
    const li = el("li");
    if (overlapSet.has(r.word)) li.classList.add("ov");
    const b = el("button", "wlink", r.word); b.onclick = () => setWord(r.word);
    li.appendChild(b);
    li.appendChild(el("span", "score", scoreFmt(r.score)));
    return li;
  }

  function render() {
    const w = state.word;
    const inVocab = Embed.row(w) !== undefined;

    // keyword search works on any string (it matches letters); vector needs the word in the embedding
    const kw = Lexical.lexicalNearest(w, idx, K);
    const vec = inVocab ? Embed.nearestIn(w, CORPUS, K) : [];

    const notice = $("notice");
    if (!inVocab && !kw.length) {
      notice.style.display = "block";
      notice.innerHTML = "";
      notice.appendChild(el("span", null, `“${w}” isn’t in the embedding and has no close spellings. Try another word.`));
      $("stageContrast").style.display = "none"; $("stageMap").style.display = "none"; $("stageGraph").style.display = "none";
      return;
    }
    notice.style.display = "none";
    $("stageContrast").style.display = "block";

    // overlap = words appearing in BOTH columns (spelling and meaning happen to agree)
    const kwSet = new Set(kw.map(r => r.word)), vecSet = new Set(vec.map(r => r.word));
    const overlap = new Set([...kwSet].filter(x => vecSet.has(x)));

    const kwList = $("kwList"); kwList.innerHTML = "";
    kw.forEach(r => kwList.appendChild(resultItem(r, overlap, s => s.toFixed(2))));
    const vecList = $("vecList"); vecList.innerHTML = "";
    if (!vec.length) vecList.appendChild(el("li", "empty", inVocab ? "—" : "(not in the embedding)"));
    vec.forEach(r => vecList.appendChild(resultItem(r, overlap, s => s.toFixed(2))));

    // recall gauge: of the top-10 nearest-in-MEANING words, how many would keyword search never surface?
    const kwReach = new Set(Lexical.lexicalNearest(w, idx, 50).map(r => r.word));
    const top = vec.slice(0, 10);
    const missed = top.filter(r => !kwReach.has(r.word)).length;
    const recall = $("recall");
    if (top.length) {
      recall.innerHTML = `<b class="big">${missed} of ${top.length}</b> words closest in <i>meaning</i> to ` +
        `<b>${w}</b> are invisible to keyword search <span class="muted">— not even in its top 50 by spelling.</span>`;
      recall.style.display = "block";
    } else recall.style.display = "none";

    // STAGE 2 — meaning-map. Deferred via rAF so the columns paint before the (heavy, first-time) PCA.
    cancelAnimationFrame(_mapRaf);
    if (inVocab) {
      _mapRaf = requestAnimationFrame(() => {
        const coords = Embed.layout2d(CORPUS);
        SearchVis.drawMeaningMap($("map"), coords, {
          query: Embed.row(w),
          vec: vec.map(r => r.idx).filter(i => i < CORPUS),
          kw: kw.map(r => Embed.row(r.word)).filter(i => i !== undefined && i < CORPUS),
          wordAt: Embed.wordAt, onWord: setWord,
        });
        $("stageMap").style.display = "block";
        // STAGE 3 — leaf concept graph (cheap; same frame)
        seedGraph(w); renderGraph(); $("stageGraph").style.display = "block";
      });
    } else { $("stageMap").style.display = "none"; $("stageGraph").style.display = "none"; }
  }

  async function init() {
    const info = await Embed.load();
    const corpus = Embed.vocabList().slice(0, CORPUS);
    idx = Lexical.buildIndex(corpus);
    $("meta").textContent = `searching the ${corpus.length.toLocaleString()} most common words · ${info.model}`;

    const h = new URLSearchParams(location.hash.slice(1));
    if (h.get("q")) state.word = h.get("q").toLowerCase();
    $("qInput").value = state.word;

    $("q").addEventListener("submit", e => { e.preventDefault(); setWord($("qInput").value); });
    document.querySelectorAll("[data-q]").forEach(b => b.addEventListener("click", () => setWord(b.dataset.q)));
    window.addEventListener("hashchange", () => {
      const p = new URLSearchParams(location.hash.slice(1));
      if (p.get("q") && p.get("q").toLowerCase() !== state.word) setWord(p.get("q"));
    });
    window.__expand = onGraphClick;   // test hook: grow the graph from a named leaf
    render();
  }

  init().catch(err => { $("meta").textContent = "load error: " + err.message; console.error(err); });
})();
