/* app.js — central state + render loop + event wiring. (state -> recompute -> re-render);
 * the click-to-re-center loop is the discovery engine. Two swappable backends behind one
 * QueryResult contract: static embeddings (geometric axes) and SAE features (learned concepts). */
(() => {
  const backends = { static: new StaticEmbeddingBackend(), sae: new SAEBackend() };
  const state = { word: "king", axisId: null, scatterX: null, scatterY: null, custom: null, backendName: "static" };
  const backend = () => backends[state.backendName];

  const $ = id => document.getElementById(id);
  const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const chip = (word, parent) => { const c = el("button", "chip", word); c.onclick = () => setWord(word); parent.appendChild(c); return c; };
  const wordLink = (word, parent) => { const s = el("button", "wlink", word); s.onclick = () => setWord(word); parent.appendChild(s); return s; };

  // good "try one of these" words — content words that show the contrast, filtered to what a backend has
  // (the raw vocab is frequency-ordered, so its head is stopwords — useless as suggestions).
  const SAMPLES = ["king", "queen", "mother", "doctor", "ocean", "wolf", "scientist", "money", "music", "war", "paris", "hug"];
  const suggestWords = be => { const ok = SAMPLES.filter(w => be.has(w)); return ok.length ? ok : be.vocab().slice(0, 12); };

  function setWord(w) {
    w = (w || "").trim().toLowerCase();
    if (!w) return;
    state.word = w; state.axisId = null; state.scatterX = null; state.scatterY = null;
    $("queryInput").value = w;
    render();
  }
  function setAxis(id) { state.axisId = (state.axisId === id ? null : id); render(); }

  function setBackend(name) {
    if (name === state.backendName) return;
    const ready = name === "compare"
      ? Promise.all([backends.static.ready(), backends.sae.ready()])
      : (backends[name] ? backends[name].ready() : Promise.reject(new Error("unknown substrate")));
    ready.then(() => {
      state.backendName = name;
      document.querySelectorAll("#backendToggle button").forEach(b => b.classList.toggle("active", b.dataset.backend === name));
      render();
    }).catch(err => { $("backendHint").textContent = "· " + err.message; });
  }

  function render() {
    if (state.backendName === "compare") { renderCompare(); updateHash(); return; }
    $("compareSection").style.display = "none";
    const be = backend(), isSAE = be.name === "sae";
    const res = be.query(state.word, { axesK: 8 });
    if (!isSAE) $("scatterWord").textContent = state.word;

    const notice = $("notice");
    const core = ["axisSection", "neighborSection"].map($);
    const staticOnly = ["spectrumWrap", "scatterSection", "customSection", "discoverySection"].map($);

    // not-in-vocab guard
    if (res.missing) {
      notice.style.display = "block"; notice.innerHTML = "";
      if (isSAE) {
        notice.appendChild(el("span", null, `“${state.word}” isn’t in the ${be.vocab().length}-word SAE demo set. Try: `));
        suggestWords(be).forEach(w => { const c = chip(w, notice); c.style.margin = "0 .15rem"; });
      } else {
        const sugg = Embed.suggest(state.word, 8);
        notice.appendChild(el("span", null, `“${state.word}” isn’t in the vocabulary` + (sugg.length ? " — did you mean: " : ". Try another word.")));
        sugg.forEach(w => { const c = chip(w, notice); c.style.margin = "0 .15rem"; });
      }
      [...core, ...staticOnly].forEach(s => s.style.display = "none");
      updateHash(); return;
    }
    notice.style.display = "none";
    core.forEach(s => s.style.display = "block");

    // axes view — static: diverging canvas bars; SAE: the bundle of features that fire on the word
    if (isSAE) {
      $("axisBars").style.display = "none"; $("featureList").style.display = "block";
      $("axisEyebrow").innerHTML = `the bundle of features that fire on <b>${state.word}</b> ` +
        `<span class="muted">— ${be.model()} SAE · the model’s own learned concepts</span>`;
      $("axisHint").textContent = "a word is many features at once. Click a feature to see what else fires it — then click a word to open its bundle.";
      renderFeatureList(res, be);
      staticOnly.forEach(s => s.style.display = "none");
    } else {
      $("featureList").style.display = "none"; $("axisBars").style.display = "block";
      $("axisEyebrow").innerHTML = `where <b>${state.word}</b> sits — its most salient axes`;
      $("axisHint").textContent = "bar points toward the pole the word leans to; length = strength. Click an axis to fan its neighbors out along it.";
      Vis.drawAxisBars($("axisBars"), res.axes, setAxis, state.axisId);
      renderStaticExtras(be, res);
    }

    // neighbors (both substrates). SAE: chips of words sharing distinctive features, with a concrete
    // "why these?" example so the basis of similarity is shown, not just asserted.
    $("nbrEyebrow").textContent = isSAE ? "words that share its distinctive features" : "nearest in meaning";
    const nb = $("neighbors"); nb.innerHTML = "";
    res.neighbors.forEach(n => { const c = chip(n.word, nb);
      c.title = isSAE ? (n.shared ? "shares: " + n.shared.map(clean).join(", ") : `shares ${n.score}`) : `similarity ${n.score.toFixed(2)}`; });
    const why = $("nbrWhy"), top = res.neighbors[0];
    if (isSAE && top && top.shared && top.shared.length) {
      why.style.display = "block";
      why.innerHTML = `why these? <b>${state.word}</b> ≈ <b>${top.word}</b> — both fire ` +
        top.shared.slice(0, 2).map(s => `“${clean(s)}”`).join(", ");
    } else why.style.display = "none";

    updateHash();
  }

  const clean = s => (s || "").replace(/\s+/g, " ").trim();

  // COMPARE — the two-substrate confrontation. The same word, decomposed two ways, side by side:
  // LEFT the curated axes WE named (a handful, signed, continuous); RIGHT the features the MODEL grew
  // (sparse, emergent, often not concepts we'd name). The lesson is that the two bases only partly align.
  function renderCompare() {
    ["axisSection", "spectrumWrap", "neighborSection", "scatterSection", "customSection", "discoverySection"]
      .forEach(id => $(id).style.display = "none");
    $("notice").style.display = "none";
    $("compareSection").style.display = "block";
    $("cmpWord").textContent = state.word;
    const st = backends.static, sa = backends.sae;
    const grid = document.querySelector("#compareSection .cmp-grid");
    const sRows = $("cmpStaticRows"), aRows = $("cmpSaeRows"), punch = $("cmpPunch");
    sRows.innerHTML = ""; aRows.innerHTML = ""; punch.innerHTML = "";

    if (!sa.has(state.word) || !st.has(state.word)) {     // nothing to line up — show only the prompt
      grid.style.display = "none";
      const where = !sa.has(state.word) ? `the ${sa.vocab().length}-word SAE set` : "the static vocabulary";
      punch.appendChild(el("span", null, `“${state.word}” isn’t in ${where}, so there’s nothing to line up. Try a word in both — e.g. `));
      suggestWords(sa).forEach(w => { const c = chip(w, punch); c.style.margin = "0 .15rem"; });
      return;
    }
    grid.style.display = "grid";

    // LEFT: our curated axes — magnitude bar + signed value toward the pole the word leans to
    const sax = st.query(state.word, { axesK: 7 }).axes;
    const maxAbs = Math.max(...sax.map(a => Math.abs(a.score)), 1e-6);
    sax.forEach(a => {
      const lean = a.poles ? (a.score >= 0 ? a.poles[0] : a.poles[1]) : "";
      sRows.appendChild(cmpRow(a.label, `${a.score >= 0 ? "+" : "−"}${Math.abs(a.score).toFixed(2)} ${lean}`,
                               Math.abs(a.score) / maxAbs, "static"));
    });

    // RIGHT: the model's features — activation bar, label links out to Neuronpedia
    const ares = sa.query(state.word, { axesK: 6 });
    const amax = Math.max(...ares.axes.map(a => a.act), 1);
    ares.axes.forEach(a => {
      const url = SAEBackend.featureUrl(sa.model(), a.id);
      aRows.appendChild(cmpRow(clean(a.label), String(a.act), a.act / amax, "sae", url));
    });
    if (ares.generic && ares.generic.length)
      aRows.appendChild(el("div", "cmp-generic", "+ generic: " + ares.generic.slice(0, 3).map(f => clean(f.label)).join(" · ")));

    const odd = ares.axes[0] ? `here it’s “${clean(ares.axes[0].label)}”` : "here it has only generic, fire-on-everything features";
    punch.innerHTML = `Left: the directions <b>we</b> chose — a handful, signed, continuous. ` +
      `Right: the features the <b>model</b> grew — sparse, emergent, often not concepts we’d name (${odd}). ` +
      `Two bases for one word; they line up only in part.`;
  }

  // one row in either compare column: name (+ optional link), right-aligned value, a magnitude bar
  function cmpRow(name, value, frac, kind, url) {
    const row = el("div", "cmp-row"), lab = el("div", "cmp-lab");
    const nm = url ? el("a", "cmp-name cmp-feat", name) : el("span", "cmp-name", name);
    if (url) { nm.href = url; nm.target = "_blank"; nm.rel = "noopener"; }
    lab.appendChild(nm); lab.appendChild(el("span", "cmp-val", value));
    const track = el("div", "cmp-track"), bar = el("div", "cmp-bar " + kind);
    bar.style.width = Math.max(4, frac * 100) + "%"; track.appendChild(bar);
    row.appendChild(lab); row.appendChild(track); return row;
  }

  // SAE bundle: distinctive features as rows you can OPEN (click a feature -> the words that also fire
  // it, in-app -> click a word to re-center). Generic features get demoted to one greyed line, and the
  // Neuronpedia page drops to a small per-row ↗ so the label click does the in-app pivot, not a leave.
  function renderFeatureList(res, be) {
    const fl = $("featureList"); fl.innerHTML = ""; const model = be.model();
    if (!res.axes.length)
      fl.appendChild(el("p", "hint", `every feature firing on “${res.word}” is generic (fires on most text) — the SAE has no sharp concept for it here.`));
    res.axes.forEach(a => fl.appendChild(featureRow(a, be, model)));
    if (res.generic && res.generic.length) {
      const g = el("div", "feat-generic");
      g.appendChild(el("span", "feat-glabel", "also fires on generic features (low signal): "));
      res.generic.forEach((f, i) => {
        if (i) g.appendChild(document.createTextNode(" · "));
        const a = el("a", "feat-gitem", clean(f.label));
        a.href = SAEBackend.featureUrl(model, f.id); a.target = "_blank"; a.rel = "noopener";
        g.appendChild(a);
      });
      fl.appendChild(g);
    }
  }

  function featureRow(a, be, model) {
    const row = el("div", "feat-row");
    const head = el("div", "feat-head");
    const lab = el("button", "feat-label", clean(a.label));
    lab.appendChild(el("span", "feat-act", " " + a.act));
    head.appendChild(lab);
    head.appendChild(el("span", "feat-meta", `· fires on ${a.fires} of ${be.N} words`));
    const np = el("a", "feat-np", "↗"); np.href = SAEBackend.featureUrl(model, a.id);
    np.target = "_blank"; np.rel = "noopener"; np.title = "open this feature on Neuronpedia";
    head.appendChild(np);
    const track = el("div", "feat-track"), meter = el("div", "feat-meter");
    meter.style.width = Math.max(6, a.score * 100) + "%"; track.appendChild(meter);
    const words = el("div", "feat-words"); words.style.display = "none";
    row.appendChild(head); row.appendChild(track); row.appendChild(words);
    lab.onclick = () => {
      if (words.style.display !== "none") { words.style.display = "none"; return; }
      if (!words.dataset.loaded) {
        const rf = be.wordsFiringOn(a.id), others = rf.words.filter(x => x.word !== state.word);
        words.appendChild(el("span", "feat-words-lead", others.length ? "also fires on: " : `fires only on “${state.word}” in this set — an idiosyncratic feature.`));
        others.forEach(x => { const c = chip(x.word, words); c.title = `activation ${x.act}`; });
        words.dataset.loaded = "1";
      }
      words.style.display = "block";
    };
    return row;
  }

  // static-only geometric views: spectrum, two-axis scatter, custom axis, PCA discovery
  function renderStaticExtras(be, res) {
    const sw = $("spectrumWrap");
    if (state.axisId) {
      const ax = Embed.axisById(state.axisId);
      sw.style.display = "block";
      $("spectrumTitle").innerHTML = `<b>${state.word}</b>’s neighbors along the <b>${ax.label}</b> axis ` +
        `<span class="muted">— click any word to explore it</span>`;
      Vis.drawSpectrum($("spectrum"), be.exploreAlongAxis(state.word, state.axisId, 22), ax, setWord);
    } else sw.style.display = "none";

    const sx = state.scatterX || res.axes[0]?.id, sy = state.scatterY || (res.axes[1]?.id || res.axes[0]?.id);
    if (sx && sy) {
      $("scatterSection").style.display = "block";
      $("scatterXsel").value = sx; $("scatterYsel").value = sy;
      Vis.drawScatter($("scatter"), be.scatter(state.word, sx, sy, 28), Embed.axisById(sx), Embed.axisById(sy), setWord);
    } else $("scatterSection").style.display = "none";

    renderCustom(be);
    renderDiscovery(be.discoverAxes(state.word, { nbr: 60, k: 3 }));
  }

  function renderCustom(be) {
    const res = $("customResult"), c = state.custom;
    $("customSection").style.display = "block";
    if (!c || !c.pos || !c.neg) { res.style.display = "none"; return; }
    res.style.display = "block";
    $("customNeg").value = c.neg; $("customPos").value = c.pos;
    const ax = be.customAxis(c.pos, c.neg);
    if (!ax) {
      const miss = [c.neg, c.pos].filter(w => !Embed.has(w));
      $("customErr").textContent = `not in vocabulary: ${miss.join(", ")}`;
      $("customTitle").textContent = ""; $("customSpectrum").style.display = "none"; return;
    }
    $("customErr").textContent = ""; $("customSpectrum").style.display = "block";
    const sc = be.scoreOn(state.word, ax);
    $("customTitle").innerHTML = `<b>${state.word}</b> on your axis <b>${c.neg} ⟷ ${c.pos}</b>: ` +
      `<b>${sc >= 0 ? "+" : "−"}${Math.abs(sc).toFixed(2)}</b> ` +
      `<span class="muted">(leans ${sc >= 0 ? c.pos : c.neg}) — click any word to explore</span>`;
    Vis.drawSpectrum($("customSpectrum"), be.exploreAlongCustom(state.word, ax, 22), ax, setWord);
  }

  function renderDiscovery(disc) {
    const box = $("discovery"); box.innerHTML = "";
    $("discoverySection").style.display = disc.length ? "block" : "none";
    disc.forEach((d, i) => {
      const row = el("div", "disc-row");
      const head = el("span", "disc-label");
      head.textContent = d.label === "(unnamed)" ? `direction ${i + 1}` : `≈ ${d.label}`;
      if (d.label !== "(unnamed)") head.appendChild(el("span", "muted", ` cos ${d.cos.toFixed(2)}`));
      row.appendChild(head);
      const poles = el("span", "disc-poles");
      d.negWords.forEach(w => wordLink(w, poles));
      poles.appendChild(el("span", "disc-arrow", " ⟷ "));
      d.posWords.forEach(w => wordLink(w, poles));
      row.appendChild(poles); box.appendChild(row);
    });
  }

  function updateHash() {
    const p = new URLSearchParams(); p.set("w", state.word);
    if (state.backendName !== "static") p.set("be", state.backendName);
    if (state.axisId) p.set("axis", state.axisId);
    if (state.custom) { p.set("cx", state.custom.pos); p.set("cn", state.custom.neg); }
    history.replaceState(null, "", "#" + p.toString());
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("w")) state.word = p.get("w");
    if (p.get("be") && (p.get("be") === "compare" || backends[p.get("be")])) state.backendName = p.get("be");
    if (p.get("axis")) state.axisId = p.get("axis");
    if (p.get("cx") && p.get("cn")) state.custom = { pos: p.get("cx"), neg: p.get("cn") };
  }

  async function init() {
    const info = await backends.static.ready();
    $("metaLine").textContent =
      `${Embed.count.toLocaleString()} words · ${info.nAxes} axes · ${info.model} (static embeddings)`;

    const bank = backends.static.axisBank();
    [["scatterXsel", v => state.scatterX = v], ["scatterYsel", v => state.scatterY = v]].forEach(([id, set]) => {
      const sel = $(id);
      bank.forEach(a => { const o = el("option", null, a.id); o.value = a.id; sel.appendChild(o); });
      sel.onchange = () => { set(sel.value); render(); };
    });

    readHash();
    if (state.backendName === "sae" || state.backendName === "compare") {
      try {
        await (state.backendName === "compare" ? Promise.all([backends.static.ready(), backends.sae.ready()]) : backends.sae.ready());
        document.querySelectorAll("#backendToggle button").forEach(b => b.classList.toggle("active", b.dataset.backend === state.backendName));
      } catch (e) { state.backendName = "static"; }
    }
    $("queryInput").value = state.word;
    $("queryForm").addEventListener("submit", e => { e.preventDefault(); setWord($("queryInput").value); });
    $("customForm").addEventListener("submit", e => {
      e.preventDefault();
      const pos = $("customPos").value.trim().toLowerCase(), neg = $("customNeg").value.trim().toLowerCase();
      if (pos && neg) { state.custom = { pos, neg }; render(); }
    });
    document.querySelectorAll("#backendToggle button").forEach(b => b.addEventListener("click", () => setBackend(b.dataset.backend)));
    window.addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(render, 120); });
    window.addEventListener("hashchange", () => {
      const p = new URLSearchParams(location.hash.slice(1));
      const w = (p.get("w") || "").toLowerCase(), ax = p.get("axis") || null;
      const custom = (p.get("cx") && p.get("cn")) ? { pos: p.get("cx"), neg: p.get("cn") } : null;
      if (w && (w !== state.word || ax !== state.axisId || JSON.stringify(custom) !== JSON.stringify(state.custom))) {
        if (w !== state.word) state.scatterX = state.scatterY = null;
        state.word = w; state.axisId = ax; state.custom = custom; $("queryInput").value = w; render();
      }
    });
    document.querySelectorAll("[data-sample]").forEach(b => b.addEventListener("click", () => setWord(b.dataset.sample)));
    render();
  }

  init().catch(err => { $("metaLine").textContent = "load error: " + err.message; console.error(err); });
})();
