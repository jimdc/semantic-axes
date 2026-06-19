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

  function setWord(w) {
    w = (w || "").trim().toLowerCase();
    if (!w) return;
    state.word = w; state.axisId = null; state.scatterX = null; state.scatterY = null;
    $("queryInput").value = w;
    render();
  }
  function setAxis(id) { state.axisId = (state.axisId === id ? null : id); render(); }

  function setBackend(name) {
    if (name === state.backendName || !backends[name]) return;
    backends[name].ready().then(() => {
      state.backendName = name;
      document.querySelectorAll("#backendToggle button").forEach(b => b.classList.toggle("active", b.dataset.backend === name));
      render();
    }).catch(err => { $("backendHint").textContent = "· " + err.message; });
  }

  function render() {
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
        be.vocab().slice(0, 16).forEach(w => { const c = chip(w, notice); c.style.margin = "0 .15rem"; });
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

    // axes view — static: diverging canvas bars; SAE: a list of features that fire on the word
    if (isSAE) {
      $("axisBars").style.display = "none"; $("featureList").style.display = "block";
      $("axisEyebrow").innerHTML = `features that fire on <b>${state.word}</b> ` +
        `<span class="muted">— ${be.model()} SAE · click a feature to open it on Neuronpedia</span>`;
      $("axisHint").textContent = "each bar = how strongly that learned feature activates on the word.";
      renderFeatureList(res.axes, be.model());
      staticOnly.forEach(s => s.style.display = "none");
    } else {
      $("featureList").style.display = "none"; $("axisBars").style.display = "block";
      $("axisEyebrow").innerHTML = `where <b>${state.word}</b> sits — its most salient axes`;
      $("axisHint").textContent = "bar points toward the pole the word leans to; length = strength. Click an axis to fan its neighbors out along it.";
      Vis.drawAxisBars($("axisBars"), res.axes, setAxis, state.axisId);
      renderStaticExtras(be, res);
    }

    // neighbors (both substrates)
    const nb = $("neighbors"); nb.innerHTML = "";
    res.neighbors.forEach(n => { const c = chip(n.word, nb); c.title = isSAE ? `shares ${n.score} features` : `similarity ${n.score.toFixed(2)}`; });

    updateHash();
  }

  // SAE feature list: bar ∝ activation, explanation links to the feature on Neuronpedia
  function renderFeatureList(axes, model) {
    const fl = $("featureList"); fl.innerHTML = "";
    axes.forEach(a => {
      const row = el("div", "feat-row");
      const lab = el("a", "feat-label"); lab.href = SAEBackend.featureUrl(model, a.id);
      lab.target = "_blank"; lab.rel = "noopener"; lab.textContent = a.label;
      lab.appendChild(el("span", "feat-act", " " + a.act));
      const track = el("div", "feat-track"); const meter = el("div", "feat-meter");
      meter.style.width = Math.max(6, a.score * 100) + "%"; track.appendChild(meter);
      row.appendChild(lab); row.appendChild(track); fl.appendChild(row);
    });
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
    if (p.get("be") && backends[p.get("be")]) state.backendName = p.get("be");
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
    if (state.backendName === "sae") { try { await backends.sae.ready(); document.querySelectorAll("#backendToggle button").forEach(b => b.classList.toggle("active", b.dataset.backend === "sae")); } catch (e) { state.backendName = "static"; } }
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
