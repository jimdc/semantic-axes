/* searchvis.js — canvas viz for the search page: the meaning-map (this file) and the
 * leaf concept graph (added in Phase 3). High-DPI fit() mirrors static/vis.js. */
const SearchVis = (() => {
  const ACCENT = "#b3541e";    // vector (meaning) hits
  const ACCENT2 = "#27506b";   // keyword (spelling) hits
  const INK = "#1a1a1a", MUTE = "#9a9a9a", FAINT = "#e4e0d8";
  const SERIF = '"Palatino Linotype", Palatino, Georgia, serif';
  const SANS = "system-ui, -apple-system, sans-serif";

  function fit(canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    return ctx;
  }

  /* Meaning-map: a PCA-2D layout where position ≈ meaning. The query's VECTOR hits form a tight
   * cluster; its KEYWORD hits scatter across the whole map — shared letters ≠ shared meaning.
   * coords: [[x,y],…] for rows 0..N-1.  query/vec/kw: row-index arrays.  wordAt/onWord: callbacks. */
  function drawMeaningMap(canvas, coords, { query, vec, kw, wordAt, onWord }) {
    const N = coords.length;
    const cssW = canvas.parentElement.clientWidth || 720, cssH = 440, pad = 26;
    const ctx = fit(canvas, cssW, cssH);
    ctx.clearRect(0, 0, cssW, cssH);

    // robust extent from the whole sample (stable across queries): mean ± 2.5σ
    let mx = 0, my = 0; for (let i = 0; i < N; i++) { mx += coords[i][0]; my += coords[i][1]; } mx /= N; my /= N;
    let sx = 0, sy = 0; for (let i = 0; i < N; i++) { sx += (coords[i][0] - mx) ** 2; sy += (coords[i][1] - my) ** 2; }
    sx = 2.5 * (Math.sqrt(sx / N) || 1); sy = 2.5 * (Math.sqrt(sy / N) || 1);
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const X = x => clamp(pad + ((x - (mx - sx)) / (2 * sx)) * (cssW - 2 * pad), pad, cssW - pad);
    const Y = y => clamp((cssH - pad) - ((y - (my - sy)) / (2 * sy)) * (cssH - 2 * pad), pad, cssH - pad);

    // faint backdrop = thinned sample, for "this is meaning-space"
    const step = Math.max(1, Math.floor(N / 900));
    ctx.fillStyle = FAINT;
    for (let i = 0; i < N; i += step) { ctx.beginPath(); ctx.arc(X(coords[i][0]), Y(coords[i][1]), 1.3, 0, 7); ctx.fill(); }

    const hits = [];
    const plot = (idx, color, r, bold) => {
      if (idx == null || idx >= N) return;
      const x = X(coords[idx][0]), y = Y(coords[idx][1]), w = wordAt(idx);
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
      ctx.font = (bold ? "bold 12px " : "11px ") + SERIF; ctx.fillStyle = bold ? color : "#444";
      ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(" " + w, x + 2, y);
      hits.push({ x0: x - 4, x1: x + ctx.measureText(w).width + 8, y0: y - 7, y1: y + 7, word: w });
    };
    // keyword hits first (scattered, behind), then vector cluster (on top), then the query
    (kw || []).forEach(i => plot(i, ACCENT2, 3, false));
    (vec || []).forEach(i => plot(i, ACCENT, 3.2, false));
    plot(query, INK, 5, true);

    // legend
    ctx.font = "11px " + SANS; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillStyle = ACCENT2; ctx.fillText("● keyword hits (scattered)", pad, 8);
    ctx.fillStyle = ACCENT; ctx.fillText("● vector hits (clustered)", pad + 170, 8);

    canvas.style.cursor = "pointer";
    canvas.onclick = e => {
      const r = canvas.getBoundingClientRect(), mx2 = e.clientX - r.left, my2 = e.clientY - r.top;
      const h = hits.find(h => mx2 >= h.x0 && mx2 <= h.x1 && my2 >= h.y0 && my2 <= h.y1);
      if (h && onWord) onWord(h.word);
    };
  }

  /* Leaf concept graph: query at center, semantic neighbors branching out; click a leaf to grow
   * the next hop. Force-directed (adapted from where-the-lines-are's drawCooccurrenceNetwork).
   * Positions persist across renders (in _pos) so expansion looks like growth, not a reshuffle. */
  let _pos = new Map();                 // word -> {x,y,vx,vy}
  function resetGraphLayout() { _pos = new Map(); }

  function drawConceptGraph(canvas, graph, onExpand) {
    const cssW = canvas.parentElement.clientWidth || 720, cssH = 460, pad = 38;
    const ctx = fit(canvas, cssW, cssH);
    const nodes = graph.nodes, edges = graph.edges, cx = cssW / 2, cy = cssH / 2;
    nodes.forEach((nd, i) => {
      let p = _pos.get(nd.word);
      if (!p) { const a = i * 2.399; p = { x: cx + Math.cos(a) * 70, y: cy + Math.sin(a) * 70, vx: 0, vy: 0 }; if (i === 0) { p.x = cx; p.y = cy; } _pos.set(nd.word, p); }
      nd._p = p;
    });
    const maxW = Math.max(1e-6, ...edges.map(e => e.w));

    for (let it = 0; it < 150; it++) {
      const alpha = 1 - it / 150;
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i]._p, b = nodes[j]._p;
        let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy || 1, d = Math.sqrt(d2);
        const f = 900 * alpha / d2, fx = dx / d * f, fy = dy / d * f;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
      for (const e of edges) {
        const a = _pos.get(e.a), b = _pos.get(e.b);
        let dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = 0.06 * alpha * (e.w / maxW) * d, fx = dx / d * f, fy = dy / d * f;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
      nodes.forEach((nd, i) => {
        const p = nd._p;
        if (i === 0) { p.x += (cx - p.x) * 0.2; p.y += (cy - p.y) * 0.2; p.vx = p.vy = 0; return; }  // pin query near center
        p.vx += (cx - p.x) * 0.01 * alpha; p.vy += (cy - p.y) * 0.01 * alpha;
        p.vx *= 0.8; p.vy *= 0.8;
        p.x += Math.max(-15, Math.min(15, p.vx)); p.y += Math.max(-15, Math.min(15, p.vy));
        p.x = Math.max(pad, Math.min(cssW - pad, p.x)); p.y = Math.max(pad, Math.min(cssH - pad, p.y));
      });
    }

    ctx.clearRect(0, 0, cssW, cssH);
    for (const e of edges) {
      const a = _pos.get(e.a), b = _pos.get(e.b), g = Math.round(225 - 170 * (e.w / maxW));
      ctx.strokeStyle = `rgb(${g},${g},${g})`; ctx.lineWidth = 0.5 + 2 * (e.w / maxW);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    const hits = [];
    nodes.forEach((nd, i) => {
      const p = nd._p, isQ = i === 0, exp = graph.expanded.has(nd.word);
      ctx.fillStyle = isQ ? INK : (exp ? ACCENT : ACCENT2);
      ctx.beginPath(); ctx.arc(p.x, p.y, isQ ? 6 : 4, 0, 7); ctx.fill();
      ctx.font = (isQ ? "bold 13px " : "12px ") + SERIF; ctx.fillStyle = isQ ? INK : "#333";
      ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(nd.word, p.x, p.y - 6);
      const w = ctx.measureText(nd.word).width;
      hits.push({ x0: p.x - w / 2 - 3, x1: p.x + w / 2 + 3, y0: p.y - 20, y1: p.y + 8, word: nd.word, exp });
    });

    canvas.style.cursor = "pointer";
    canvas.onclick = e => {
      const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const h = hits.find(h => mx >= h.x0 && mx <= h.x1 && my >= h.y0 && my <= h.y1);
      if (h && onExpand) onExpand(h.word);
    };
  }

  return { drawMeaningMap, drawConceptGraph, resetGraphLayout, fit,
           _palette: { ACCENT, ACCENT2, INK, MUTE, FAINT, SERIF, SANS } };
})();
