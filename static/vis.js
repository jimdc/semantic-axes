/* vis.js — canvas visualizations. Pure draw functions: (canvas, data, callbacks) in, pixels out.
 * Tufte-minimal: grayscale + one accent, direct labels, every mark carries data. */
const Vis = (() => {
  const ACCENT = "#b3541e";      // toward the POSITIVE pole
  const ACCENT2 = "#27506b";     // toward the NEGATIVE pole
  const INK = "#1a1a1a", MUTE = "#8a8a8a", FAINT = "#d8d8d8";
  const SERIF = '"Palatino Linotype", Palatino, Georgia, serif';
  const SANS = 'system-ui, -apple-system, sans-serif';
  const SCALE = 0.62;            // |score| that fills the half-bar (typical max projection)

  function fit(canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    return ctx;
  }

  /* Diverging horizontal bars: a word's position on each of its top axes.
   * axes = [{id,label,score,poles:[pos,neg]}], onPick(axisId). */
  function drawAxisBars(canvas, axes, onPick, activeId) {
    const rowH = 30, leftW = 116, rightW = 44, padX = 12;
    const cssW = canvas.parentElement.clientWidth || 720;
    const cssH = Math.max(rowH, axes.length * rowH) + 8;
    const ctx = fit(canvas, cssW, cssH);
    const plotL = leftW, plotR = cssW - rightW, plotW = plotR - plotL, cx = plotL + plotW / 2;

    ctx.clearRect(0, 0, cssW, cssH);
    // zero line
    ctx.strokeStyle = FAINT; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, 4); ctx.lineTo(cx, cssH - 4); ctx.stroke();

    axes.forEach((a, i) => {
      const y = i * rowH + 6, midY = y + rowH / 2 - 3;
      if (a.id === activeId) { ctx.fillStyle = "#f4efe9"; ctx.fillRect(0, y - 2, cssW, rowH); }

      // axis label (left, right-aligned)
      ctx.fillStyle = INK; ctx.font = "13px " + SANS; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(a.label.split(" ↔ ")[0] === a.poles[0] ? a.id : a.id, leftW - 10, midY);

      // pole words at the bar ends (faint)
      ctx.font = "10px " + SANS; ctx.fillStyle = MUTE;
      ctx.textAlign = "left";  ctx.fillText(a.poles[1], plotL + 1, midY);      // neg pole (left)
      ctx.textAlign = "right"; ctx.fillText(a.poles[0], plotR - 1, midY);      // pos pole (right)

      // diverging bar
      const frac = Math.max(-1, Math.min(1, a.score / SCALE));
      const x2 = cx + frac * (plotW / 2 - 30);
      ctx.fillStyle = a.score >= 0 ? ACCENT : ACCENT2;
      const bx = Math.min(cx, x2), bw = Math.abs(x2 - cx);
      ctx.fillRect(bx, midY - 5, Math.max(bw, 1), 10);

      // signed score (right)
      ctx.fillStyle = INK; ctx.font = "11px " + SANS; ctx.textAlign = "left";
      ctx.fillText((a.score >= 0 ? "+" : "−") + Math.abs(a.score).toFixed(2), plotR + 6, midY);
    });

    canvas.style.cursor = "pointer";
    canvas.onclick = e => {
      const r = canvas.getBoundingClientRect();
      const i = Math.floor((e.clientY - r.top - 6) / rowH);
      if (i >= 0 && i < axes.length && onPick) onPick(axes[i].id);
    };
    return cssH;
  }

  /* Spectrum: a word's neighborhood laid out along ONE axis (a number line).
   * items = [{word,score,isSelf}] sorted ascending by score; onWord(word). */
  function drawSpectrum(canvas, items, axis, onWord) {
    const cssW = canvas.parentElement.clientWidth || 720, cssH = 132, padX = 56, lanes = 4, laneH = 17;
    const ctx = fit(canvas, cssW, cssH);
    ctx.clearRect(0, 0, cssW, cssH);
    if (!items.length) return;
    const baseY = 34;
    const lo = Math.min(...items.map(d => d.score)), hi = Math.max(...items.map(d => d.score));
    const span = (hi - lo) || 1;
    const X = s => padX + ((s - lo) / span) * (cssW - 2 * padX);

    // baseline + pole labels
    ctx.strokeStyle = FAINT; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padX, baseY); ctx.lineTo(cssW - padX, baseY); ctx.stroke();
    ctx.font = "10px " + SANS; ctx.fillStyle = MUTE; ctx.textBaseline = "middle";
    ctx.textAlign = "right"; ctx.fillText("← " + axis.poles[1], padX - 4, baseY);
    ctx.textAlign = "left";  ctx.fillText(axis.poles[0] + " →", cssW - padX + 4, baseY);

    canvas._hit = [];
    items.forEach((d, i) => {
      const x = X(d.score), lane = i % lanes, y = baseY + 14 + lane * laneH;
      ctx.strokeStyle = d.isSelf ? ACCENT : FAINT;
      ctx.beginPath(); ctx.moveTo(x, baseY); ctx.lineTo(x, y - 8); ctx.stroke();
      ctx.font = (d.isSelf ? "bold 12px " : "12px ") + SERIF;
      ctx.fillStyle = d.isSelf ? ACCENT : INK; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(d.word, x, y);
      const w = ctx.measureText(d.word).width;
      canvas._hit.push({ x0: x - w / 2 - 3, x1: x + w / 2 + 3, y0: y - 8, y1: y + 8, word: d.word });
    });

    canvas.style.cursor = "pointer";
    canvas.onclick = e => {
      const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const hit = (canvas._hit || []).find(h => mx >= h.x0 && mx <= h.x1 && my >= h.y0 && my <= h.y1);
      if (hit && onWord) onWord(hit.word);
    };
  }

  /* Scatter: a word's neighborhood on two named axes. pts=[{word,x,y,isSelf}]; onWord(word). */
  function drawScatter(canvas, pts, axX, axY, onWord) {
    const cssW = canvas.parentElement.clientWidth || 720, cssH = 380, pad = 40;
    const ctx = fit(canvas, cssW, cssH);
    ctx.clearRect(0, 0, cssW, cssH);
    if (!pts.length) return;
    const ext = arr => { const m = Math.max(0.1, ...arr.map(Math.abs)); return [-m, m]; };
    const [xlo, xhi] = ext(pts.map(p => p.x)), [ylo, yhi] = ext(pts.map(p => p.y));
    const X = v => pad + ((v - xlo) / (xhi - xlo)) * (cssW - 2 * pad);
    const Y = v => (cssH - pad) - ((v - ylo) / (yhi - ylo)) * (cssH - 2 * pad);

    ctx.strokeStyle = FAINT; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(0), pad); ctx.lineTo(X(0), cssH - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(cssW - pad, Y(0)); ctx.stroke();
    ctx.font = "10px " + SANS; ctx.fillStyle = MUTE;
    ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText(axX.poles[0] + " →", cssW - pad, Y(0) + 4);
    ctx.textAlign = "left"; ctx.fillText("← " + axX.poles[1], pad, Y(0) + 4);
    ctx.textBaseline = "bottom"; ctx.fillText("↑ " + axY.poles[0], X(0) + 4, pad + 10);
    ctx.fillText("↓ " + axY.poles[1], X(0) + 4, cssH - pad);

    canvas._hit = [];
    pts.forEach(p => {
      const x = X(p.x), y = Y(p.y);
      ctx.fillStyle = p.isSelf ? ACCENT : INK;
      ctx.beginPath(); ctx.arc(x, y, p.isSelf ? 4 : 2.4, 0, 7); ctx.fill();
      ctx.font = (p.isSelf ? "bold 12px " : "11px ") + SERIF; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillStyle = p.isSelf ? ACCENT : "#444"; ctx.fillText(" " + p.word, x + 2, y);
      const w = ctx.measureText(p.word).width;
      canvas._hit.push({ x0: x - 4, x1: x + w + 8, y0: y - 7, y1: y + 7, word: p.word });
    });
    canvas.style.cursor = "pointer";
    canvas.onclick = e => {
      const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const hit = (canvas._hit || []).find(h => mx >= h.x0 && mx <= h.x1 && my >= h.y0 && my <= h.y1);
      if (hit && onWord) onWord(hit.word);
    };
  }

  return { drawAxisBars, drawSpectrum, drawScatter };
})();
