// Exercises the actual geometry baked into the shipped vectors — replicating the browser's int8
// math — so a broken rebuild (wrong centering, scale, anchors) is caught.
const { test } = require("node:test");
const assert = require("node:assert");
const { load } = require("./_load");

const D = load();
const { dim, q, scale } = D;

// dequantized float row
function vec(i) { const a = new Float64Array(dim), o = i * dim; for (let d = 0; d < dim; d++) a[d] = q[o + d] * scale; return a; }
// word's position on an axis (centered dot product) — same as Embed.dotRowFloat
function onAxis(i, v) { let s = 0, o = i * dim; for (let d = 0; d < dim; d++) s += q[o + d] * v[d]; return s * scale; }

test("king's most salient axis is royalty (+, strong)", () => {
  const i = D.index.get("king");
  const ranked = D.axes.map(a => ({ id: a.id, s: onAxis(i, a.vector) }))
    .sort((x, y) => Math.abs(y.s) - Math.abs(x.s));
  assert.equal(ranked[0].id, "royalty", "top axis");
  assert.ok(ranked[0].s > 0.5, `royalty score should be strong, got ${ranked[0].s.toFixed(3)}`);
});

test("gender axis orders man (+) vs woman (-)", () => {
  const g = D.axes.find(a => a.id === "gender").vector;
  assert.ok(onAxis(D.index.get("man"), g) > 0, "man on + (masculine) side");
  assert.ok(onAxis(D.index.get("woman"), g) < 0, "woman on - (feminine) side");
});

test("analogy: king - man + woman ≈ queen", () => {
  const k = vec(D.index.get("king")), m = vec(D.index.get("man")), w = vec(D.index.get("woman"));
  const t = new Float64Array(dim); let tn = 0;
  for (let d = 0; d < dim; d++) { t[d] = k[d] - m[d] + w[d]; tn += t[d] * t[d]; }
  tn = Math.sqrt(tn) || 1; for (let d = 0; d < dim; d++) t[d] /= tn;
  const exclude = new Set(["king", "man", "woman"]);
  let best = "", bestS = -2;
  for (let i = 0; i < D.count; i++) {
    if (exclude.has(D.words[i])) continue;
    let s = 0, o = i * dim; for (let d = 0; d < dim; d++) s += q[o + d] * t[d];
    if (s > bestS) { bestS = s; best = D.words[i]; }
  }
  assert.equal(best, "queen", "nearest to king-man+woman");
});
