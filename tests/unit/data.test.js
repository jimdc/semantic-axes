const { test } = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const path = require("path");
const { load, ROOT } = require("./_load");

const D = load();

test("meta, vocab and vectors.bin are mutually consistent", () => {
  assert.equal(D.words.length, D.count, "vocab length == meta.count");
  assert.equal(D.q.length, D.count * D.dim, "vectors.bin size == count * dim");
  assert.ok(D.scale > 0, "scale is positive");
});

test("every bank axis is a unit vector of the right length, with two poles", () => {
  for (const a of D.axes) {
    assert.equal(a.vector.length, D.dim, `${a.id} vector length`);
    let n = 0; for (const x of a.vector) n += x * x;
    assert.ok(Math.abs(Math.sqrt(n) - 1) < 1e-3, `${a.id} should be ~unit, got ${Math.sqrt(n).toFixed(4)}`);
    assert.equal(a.poles.length, 2, `${a.id} has two poles`);
  }
});

test("sae_features.json (if built) is well-formed", () => {
  const p = path.join(ROOT, "data/sae_features.json");
  if (!fs.existsSync(p)) return;                       // optional artifact
  const d = JSON.parse(fs.readFileSync(p));
  assert.ok(d.count >= 1 && d.words, "has words");
  const sample = Object.values(d.words)[0];
  assert.ok(Array.isArray(sample) && sample.length, "a word maps to a non-empty feature list");
  assert.ok(sample[0].id && typeof sample[0].label === "string", "feature has id + label");
});
