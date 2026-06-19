// Load the shipped data artifacts in Node for testing (no browser needed).
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "../..");

function load() {
  const meta = JSON.parse(fs.readFileSync(path.join(ROOT, "data/meta.json")));
  const words = JSON.parse(fs.readFileSync(path.join(ROOT, "data/vocab.json")));
  const axes = JSON.parse(fs.readFileSync(path.join(ROOT, "data/axes.json"))).axes;
  const buf = fs.readFileSync(path.join(ROOT, "data/vectors.bin"));
  const q = new Int8Array(buf.buffer, buf.byteOffset, buf.length);
  const index = new Map(words.map((w, i) => [w, i]));
  return { ...meta, words, axes, q, index };
}

module.exports = { load, ROOT };
