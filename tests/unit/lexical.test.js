const { test } = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const path = require("path");
const Lexical = require("../../static/lexical.js");

test("trigrams pad pg_trgm-style", () => {
  const t = Lexical.trigrams("king");
  assert.deepEqual([...t].sort(), ["  k", " ki", "ing", "kin", "ng "].sort());
});

test("jaccard is 1 for identical, 0 for disjoint", () => {
  assert.equal(Lexical.jaccard(Lexical.trigrams("king"), Lexical.trigrams("king")), 1);
  assert.equal(Lexical.jaccard(Lexical.trigrams("abc"), Lexical.trigrams("xyz")), 0);
});

// the whole lesson: keyword search finds spelling look-alikes, NOT meaning matches
test("keyword search over the real vocab: shares-letters, not meaning", () => {
  const vocab = JSON.parse(fs.readFileSync(path.join(__dirname, "../../data/vocab.json"))).slice(0, 8000);
  const idx = Lexical.buildIndex(vocab);

  const kingTop = Lexical.lexicalNearest("king", idx, 12).map(r => r.word);
  assert.ok(kingTop.includes("kingdom") || kingTop.includes("kings"), "spelling cousins surface");
  const king50 = new Set(Lexical.lexicalNearest("king", idx, 50).map(r => r.word));
  assert.ok(!king50.has("queen"), "the nearest-in-meaning word is NOT keyword-reachable");

  // 'car' keyword hits are look-alikes (card/care/cart), never the synonym 'automobile'
  const car50 = new Set(Lexical.lexicalNearest("car", idx, 50).map(r => r.word));
  assert.ok(!car50.has("automobile"), "synonym with no shared letters is invisible to keyword search");
});
