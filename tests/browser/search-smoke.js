// Smoke for the "why vector search?" page: contrast columns, recall gauge, meaning-map, leaf graph.
const { chromium } = require("@playwright/test");
const PORT = process.env.PORT || 8777;

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 1900 }, deviceScaleFactor: 2 });
  const errors = [];
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", e => errors.push(String(e)));

  await page.goto(`http://localhost:${PORT}/search.html#q=king`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.getElementById("stageGraph").style.display !== "none", { timeout: 15000 });

  // Stage 1 — the contrast: keyword finds look-alikes, vector finds meaning, near-disjoint
  const kw = await page.$$eval("#kwList .wlink", e => e.map(x => x.textContent));
  const vec = await page.$$eval("#vecList .wlink", e => e.map(x => x.textContent));
  if (!(kw.includes("kingdom") || kw.includes("kings"))) throw new Error("keyword col missing spelling cousins: " + kw.join(","));
  if (!vec.includes("queen")) throw new Error("vector col missing 'queen': " + vec.join(","));
  const overlap = kw.filter(w => vec.includes(w));
  if (overlap.length > 3) throw new Error("columns should be near-disjoint, overlap=" + overlap.join(","));
  const recall = await page.$eval("#recall", e => e.textContent);
  if (!/of\s+10/.test(recall)) throw new Error("recall gauge missing: " + recall);

  // Stage 2 — meaning-map rendered
  if (await page.$eval("#stageMap", s => s.style.display === "none")) throw new Error("meaning-map hidden");

  // Stage 3 — leaf graph grows when a leaf is expanded
  const before = await page.evaluate(() => window.__graphSize);
  await page.evaluate(() => window.__expand("queen"));
  const after = await page.evaluate(() => window.__graphSize);
  if (!(after > before)) throw new Error(`graph did not grow on expand (${before} -> ${after})`);

  await page.screenshot({ path: "tests/browser/search.png", fullPage: true });
  console.log("keyword:", kw.slice(0, 5).join(", "));
  console.log("vector: ", vec.slice(0, 5).join(", "));
  console.log("recall: ", recall.replace(/\s+/g, " ").trim());
  console.log("graph grew:", before, "->", after, "nodes");
  console.log("console errors:", errors.length ? errors : "none");
  await browser.close();
  if (errors.length) process.exit(2);
})().catch(e => { console.error("SEARCH SMOKE FAILED:", e.message); process.exit(1); });
