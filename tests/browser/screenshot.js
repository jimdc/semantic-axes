// Visual smoke for the full feature set: axis bars, neighbors, two-axis scatter, discovered axes.
// Loads "king", asserts each view renders, screenshots; also checks a non-word lands softly.
const { chromium } = require("@playwright/test");
const PORT = process.env.PORT || 8777;

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 1400 }, deviceScaleFactor: 2 });
  const errors = [];
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", e => errors.push(String(e)));

  await page.goto(`http://localhost:${PORT}/#w=king`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.getElementById("metaLine").textContent.includes("axes"));
  await page.waitForFunction(() => document.getElementById("discoverySection").style.display !== "none");

  const nbr = await page.$$eval("#neighbors .chip", e => e.map(x => x.textContent));
  if (!nbr.includes("queen")) throw new Error("expected 'queen' in neighbors, got: " + nbr.join(","));

  const scatterShown = await page.$eval("#scatterSection", s => s.style.display !== "none");
  const nOpts = await page.$$eval("#scatterXsel option", o => o.length);
  if (!scatterShown || nOpts < 30) throw new Error(`scatter not ready (shown=${scatterShown}, opts=${nOpts})`);

  const disc = await page.$$eval("#discovery .disc-row", rows => rows.map(r => r.querySelector(".disc-label").textContent.trim()));
  if (disc.length < 2) throw new Error("expected discovered axes, got: " + disc.join(" | "));

  // open the royalty spectrum too, for the screenshot
  await page.evaluate(() => location.hash = "#w=king&axis=royalty");
  await page.waitForFunction(() => document.getElementById("spectrumWrap").style.display !== "none");
  await page.screenshot({ path: "tests/browser/king.png", fullPage: true });

  // custom axis: build "art <-> science" on a fresh word, assert spectrum renders, screenshot
  await page.evaluate(() => location.hash = "#w=doctor&cx=science&cn=art");
  await page.waitForFunction(() => document.getElementById("customResult").style.display !== "none");
  const customShown = await page.$eval("#customSpectrum", c => c.style.display !== "none");
  const customTitle = await page.$eval("#customTitle", e => e.textContent);
  if (!customShown) throw new Error("custom-axis spectrum did not render: " + customTitle);
  await page.screenshot({ path: "tests/browser/custom.png", fullPage: true });

  // SAE substrate: toggle, assert the distinctive-feature bundle renders, then WANDER —
  // open a feature -> its receptive field of words -> click one -> a fresh bundle.
  await page.evaluate(() => location.hash = "#w=king");
  await page.waitForTimeout(200);
  await page.click('#backendToggle button[data-backend="sae"]');
  await page.waitForFunction(() => document.getElementById("featureList").style.display !== "none");
  const feats = await page.$$eval("#featureList .feat-label", els => els.map(e => e.textContent.trim()));
  if (feats.length < 3) throw new Error("SAE feature bundle did not render: " + feats.join(" | "));
  const npHref = await page.$eval("#featureList .feat-np", a => a.href);
  if (!npHref.includes("neuronpedia.org")) throw new Error("feature ↗ link not to neuronpedia: " + npHref);

  // open the top feature's receptive field, screenshot the bundle, then hop to a word in it
  await page.click("#featureList .feat-row:first-child .feat-label");
  await page.waitForFunction(() => {
    const w = document.querySelector("#featureList .feat-row:first-child .feat-words");
    return w && w.style.display !== "none" && w.querySelectorAll(".chip").length > 0;
  });
  const rf = await page.$$eval("#featureList .feat-row:first-child .feat-words .chip", c => c.map(x => x.textContent));
  await page.screenshot({ path: "tests/browser/sae.png", fullPage: true });
  await page.click("#featureList .feat-row:first-child .feat-words .chip");
  await page.waitForFunction(() => document.getElementById("queryInput").value !== "king");
  const hopped = await page.$eval("#queryInput", i => i.value);
  console.log("SAE features (king):", feats.slice(0, 3).join("  |  "));
  console.log("receptive field of top feature:", rf.slice(0, 8).join(", "), " → hopped to", hopped);

  // compare mode: the two-substrate confrontation renders both columns for the same word
  await page.evaluate(() => location.hash = "#w=king");
  await page.waitForTimeout(150);
  await page.click('#backendToggle button[data-backend="compare"]');
  await page.waitForFunction(() => document.getElementById("compareSection").style.display !== "none");
  await page.waitForFunction(() =>
    document.querySelectorAll("#cmpStaticRows .cmp-row").length > 0 &&
    document.querySelectorAll("#cmpSaeRows .cmp-row").length > 0);
  const cmpL = await page.$$eval("#cmpStaticRows .cmp-name", e => e.map(x => x.textContent.trim()));
  const cmpR = await page.$$eval("#cmpSaeRows .cmp-name", e => e.map(x => x.textContent.trim()));
  await page.screenshot({ path: "tests/browser/compare.png", fullPage: true });
  console.log("compare — ours:", cmpL.slice(0, 3).join(", "), " || model's:", cmpR.slice(0, 2).join(" / "));

  // back to static, then soft landing on a non-word
  await page.click('#backendToggle button[data-backend="static"]');
  await page.waitForFunction(() => document.getElementById("axisSection").style.display !== "none");
  await page.evaluate(() => location.hash = "#w=zzqx");
  await page.waitForFunction(() => document.getElementById("notice").style.display === "block");

  console.log("neighbors:", nbr.slice(0, 6).join(", "));
  console.log("scatter axes available:", nOpts);
  console.log("discovered axes:", disc.join("  |  "));
  console.log("custom axis:", customTitle.replace(/\s+/g, " ").trim());
  console.log("console errors:", errors.length ? errors : "none");
  await browser.close();
  if (errors.length) process.exit(2);
})().catch(e => { console.error("SMOKE FAILED:", e.message); process.exit(1); });
