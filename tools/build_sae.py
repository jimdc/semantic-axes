#!/usr/bin/env python3
"""
Phase 4 precompute for the SAE backend — the substrate swap.

For each word in a curated set, calls Neuronpedia's /api/search-all to get the TOP ACTIVATING
SAE features (with their human-written explanations) and writes them to data/sae_features.json.
The browser then reads that file and drives the SAME QueryResult interface as the static backend —
so the only thing that changed is what fills "axes" (curated geometric directions -> live SAE
features). Keeps the site fully static: the API key stays server-side here, never in the browser,
which also sidesteps CORS.

Auth: set NEURONPEDIA_API_KEY (env var, or a .secrets.env / .env file in the repo root or its
parent). Get a free key at https://neuronpedia.org.
Model/SAE: gpt2-small / res-jb by default (fast, verified). Override with SAE_MODEL / SAE_SOURCESET.

Usage:
  tools/build_sae.py                 # curated demo word set
  tools/build_sae.py word1 word2 …   # specific words
"""
import os, sys, json, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# key from env first, else a .secrets.env / .env in the repo root or its parent
SECRET_FILES = [ROOT / ".secrets.env", ROOT / ".env", ROOT.parent / ".secrets.env"]
MODEL = os.environ.get("SAE_MODEL", "gpt2-small")
SOURCESET = os.environ.get("SAE_SOURCESET", "res-jb")
TOPK = int(os.environ.get("SAE_TOPK", "8"))
# restrict to specific SAE layer(s); empty = all layers in the set. e.g. SAE_LAYERS=20-gemmascope-res-16k
LAYERS = [s for s in os.environ.get("SAE_LAYERS", "").split(",") if s.strip()]

# curated demo set (all in the GloVe vocab too, so the static/SAE toggle lines up)
WORDS = """
king queen prince princess monarch throne crown royal
man woman boy girl father mother son daughter brother sister husband wife uncle aunt grandmother
doctor nurse teacher scientist engineer lawyer artist soldier president officer
happy sad angry love hate fear joy hope anger
dog cat wolf lion tiger horse bird fish snake bear eagle
ocean sea river lake mountain forest desert island beach
money gold rich poor wealth bank
science art music religion war peace law
hug kiss smile cry laugh dance sing
red blue green black white gold
paris london rome tokyo berlin
car train plane ship bicycle
apple bread water wine coffee milk
fire water earth wind ice
strong weak fast slow big small hot cold young old
nurse mother king queen
""".split()


def load_key():
    k = os.environ.get("NEURONPEDIA_API_KEY")
    if k:
        return k
    for path in SECRET_FILES:
        if path.exists():
            for line in open(path):
                s = line.strip()
                if "NEURONPEDIA_API_KEY" in s and "=" in s:
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def search_all(word, key):
    body = json.dumps({"modelId": MODEL, "sourceSet": SOURCESET, "text": word,
                       "selectedLayers": LAYERS, "sortIndexes": [1], "ignoreBos": True,
                       "numResults": TOPK * 2}).encode()
    req = urllib.request.Request("https://www.neuronpedia.org/api/search-all", data=body,
                                 headers={"x-api-key": key, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def extract(d):
    feats = []
    for r in d.get("result", []):
        n = r.get("neuron") or {}
        exps = n.get("explanations") or []
        label = exps[0].get("description") if exps else None
        if not label:
            continue
        feats.append({"id": f"{r['layer']}/{r['index']}", "layer": r["layer"], "index": str(r["index"]),
                      "label": label, "act": round(float(r.get("maxValue", 0)), 2)})
        if len(feats) >= TOPK:
            break
    return feats


OUTPATH = ROOT / "data/sae_features.json"


def save(out):
    json.dump(dict(model=MODEL, sourceSet=SOURCESET, topk=TOPK, layers=LAYERS,
                   built=time.strftime("%Y-%m-%d"), count=len(out), words=out,
                   note="Top activating SAE features per word (Neuronpedia search-all)."),
              open(OUTPATH, "w"))


def pick_words(args):
    if args.words:
        return list(dict.fromkeys(args.words))
    if args.top or args.all:                 # the most-frequent words = the static vocab, in order
        vocab = json.load(open(ROOT / "data/vocab.json"))
        return vocab if args.all else vocab[: args.top]
    return list(dict.fromkeys(WORDS))         # curated demo set


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Precompute per-word SAE features into data/sae_features.json")
    ap.add_argument("words", nargs="*", help="specific words (default: a curated demo set)")
    ap.add_argument("--top", type=int, default=0, help="precompute the N most-frequent words from the static vocab")
    ap.add_argument("--all", action="store_true", help="precompute the ENTIRE static vocab (full static<->SAE symmetry; very slow)")
    ap.add_argument("--no-merge", action="store_true", help="start fresh instead of extending existing sae_features.json")
    args = ap.parse_args()

    key = load_key()
    if not key:
        sys.exit("No NEURONPEDIA_API_KEY (env var, or a .secrets.env / .env file)")
    words = pick_words(args)

    # merge/resume: keep already-fetched words (same model+set) so coverage grows incrementally
    existing = {}
    if not args.no_merge and OUTPATH.exists():
        prev = json.load(open(OUTPATH))
        if prev.get("model") == MODEL and prev.get("sourceSet") == SOURCESET:
            existing = prev.get("words", {})
    todo = [w for w in words if w not in existing]
    print(f"SAE precompute: {len(words)} target words on {MODEL}/{SOURCESET}"
          + (f" layers={LAYERS}" if LAYERS else "") + f" — {len(existing)} cached, {len(todo)} to fetch", flush=True)

    (ROOT / "data").mkdir(exist_ok=True)
    out, t0 = dict(existing), time.time()
    for i, w in enumerate(todo):
        try:
            feats = extract(search_all(w, key))
            if feats:
                out[w] = feats
            tag = feats[0]["label"][:46] if feats else "(no labeled features)"
            print(f"  [{i+1}/{len(todo)}] {w:14s} {len(feats)} feats · {tag}", flush=True)
        except urllib.error.HTTPError as e:
            print(f"  [{i+1}/{len(todo)}] {w:14s} HTTP {e.code}", flush=True)
            if e.code == 429:
                time.sleep(5)
        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] {w:14s} ERR {e}", flush=True)
        time.sleep(0.25)
        if (i + 1) % 100 == 0:               # checkpoint long runs so progress is crash-safe + usable
            save(out); print(f"   … checkpoint: {len(out)} words saved", flush=True)

    save(out)
    print(f"\nwrote {OUTPATH.name} — {len(out)} words total in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
