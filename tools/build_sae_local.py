#!/usr/bin/env python3
"""
Local (no-API) variant of build_sae.py — computes SAE features ON-DEVICE, removing the rate-limit
ceiling. Activations come from a local Gemma-2-2b + Gemma Scope SAE; only feature LABELS are fetched
from Neuronpedia, and only ONCE PER FEATURE (cached in data/feature_labels.json) rather than once per
word — so coverage stops costing one API call per word.

The local extractor is intentionally kept OUTSIDE this repo. Point GEMMA_SCOPE_LOCAL at the directory
that holds it, then run this like build_sae.py:

    export GEMMA_SCOPE_LOCAL=/path/to/your/extractor   # dir containing gemma_scope_local.py
    export NEURONPEDIA_API_KEY=sk-...                   # only for first-time label lookups
    tools/build_sae_local.py --top 5000                 # resumable; merges into sae_features.json

With GEMMA_SCOPE_LOCAL unset this script does nothing — use tools/build_sae.py for the portable
Neuronpedia API path. (This file references only the env-var name, never a path or how the extractor
is hosted, so the public repo reveals nothing about the local setup.)
"""
import os, sys, json, time, urllib.request, urllib.error

import build_sae as bs          # reuse ROOT, OUTPATH, load_key, pick_words (none carry a model id)

MODEL = "gemma-2-2b"
SOURCESET = "gemmascope-res-16k"
SOURCE = "20-gemmascope-res-16k"          # must match gemma_scope_local.SOURCE so indices line up
TOPK = int(os.environ.get("SAE_TOPK", "8"))
LABELS_PATH = bs.ROOT / "data/feature_labels.json"


def get_extractor():
    d = os.environ.get("GEMMA_SCOPE_LOCAL")
    if not d:
        sys.exit("GEMMA_SCOPE_LOCAL is unset — set it to your local Gemma Scope extractor directory, "
                 "or use tools/build_sae.py for the Neuronpedia API path.")
    sys.path.insert(0, d)
    try:
        import gemma_scope_local as gs
    except Exception as e:
        sys.exit(f"could not import gemma_scope_local from $GEMMA_SCOPE_LOCAL: {e}")
    if getattr(gs, "SOURCE", None) != SOURCE:
        sys.exit(f"extractor SOURCE {getattr(gs, 'SOURCE', None)} != {SOURCE}; indices would not match labels.")
    print("loading local extractor from $GEMMA_SCOPE_LOCAL …", flush=True)
    return gs.load()


def label_lookup(index, key, cache):
    """Rich English label for a feature, fetched from Neuronpedia ONCE per feature index and cached.
    (The bulk S3 export has only a terser explanation per feature; the per-feature API returns the
    descriptive one that matches the existing data — so we use the API and cache aggressively.)"""
    fid = f"{SOURCE}/{index}"
    if fid in cache:
        return cache[fid]
    label = ""
    if key:
        url = f"https://www.neuronpedia.org/api/feature/{MODEL}/{SOURCE}/{index}"
        for _ in range(3):
            try:
                req = urllib.request.Request(url, headers={"x-api-key": key})
                with urllib.request.urlopen(req, timeout=60) as r:
                    exps = (json.load(r).get("explanations") or [])
                label = (exps[0].get("description") if exps else "") or ""
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(5); continue          # rate-limited — back off and retry
                break
            except Exception:
                break
        time.sleep(0.15)
    cache[fid] = label
    return label


def save(out):
    json.dump(dict(model=MODEL, sourceSet=SOURCESET, topk=TOPK, layers=[SOURCE],
                   built=time.strftime("%Y-%m-%d"), count=len(out), words=out,
                   note="Top activating SAE features per word (local Gemma Scope; labels via Neuronpedia per-feature)."),
              open(bs.OUTPATH, "w"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Local (no-API) SAE feature precompute into data/sae_features.json")
    ap.add_argument("words", nargs="*", help="specific words (default: the curated demo set)")
    ap.add_argument("--top", type=int, default=0, help="precompute the N most-frequent words from the static vocab")
    ap.add_argument("--all", action="store_true", help="precompute the ENTIRE static vocab (slow)")
    ap.add_argument("--no-merge", action="store_true", help="start fresh instead of extending sae_features.json")
    args = ap.parse_args()

    words = bs.pick_words(args)
    key = bs.load_key()                    # only used to fill NEW feature labels
    labels = json.load(open(LABELS_PATH)) if LABELS_PATH.exists() else {}

    existing = {}
    if not args.no_merge and bs.OUTPATH.exists():
        prev = json.load(open(bs.OUTPATH))
        if prev.get("model") == MODEL and prev.get("sourceSet") == SOURCESET:
            existing = prev.get("words", {})
    todo = [w for w in words if w not in existing]
    print(f"local SAE precompute: {len(words)} target words on {MODEL}/{SOURCE} — "
          f"{len(existing)} cached, {len(todo)} to compute", flush=True)

    ex = get_extractor()
    out, t0 = dict(existing), time.time()
    BATCH = int(os.environ.get("SAE_BATCH", "64"))     # activations are batched (one forward pass)
    done = 0
    for s in range(0, len(todo), BATCH):
        chunk = todo[s:s + BATCH]
        try:
            batch_feats = ex.features_batch(chunk, topk=TOPK)
        except Exception as e:
            print(f"  batch @{s}: ERR {e}", flush=True); continue
        for w, fl in zip(chunk, batch_feats):
            feats = [{"id": f"{SOURCE}/{idx}", "layer": SOURCE, "index": str(idx),
                      "label": label_lookup(idx, key, labels), "act": act} for idx, act in fl]
            labeled = [f for f in feats if f["label"]]
            feats = labeled or feats           # prefer labeled features, but never drop a word entirely
            if feats:
                out[w] = feats
            done += 1
        rate = done / max(1e-6, time.time() - t0)
        eta = (len(todo) - done) / max(1e-6, rate)
        print(f"  [{done}/{len(todo)}] …{chunk[-1]:14s} {len(out)} words, {len(labels)} labels "
              f"· {rate:.1f} w/s · ETA {eta/60:.0f}m", flush=True)
        if s // BATCH % 8 == 0:                # checkpoint ~every 8 batches (resumable, crash-safe)
            save(out); json.dump(labels, open(LABELS_PATH, "w"))

    save(out)
    json.dump(labels, open(LABELS_PATH, "w"))
    print(f"\nwrote {bs.OUTPATH.name} — {len(out)} words; {len(labels)} feature labels cached "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
