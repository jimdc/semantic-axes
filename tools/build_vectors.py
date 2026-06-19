#!/usr/bin/env python3
"""
Phase 0 precompute pipeline for semantic-axes.

Loads a static word-embedding file (GloVe text format: `word f1 f2 ... fN`), curates a
small "common English" vocabulary, normalizes it for clean axis geometry, int8-quantizes
it for the browser, and bakes the curated axis bank (tools/axis_bank.json) into unit
direction vectors. Emits the four data files the front-end consumes:

  data/vectors.bin   int8, (count x dim) row-major     -- the explorable word vectors
  data/vocab.json    ["the","king",...]                -- words, in row order of vectors.bin
  data/axes.json     {axes:[{id,label,vector:[...]}]}  -- the named-direction bank
  data/meta.json     {model,dim,count,scale,...}        -- how to dequantize + provenance

Math:
  mean-center over the curated vocab -> L2-normalize -> (optional all-but-the-top) -> int8.
  axis = unit(mean over pairs of (v(pos) - v(neg)))   [Bolukbasi 2016 / SemAxis]
  score(word, axis) = dot(centered_normalized_word, axis)   ~[-1,1], + = toward `pos` pole

Usage:
  tools/build_vectors.py --glove data/glove.6B.300d.txt --vocab 12000 --pool 120000
"""
import argparse, json, re, struct, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"^[a-z]+$")  # curated explorable vocab: pure lowercase letters


def load_glove(path, pool):
    """Read the first `pool` (most-frequent) rows -> {word: float32 vec}. GloVe is freq-ordered."""
    words, vecs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= pool:
                break
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    dim = len(vecs[0])
    return words, np.vstack(vecs), dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", default=str(ROOT / "data/glove.6B.300d.txt"))
    ap.add_argument("--vocab", type=int, default=200000, help="max explorable vocab size (clean tokens)")
    ap.add_argument("--pool", type=int, default=200000, help="how many freq-ordered rows to read (anchor pool)")
    ap.add_argument("--abtt", type=int, default=0, help="all-but-the-top: # of top PCs to remove (0=off)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.glove} (first {args.pool} rows) ...", flush=True)
    pool_words, pool_vecs, dim = load_glove(args.glove, args.pool)
    pool_idx = {w: i for i, w in enumerate(pool_words)}
    print(f"  {len(pool_words)} rows, dim={dim}  ({time.time()-t0:.0f}s)", flush=True)

    # --- curate the explorable vocab: first N clean tokens, in frequency order ---
    sub_words = [w for w in pool_words if WORD_RE.match(w) and len(w) >= 2][: args.vocab]
    sub_idx = {w: i for i, w in enumerate(sub_words)}
    X = pool_vecs[[pool_idx[w] for w in sub_words]].astype(np.float32)  # (V, dim) raw
    V = len(sub_words)
    print(f"curated vocab: {V} words", flush=True)

    # --- preprocessing: center -> (optional all-but-the-top) -> L2-normalize ---
    mean = X.mean(0)
    Xc = X - mean
    if args.abtt > 0:
        # Mu & Viswanath 2018: remove top PCs (frequency/anisotropy directions)
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        for k in range(min(args.abtt, vt.shape[0])):
            comp = vt[k]
            Xc = Xc - np.outer(Xc @ comp, comp)
        print(f"  all-but-the-top: removed {args.abtt} component(s)", flush=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)

    # --- int8 quantization (global scale) ---
    scale = float(np.abs(Xn).max() / 127.0)
    Q = np.clip(np.round(Xn / scale), -127, 127).astype(np.int8)
    rt_err = float(np.abs(Q.astype(np.float32) * scale - Xn).max())
    print(f"  int8 quantize: scale={scale:.6g}  max round-trip err={rt_err:.4g}", flush=True)

    # --- build the axis bank from raw anchor vectors ---
    bank = json.load(open(ROOT / "tools/axis_bank.json"))["axes"]
    axes_out, A = [], []
    for ax in bank:
        diffs, used, total = [], 0, len(ax["pairs"])
        for pos, neg in ax["pairs"]:
            if pos in pool_idx and neg in pool_idx:
                diffs.append(pool_vecs[pool_idx[pos]] - pool_vecs[pool_idx[neg]])
                used += 1
        if not diffs:
            print(f"  !! axis '{ax['id']}': no usable anchor pairs, skipping", flush=True)
            continue
        d = np.mean(diffs, axis=0)
        v = d / (np.linalg.norm(d) + 1e-9)
        A.append(v)
        axes_out.append(dict(id=ax["id"], label=ax["label"], pos=ax["pos"], neg=ax["neg"],
                             poles=[ax["pos"], ax["neg"]], pairs_used=used, pairs_total=total,
                             vector=[round(float(x), 6) for x in v]))
        if used < total:
            print(f"  ~ axis '{ax['id']}': used {used}/{total} pairs", flush=True)
    A = np.vstack(A)  # (nAxes, dim)
    print(f"axis bank: {len(axes_out)} axes built", flush=True)

    # --- write outputs ---
    (ROOT / "data").mkdir(exist_ok=True)
    Q.tofile(ROOT / "data/vectors.bin")
    json.dump(sub_words, open(ROOT / "data/vocab.json", "w"))
    json.dump(dict(axes=axes_out), open(ROOT / "data/axes.json", "w"), indent=1)
    json.dump(dict(model=Path(args.glove).name, dim=dim, count=V, scale=scale,
                   quant="int8", preprocessing=dict(center=True, l2=True, abtt=args.abtt),
                   n_axes=len(axes_out), source="GloVe (Wikipedia+Gigaword, 6B, 300d)",
                   note="Axes encode the embedding's LEARNED ASSOCIATIONS, not ground truth.",
                   built=time.strftime("%Y-%m-%d")),
              open(ROOT / "data/meta.json", "w"), indent=1)
    sz = (ROOT / "data/vectors.bin").stat().st_size
    print(f"wrote data/ (vectors.bin = {sz/1e6:.1f} MB)  ({time.time()-t0:.0f}s)", flush=True)

    # ================= sanity checks =================
    def scores(word):
        return Xn[sub_idx[word]] @ A.T if word in sub_idx else None

    def top_axes(word, k=6):
        s = scores(word)
        if s is None:
            return f"'{word}' not in vocab"
        order = np.argsort(-np.abs(s))[:k]
        return "  ".join(f"{axes_out[i]['id']}{'+' if s[i]>=0 else '-'}{abs(s[i]):.2f}" for i in order)

    def neighbors(word, k=8):
        if word not in sub_idx:
            return f"'{word}' not in vocab"
        sim = Xn @ Xn[sub_idx[word]]
        order = np.argsort(-sim)[: k + 1]
        return ", ".join(sub_words[i] for i in order if sub_words[i] != word)[:200]

    def analogy(a, b, c, k=5):  # a is to b as c is to ?  (a - b + c)
        if not all(w in pool_idx for w in (a, b, c)):
            return "missing anchor"
        target = pool_vecs[pool_idx[a]] - pool_vecs[pool_idx[b]] + pool_vecs[pool_idx[c]]
        target = target / (np.linalg.norm(target) + 1e-9)
        raw_n = pool_vecs / (np.linalg.norm(pool_vecs, axis=1, keepdims=True) + 1e-9)
        sim = raw_n @ target
        order = np.argsort(-sim)[:k + 3]
        return ", ".join(pool_words[i] for i in order if pool_words[i] not in (a, b, c))[:120]

    print("\n=== SANITY CHECKS ===")
    for w in ["king", "queen", "mother", "scientist", "ocean", "money"]:
        print(f"  axes[{w:9s}] = {top_axes(w)}")
    print(f"\n  king - man + woman -> {analogy('king','man','woman')}")
    print(f"  paris - france + italy -> {analogy('paris','france','italy')}")
    print(f"\n  neighbors[mother] = {neighbors('mother')}")
    print(f"  neighbors[king]   = {neighbors('king')}")
    print(f"\nbuilt in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
