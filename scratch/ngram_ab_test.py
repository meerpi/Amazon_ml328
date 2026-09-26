"""N-Gram A/B Test: Compare two TF-IDF configurations on the SAME 5,000 query sample.

Config A: ngram_range=(3,4), min_df=5, max_df=0.05  [full-scale production config]
Config B: ngram_range=(3,5), min_df=2, lower_bound=0.1  [original tfidf_blocker.py config]

Reports: pair_completeness, macro_precision at k=30, vocab size, fit/transform time.
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tfidf_blocker import romanize_text, compute_macro_f05

SAMPLE_SIZE = 5000

print("=" * 95)
print("N-GRAM A/B TEST: Config A (3,4) vs Config B (3,5) on 5,000 queries")
print("=" * 95)

# 1. Load 5,000 S1 queries
print(f"\n[Step 1] Loading {SAMPLE_SIZE:,} real S1 queries...")
s1_ids: List[str] = []
query_texts: List[str] = []
with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip("\n").split("\t")
        eid = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        raw_txt = ((name or "") + " " + (addr or "")).lower()
        txt = raw_txt if raw_txt.isascii() else romanize_text(raw_txt)
        s1_ids.append(eid)
        query_texts.append(txt)
        if len(s1_ids) >= SAMPLE_SIZE:
            break

s1_id_set = set(s1_ids)

# 2. Load ground truth
print("[Step 2] Loading ground truth...")
gt_map: Dict[str, Set[str]] = {}
needed_cand_ids = set()
singletons = 0
with open("student_resource/dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip("\n").split("\t")
        s1 = parts[0]
        if s1 in s1_id_set:
            raw = parts[1] if len(parts) > 1 else ""
            if raw:
                cands = set(c.strip() for c in raw.split(",") if c.strip())
                gt_map[s1] = cands
                needed_cand_ids.update(cands)
            else:
                gt_map[s1] = set()
                singletons += 1
        if len(gt_map) >= len(s1_ids):
            break

for s1 in s1_ids:
    if s1 not in gt_map:
        gt_map[s1] = set()
        singletons += 1

total_true = sum(len(c) for c in gt_map.values())
print(f"  Ground truth: {len(gt_map):,} entities, {total_true:,} true links, {singletons:,} singletons")

# 3. Assemble candidate pool
print("[Step 3] Assembling candidate pool (true candidates + 150,000 distractors)...")
cand_ids: List[str] = []
cand_texts: List[str] = []
EXTRA = 150000
distractors = 0
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    fpath = f"student_resource/dataset/train/{fname}"
    with open(fpath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\n").split("\t")
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            raw_txt = ((name or "") + " " + (addr or "")).lower()
            txt = raw_txt if raw_txt.isascii() else romanize_text(raw_txt)
            if eid in needed_cand_ids:
                cand_ids.append(eid)
                cand_texts.append(txt)
            elif distractors < EXTRA:
                cand_ids.append(eid)
                cand_texts.append(txt)
                distractors += 1

c_id_arr = np.array(cand_ids)
print(f"  Pool: {len(cand_ids):,} candidates")

# 4. Run both configs
configs = [
    {
        "label": "Config A (3,4) min_df=5 max_df=0.05",
        "vectorizer_kwargs": dict(
            analyzer="char_wb", ngram_range=(3, 4), min_df=5, max_df=0.05,
            sublinear_tf=True, dtype=np.float32, norm="l2",
        ),
        "lower_bound": 0.08,  # production candidate_threshold
    },
    {
        "label": "Config B (3,5) min_df=2, lower_bound=0.1",
        "vectorizer_kwargs": dict(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, dtype=np.float32, norm="l2",
        ),
        "lower_bound": 0.1,  # original tfidf_blocker config
    },
]

print("\n" + "=" * 95)
for cfg in configs:
    print(f"\n--- {cfg['label']} ---")

    # Fit
    t_fit = time.time()
    vec = TfidfVectorizer(**cfg["vectorizer_kwargs"])
    vec.fit(cand_texts[:150000])
    fit_time = time.time() - t_fit
    vocab_size = len(vec.vocabulary_)
    print(f"  Fit time: {fit_time:.2f}s | Vocab size: {vocab_size:,}")

    # Transform
    t_xform = time.time()
    mat_cand = vec.transform(cand_texts)
    mat_q = vec.transform(query_texts)
    xform_time = time.time() - t_xform
    print(f"  Transform time: {xform_time:.2f}s | Cand matrix: {mat_cand.shape}, Query matrix: {mat_q.shape}")

    # Search
    t_search = time.time()
    mat_cand_T = mat_cand.T  # CSR->CSC zero-copy
    sim_res = sp_matmul_topn(
        mat_q, mat_cand_T, top_n=30, threshold=cfg["lower_bound"],
        n_threads=8, sort=True,
    )
    search_time = time.time() - t_search
    print(f"  Search time: {search_time:.2f}s ({len(s1_ids)/search_time:.1f} q/s)")

    # Extract predictions
    pred_by_s1 = defaultdict(list)
    indptr = sim_res.indptr
    indices = sim_res.indices
    data = sim_res.data
    for i, qid in enumerate(s1_ids):
        r_start, r_end = indptr[i], indptr[i + 1]
        if r_start < r_end:
            row_cids = c_id_arr[indices[r_start:r_end]]
            row_scores = data[r_start:r_end]
            for cid, sc in zip(row_cids, row_scores):
                pred_by_s1[qid].append((cid, float(sc)))

    # Compute metrics at k=30
    m = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=30)
    print(f"  pair_completeness: {m['pair_completeness']*100:.2f}%")
    print(f"  macro_precision:   {m['macro_precision']*100:.2f}%")
    print(f"  macro_recall:      {m['macro_recall']*100:.2f}%")
    print(f"  macro_f05:         {m['macro_f05']:.4f}")

    cfg["results"] = {
        "vocab_size": vocab_size,
        "fit_time": fit_time,
        "xform_time": xform_time,
        "search_time": search_time,
        **m,
    }

# Summary table
print("\n" + "=" * 95)
print("SUMMARY COMPARISON")
print("=" * 95)
print(f"{'Metric':<25} | {'Config A (3,4)':<20} | {'Config B (3,5)':<20} | {'Delta':<15}")
print("-" * 85)
a = configs[0]["results"]
b = configs[1]["results"]
for metric, fmt in [
    ("vocab_size", ",d"),
    ("fit_time", ".2f"),
    ("xform_time", ".2f"),
    ("search_time", ".2f"),
    ("pair_completeness", ".4f"),
    ("macro_precision", ".4f"),
    ("macro_recall", ".4f"),
    ("macro_f05", ".4f"),
]:
    va = a[metric]
    vb = b[metric]
    if isinstance(va, int):
        delta = vb - va
        print(f"{metric:<25} | {va:<20,} | {vb:<20,} | {delta:+,d}")
    else:
        delta = vb - va
        print(f"{metric:<25} | {va:<20{fmt}} | {vb:<20{fmt}} | {delta:+{fmt}}")
print("=" * 95)
print("\nRECOMMENDATION:")
if a["pair_completeness"] >= b["pair_completeness"] - 0.005:
    print("  Config A (3,4) retains recall within 0.5pp while having a smaller vocab")
    print("  => KEEP Config A for production (faster, leaner, comparable recall)")
else:
    delta_pct = (b["pair_completeness"] - a["pair_completeness"]) * 100
    print(f"  Config B (3,5) has {delta_pct:.2f}pp HIGHER recall than Config A")
    print("  => Recall delta is significant; consider switching to Config B")
