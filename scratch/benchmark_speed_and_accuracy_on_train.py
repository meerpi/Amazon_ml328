import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, ".")
from src.tfidf_blocker import romanize_text, compute_macro_f05

print("=" * 95)
print("RIGOROUS BENCHMARK: SPEED VS. ACCURACY ON TRAINING DATASET")
print("Directory: student_resource/dataset/train")
print("=" * 95)

# 1. Load real S1 queries from training set
N_QUERIES = 3000
print(f"\n[Step 1] Loading {N_QUERIES:,} real reference queries from train_source1.tsv...")
s1_records = []
s1_ids = []
with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip("\n").split("\t")
        eid = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        country = parts[3] if len(parts) > 3 else ""
        s1_records.append({
            "entity_id": eid,
            "business_name": name,
            "business_address": addr,
            "country": country,
            "text": romanize_text((name + " " + addr).lower())
        })
        s1_ids.append(eid)
        if len(s1_records) >= N_QUERIES:
            break

s1_id_set = set(s1_ids)

# 2. Load ground truth matches
print("[Step 2] Loading Ground Truth matches from train_ground_truth.tsv...")
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

total_true_matches = sum(len(c) for c in gt_map.values())
print(f"  Loaded Ground Truth for {len(gt_map):,} queries:")
print(f"    Queries with matches: {len(gt_map) - singletons:,}")
print(f"    Singletons (0 matches): {singletons:,} ({singletons/len(gt_map):.2%})")
print(f"    Total true matching links: {total_true_matches:,}")

# 3. Assemble candidate pool (all true candidates + 150,000 real distractors)
print("\n[Step 3] Loading candidate pool (all true candidates + 150,000 real distractors)...")
cand_records = []
cand_ids = []
cand_texts = []
EXTRA_DISTRACTORS = 150000
distractors_loaded = 0

for fname in ["train_source2.tsv", "train_source3.tsv"]:
    fpath = f"student_resource/dataset/train/{fname}"
    with open(fpath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\n").split("\t")
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            txt = romanize_text((name + " " + addr).lower())

            if eid in needed_cand_ids:
                cand_ids.append(eid)
                cand_texts.append(txt)
            elif distractors_loaded < EXTRA_DISTRACTORS:
                cand_ids.append(eid)
                cand_texts.append(txt)
                distractors_loaded += 1

c_id_arr = np.array(cand_ids)
n_cands = len(cand_ids)
print(f"  Candidate pool assembled: {n_cands:,} total candidate records (100% of true matches present)")

query_texts = [r["text"] for r in s1_records]

# 4. Benchmarking Methods
print("\n" + "=" * 95)
print("BENCHMARKING CANDIDATE BLOCKING METHODS")
print("=" * 95)

methods = [
    {
        "name": "Method A: Exact tfidf_blocker.py Baseline (Char_wb 3-5, min_df=2, no max_df)",
        "vectorizer": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        "two_stage": False,
    },
    {
        "name": "Method B: Frequency-Pruned Char_wb (3-5, min_df=5, max_df=0.15)",
        "vectorizer": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.15, sublinear_tf=True),
        "two_stage": False,
    },
    {
        "name": "Method C: Fast Word (1-2, min_df=3, max_df=0.20)",
        "vectorizer": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.20, sublinear_tf=True),
        "two_stage": False,
    },
    {
        "name": "Method D: Two-Stage (Stage 1: Fast Word Top-60 -> Stage 2: Char_wb 3-5 Rescoring)",
        "vectorizer": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.20, sublinear_tf=True),
        "rescore_vectorizer": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.15, sublinear_tf=True),
        "two_stage": True,
    },
]

results = []

for m in methods:
    print(f"\nEvaluating: {m['name']}...")
    vec = m["vectorizer"]
    
    t0_fit = time.time()
    vec.fit(cand_texts[:50000])
    mat_cand_T = vec.transform(cand_texts).T.tocsr()
    mat_q = vec.transform(query_texts)
    fit_and_transform_time = time.time() - t0_fit
    
    t0_search = time.time()
    
    if not m["two_stage"]:
        sim_res = sp_matmul_topn(mat_q, mat_cand_T, top_n=30, threshold=0.08, n_threads=8, sort=True)
        search_time = time.time() - t0_search
        
        indptr = sim_res.indptr
        indices = sim_res.indices
        data = sim_res.data
        
        pred_by_s1 = defaultdict(list)
        for i, qid in enumerate(s1_ids):
            r_start, r_end = indptr[i], indptr[i+1]
            if r_start < r_end:
                row_cand_ids = c_id_arr[indices[r_start:r_end]]
                row_scores = data[r_start:r_end]
                for cid, sc in zip(row_cand_ids, row_scores):
                    pred_by_s1[qid].append((cid, float(sc)))
    else:
        # Two-stage: Fast word retrieval top-60, then rescore candidates with fine char n-grams
        sim_res = sp_matmul_topn(mat_q, mat_cand_T, top_n=60, threshold=0.05, n_threads=8, sort=True)
        
        r_vec = m["rescore_vectorizer"]
        r_vec.fit(cand_texts[:50000])
        
        indptr = sim_res.indptr
        indices = sim_res.indices
        
        # Batch rescoring
        pred_by_s1 = defaultdict(list)
        for i, qid in enumerate(s1_ids):
            r_start, r_end = indptr[i], indptr[i+1]
            if r_start < r_end:
                cand_sub_indices = indices[r_start:r_end]
                cand_sub_ids = c_id_arr[cand_sub_indices]
                cand_sub_texts = [cand_texts[idx] for idx in cand_sub_indices]
                
                # Rescore only these <=60 candidates
                q_vec = r_vec.transform([query_texts[i]])
                c_mat = r_vec.transform(cand_sub_texts)
                scores = (q_vec @ c_mat.T).toarray().ravel()
                
                # Sort top-30
                top_order = np.argsort(-scores)[:30]
                for top_i in top_order:
                    if scores[top_i] >= 0.08:
                        pred_by_s1[qid].append((cand_sub_ids[top_i], float(scores[top_i])))
                        
        search_time = time.time() - t0_search
        
    # Evaluate Accuracy Metrics
    eval_k30 = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=30)
    eval_tau55 = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=30, threshold=0.55)
    
    q_per_sec = len(s1_ids) / search_time
    
    results.append({
        "Method": m["name"][:35] + "...",
        "Search Time (s)": search_time,
        "Queries/sec": q_per_sec,
        "Pair Recall@30": eval_k30["pair_completeness"],
        "Precision (tau=0.55)": eval_tau55["macro_precision"],
        "Macro F0.5 (tau=0.55)": eval_tau55["macro_f05"],
    })
    
    print(f"  -> Search Time: {search_time:.2f}s ({q_per_sec:.1f} q/s)")
    print(f"  -> Pair Recall@30: {eval_k30['pair_completeness']*100:.2f}%")
    print(f"  -> Macro Precision (@tau=0.55): {eval_tau55['macro_precision']*100:.2f}%")
    print(f"  -> Macro F_0.5 (@tau=0.55): {eval_tau55['macro_f05']:.4f}")

# 5. Summary Table
print("\n" + "=" * 105)
print(f"{'Method':<38} | {'Search Time':>11} | {'Queries/sec':>12} | {'Recall@30':>10} | {'Prec (tau=.55)':>15} | {'Macro F0.5':>11}")
print("=" * 105)
for r in results:
    print(f"{r['Method']:<38} | {r['Search Time']:>10.2f}s | {r['Queries/sec']:>11.1f} | {r['Pair Recall@30']*100:>9.2f}% | {r['Precision (tau=0.55)']*100:>14.2f}% | {r['Macro F0.5 (tau=0.55)']:>11.4f}")
print("=" * 105)
