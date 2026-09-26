"""Rigorous evaluation on the WHOLE training candidate dataset (all 10.32 million records in S2 & S3).

Zero subsetting or sampling of candidates: every query is searched against ALL 10,320,219 records.
Evaluates exact candidate recall at k, downstream precision, and competition Macro F_0.5.
"""

import heapq
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn

sys.path.insert(0, ".")
from src.tfidf_blocker import romanize_text, compute_macro_f05

print("=" * 95)
print("WHOLE DATASET EVALUATION: SEARCHING AGAINST ALL 10.32 MILLION CANDIDATES")
print("Data Directory: student_resource/dataset/train")
print("No candidate subsetting: 100% of Source 2 and Source 3 are evaluated.")
print("=" * 95)

# 1. Select 5,000 representative reference queries from train_source1.tsv
N_QUERIES = 5000
print(f"\n[Step 1] Loading {N_QUERIES:,} reference queries from train_source1.tsv...")
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

# 2. Load Ground Truth matches and identify singletons
print("[Step 2] Loading Ground Truth matches from train_ground_truth.tsv...")
gt_map: Dict[str, Set[str]] = {}
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
print(f"    Total true positive links: {total_true_matches:,}")

# Partition queries by country
s1_by_country = defaultdict(lambda: {"ids": [], "texts": [], "id_to_idx": {}})
for r in s1_records:
    c = r["country"]
    c_dict = s1_by_country[c]
    idx = len(c_dict["ids"])
    c_dict["ids"].append(r["entity_id"])
    c_dict["texts"].append(r["text"])
    c_dict["id_to_idx"][r["entity_id"]] = idx

for c in sorted(s1_by_country.keys()):
    print(f"    - {c:10s}: {len(s1_by_country[c]['ids']):,} queries")

# 3. Fit TF-IDF Vectorizers per country on 150k representative records
print("\n[Step 3] Fitting TF-IDF Vectorizers (char_wb 3-5, min_df=2) per country...")
vectorizers = {}
mat_queries = {}

for country in s1_by_country:
    q_texts = s1_by_country[country]["texts"]
    sample_texts = q_texts.copy()
    
    # Add candidate samples to vocabulary
    with open("student_resource/dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) > 3 and parts[3].strip() == country:
                txt = romanize_text(((parts[1] or "") + " " + (parts[2] or "")).lower())
                sample_texts.append(txt)
                if len(sample_texts) >= 150000:
                    break
                    
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )
    t0 = time.time()
    vec.fit(sample_texts)
    vectorizers[country] = vec
    mat_queries[country] = vec.transform(q_texts)
    print(f"  [{country}] Vectorizer fitted in {time.time()-t0:.2f}s! Vocab size: {len(vec.vocabulary_):,}")

# 4. Stream and Search ALL 10.32 MILLION CANDIDATES in train_source2.tsv and train_source3.tsv
print("\n[Step 4] Streaming and Searching ALL 10,320,219 candidates from Source 2 and Source 3...")
K_CAPACITY = 50
SIMILARITY_FLOOR = 0.05

# Min-heap per query: top_candidates[country][q_idx] -> list of (score, cand_id)
top_candidates = {
    c: [[] for _ in range(len(s1_by_country[c]["ids"]))]
    for c in s1_by_country
}

CHUNK_SIZE = 500000
total_candidates_scanned = 0
t_stream_start = time.time()

for fname in ["train_source2.tsv", "train_source3.tsv"]:
    fpath = f"student_resource/dataset/train/{fname}"
    print(f"\nStreaming candidates from {fname}...")
    
    chunk_by_country = defaultdict(lambda: {"ids": [], "texts": []})
    chunk_count = 0
    
    with open(fpath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            total_candidates_scanned += 1
            chunk_count += 1
            parts = line.strip("\n").split("\t")
            cid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            c = parts[3].strip() if len(parts) > 3 else ""
            
            if c in s1_by_country:
                chunk_by_country[c]["ids"].append(cid)
                chunk_by_country[c]["texts"].append(romanize_text((name + " " + addr).lower()))
                
            if chunk_count >= CHUNK_SIZE:
                # Process chunks for each country
                t_sub = time.time()
                for c in chunk_by_country:
                    c_ids = chunk_by_country[c]["ids"]
                    c_texts = chunk_by_country[c]["texts"]
                    if not c_ids or c not in vectorizers:
                        continue
                        
                    mat_c = vectorizers[c].transform(c_texts)
                    sim_mat = awesome_cossim_topn(
                        mat_queries[c],
                        mat_c.T,
                        ntop=K_CAPACITY,
                        lower_bound=SIMILARITY_FLOOR
                    )
                    
                    indptr = sim_mat.indptr
                    indices = sim_mat.indices
                    data = sim_mat.data
                    
                    for q_idx in range(len(indptr) - 1):
                        r_start = indptr[q_idx]
                        r_end = indptr[q_idx + 1]
                        if r_start == r_end:
                            continue
                            
                        heap = top_candidates[c][q_idx]
                        for ptr in range(r_start, r_end):
                            score = float(data[ptr])
                            cand_id = c_ids[indices[ptr]]
                            
                            if len(heap) < K_CAPACITY:
                                heapq.heappush(heap, (score, cand_id))
                            elif score > heap[0][0]:
                                heapq.heapreplace(heap, (score, cand_id))
                                
                chunk_by_country.clear()
                chunk_count = 0
                elapsed_total = time.time() - t_stream_start
                print(f"  Scanned {total_candidates_scanned:,} / 10,320,219 candidates ({elapsed_total:.1f}s, {total_candidates_scanned/elapsed_total:.0f} cands/s)...")

    # Flush remainder of file
    for c in chunk_by_country:
        c_ids = chunk_by_country[c]["ids"]
        c_texts = chunk_by_country[c]["texts"]
        if not c_ids or c not in vectorizers:
            continue
            
        mat_c = vectorizers[c].transform(c_texts)
        sim_mat = awesome_cossim_topn(
            mat_queries[c],
            mat_c.T,
            ntop=K_CAPACITY,
            lower_bound=SIMILARITY_FLOOR
        )
        
        indptr = sim_mat.indptr
        indices = sim_mat.indices
        data = sim_mat.data
        
        for q_idx in range(len(indptr) - 1):
            r_start = indptr[q_idx]
            r_end = indptr[q_idx + 1]
            if r_start == r_end:
                continue
                
            heap = top_candidates[c][q_idx]
            for ptr in range(r_start, r_end):
                score = float(data[ptr])
                cand_id = c_ids[indices[ptr]]
                
                if len(heap) < K_CAPACITY:
                    heapq.heappush(heap, (score, cand_id))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, cand_id))

total_scan_time = time.time() - t_stream_start
print(f"\nALL 10,320,219 CANDIDATES SCANNED IN {total_scan_time:.2f}s ({total_scan_time/60:.2f} minutes)!")

# 5. Assemble Sorted Predictions
pred_by_s1 = defaultdict(list)
for c in s1_by_country:
    q_ids = s1_by_country[c]["ids"]
    for q_idx, qid in enumerate(q_ids):
        heap = top_candidates[c][q_idx]
        sorted_cands = sorted(heap, key=lambda x: x[0], reverse=True)
        # Store (cid, score)
        pred_by_s1[qid] = [(cid, score) for score, cid in sorted_cands]

# 6. Evaluate Recall at different k across ALL 10.32 MILLION CANDIDATES
print("\n" + "=" * 95)
print("EVALUATION RESULTS ACROSS THE ENTIRE 10.32 MILLION CANDIDATE UNIVERSE")
print("=" * 95)

us_ids = s1_by_country["US"]["ids"]
in_ids = s1_by_country["India"]["ids"]

print(f"{'k':>4} | {'US Recall':>11} | {'India Recall':>14} | {'Overall Recall':>16} | {'Macro Precision':>17} | {'Macro F0.5':>12}")
print("-" * 95)

for k in [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
    m_us = compute_macro_f05(pred_by_s1, gt_map, us_ids, k=k)
    m_in = compute_macro_f05(pred_by_s1, gt_map, in_ids, k=k)
    m_all = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=k)
    print(
        f"{k:4d} | {m_us['pair_completeness']*100:10.2f}% | "
        f"{m_in['pair_completeness']*100:13.2f}% | "
        f"{m_all['pair_completeness']*100:15.2f}% | "
        f"{m_all['macro_precision']*100:16.2f}% | "
        f"{m_all['macro_f05']:12.4f}"
    )
print("-" * 95)

# 7. Similarity Threshold Sweep (tau) for matching_results.tsv at k=30
print("\n[Step 7] Downstream Thresholding Sweep (tau) at k=30 across ALL 10.32M Candidates:")
print("-" * 80)
print(f"{'Threshold (tau)':>16} | {'Pair Recall':>14} | {'Macro Precision':>18} | {'Macro F_0.5':>14}")
print("-" * 80)

for tau in [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
    m_tau = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=30, threshold=tau)
    print(
        f"{tau:16.2f} | {m_tau['pair_completeness']*100:13.2f}% | "
        f"{m_tau['macro_precision']*100:17.2f}% | "
        f"{m_tau['macro_f05']:14.4f}"
    )
print("-" * 80)

# 8. Check Singleton Performance
singletons_perfect = 0
for qid in s1_ids:
    if len(gt_map[qid]) == 0:  # Singleton
        # Check predictions at tau=0.55
        cands_at_55 = [cid for cid, score in pred_by_s1.get(qid, []) if score >= 0.55]
        if len(cands_at_55) == 0:
            singletons_perfect += 1

print(f"\n[Step 8] Singleton Performance Verification across 10.32M Distractors:")
print(f"  Total Singletons evaluated: {singletons:,}")
print(f"  Singletons correctly predicted empty (Score = 1.0): {singletons_perfect:,} ({singletons_perfect/max(1, singletons):.2%})")
print("=" * 95)
