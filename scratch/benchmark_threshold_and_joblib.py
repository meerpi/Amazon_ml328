"""Benchmark: candidate_threshold 0.08 vs 0.20, and joblib vs single-process transform.

Reports queries/sec before and after candidate_threshold change,
plus transform time with/without joblib.Parallel.
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tfidf_blocker import romanize_text, compute_macro_f05

N_QUERIES = 5000

print("=" * 95)
print("BENCHMARK: candidate_threshold AND joblib removal")
print("=" * 95)

# 1. Load queries
print(f"\n[Step 1] Loading {N_QUERIES:,} queries...")
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
        if len(s1_ids) >= N_QUERIES:
            break

# 2. Load GT
print("[Step 2] Loading GT...")
s1_id_set = set(s1_ids)
gt_map: Dict[str, Set[str]] = {}
needed_cand_ids = set()
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
        if len(gt_map) >= len(s1_ids):
            break
for s1 in s1_ids:
    if s1 not in gt_map:
        gt_map[s1] = set()

# 3. Load candidates
print("[Step 3] Loading candidate pool...")
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

# 4. Fit vectorizer
vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 4), min_df=5, max_df=0.05,
    sublinear_tf=True, dtype=np.float32, norm="l2",
)
vec.fit(cand_texts[:150000])
print(f"  Vocab: {len(vec.vocabulary_):,}")

# 5. Transform candidates and queries
mat_cand = vec.transform(cand_texts)
mat_q = vec.transform(query_texts)
mat_cand_T = mat_cand.T  # CSR -> CSC zero-copy

# ============================================================
# Benchmark A: candidate_threshold 0.08 vs 0.20
# ============================================================
print("\n" + "=" * 80)
print("BENCHMARK A: candidate_threshold 0.08 vs 0.20")
print("=" * 80)

for thresh in [0.08, 0.20]:
    t0 = time.time()
    sim_res = sp_matmul_topn(
        mat_q, mat_cand_T, top_n=30, threshold=thresh,
        n_threads=8, sort=True,
    )
    elapsed = time.time() - t0
    qps = len(s1_ids) / elapsed

    # Check recall
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

    m = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=30)
    print(f"  threshold={thresh:.2f}: {elapsed:.3f}s ({qps:.1f} q/s) | "
          f"nnz={sim_res.nnz:,} | "
          f"pair_comp={m['pair_completeness']*100:.2f}% | "
          f"macro_prec={m['macro_precision']*100:.2f}%")

# ============================================================
# Benchmark B: joblib.Parallel vs single-process transform
# ============================================================
print("\n" + "=" * 80)
print("BENCHMARK B: joblib.Parallel(n_jobs=8) vs single-process vectorizer.transform")
print("=" * 80)

texts_to_transform = cand_texts[:500000] if len(cand_texts) >= 500000 else cand_texts
n_texts = len(texts_to_transform)
print(f"  Transforming {n_texts:,} texts...")

# Single-process
t0 = time.time()
mat_single = vec.transform(texts_to_transform)
t_single = time.time() - t0
print(f"  Single-process: {t_single:.2f}s ({n_texts/t_single:.0f} texts/sec)")

# joblib.Parallel (old method)
def parallel_transform_old(vec_obj, texts, n_jobs=8):
    if len(texts) < 10000:
        return vec_obj.transform(texts)
    chunks = np.array_split(texts, n_jobs)
    matrices = Parallel(n_jobs=n_jobs)(delayed(vec_obj.transform)(c.tolist()) for c in chunks)
    return sp.vstack(matrices, format="csr")

t0 = time.time()
mat_par = parallel_transform_old(vec, texts_to_transform, n_jobs=8)
t_par = time.time() - t0
print(f"  joblib.Parallel: {t_par:.2f}s ({n_texts/t_par:.0f} texts/sec)")

speedup = t_par / t_single
print(f"\n  Result: single-process is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than joblib")
print(f"  Matrices identical shape: {mat_single.shape == mat_par.shape}")
print(f"  Matrices identical nnz: {mat_single.nnz == mat_par.nnz}")
