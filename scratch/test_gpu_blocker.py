"""GPU-Accelerated Candidate Retrieval using FAISS-GPU and TruncatedSVD.

Demonstrates Step 7 of the user requirements:
  - Dimensionality reduction: TF-IDF -> TruncatedSVD(n_components=128)
  - Indexing: faiss.IndexFlatIP on NVIDIA RTX 3060 Laptop (6GB VRAM)
  - Drop-in schema: (source1_entity_id, candidate_entity_id, score, rank, country)
  - VRAM tracking and graceful CPU fallback
  - Benchmark on the same 5,000-query sample
"""

import gc
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tfidf_blocker import romanize_text, compute_macro_f05


def get_gpu_vram_mb() -> float:
    """Returns currently allocated GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024 * 1024)
    return 0.0


def get_gpu_total_vram_mb() -> float:
    """Returns total device VRAM in MB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    return 0.0


def estimate_vram_requirement_mb(n_candidates: int, dim: int = 128) -> float:
    """Estimates VRAM needed for candidate vectors and Faiss IndexFlatIP."""
    raw_data_mb = (n_candidates * dim * 4) / (1024 * 1024)
    # Faiss IndexFlatIP overhead + PyTorch/CUDA context overhead (~400MB)
    return raw_data_mb + 450.0


def run_gpu_benchmark():
    print("=" * 80)
    print("STEP 7: GPU-ACCELERATED CANDIDATE RETRIEVAL BENCHMARK")
    print("=" * 80)

    # Check CUDA device
    assert torch.cuda.is_available(), "CUDA is not available!"
    dev_name = torch.cuda.get_device_name(0)
    total_vram_mb = get_gpu_total_vram_mb()
    print(f"GPU Device: {dev_name} | Total VRAM: {total_vram_mb:.1f} MB ({total_vram_mb/1024:.2f} GB)")

    # 1. Load benchmark sample (same 5,000 queries as Step 1)
    train_dir = "student_resource/dataset/train"
    print("\n[Step 1] Loading 5,000 real S1 queries...")
    s1_rows = []
    with open(f"{train_dir}/train_source1.tsv", "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        id_i = header.index("entity_id")
        name_i = header.index("business_name")
        addr_i = header.index("business_address")
        c_i = header.index("country")
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) > c_i and parts[c_i].strip() == "India":
                b_name = parts[name_i] if len(parts) > name_i else ""
                b_addr = parts[addr_i] if len(parts) > addr_i else ""
                raw_txt = ((b_name or "") + " " + (b_addr or "")).lower()
                txt = raw_txt if raw_txt.isascii() else romanize_text(raw_txt)
                s1_rows.append((parts[id_i], txt))
                if len(s1_rows) >= 5000:
                    break

    s1_ids = [r[0] for r in s1_rows]
    s1_texts = [r[1] for r in s1_rows]
    q_set = set(s1_ids)

    # 2. Load ground truth
    print("[Step 2] Loading ground truth...")
    gt_map: Dict[str, Set[str]] = {qid: set() for qid in s1_ids}
    with open(f"{train_dir}/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                s1_e, cand_e = parts[0], parts[1]
                if s1_e in q_set:
                    gt_map[s1_e].add(cand_e)

    # 3. Assemble candidate pool (true candidates + distractors)
    print("[Step 3] Assembling candidate pool (true candidates + distractors)...")
    all_true_cands = set().union(*gt_map.values())

    cand_rows = []
    found_true = set()
    distractor_limit = 150000
    n_distractors = 0

    for s_name in ["train_source2.tsv", "train_source3.tsv"]:
        with open(f"{train_dir}/{s_name}", "r", encoding="utf-8") as f:
            header = next(f).strip().split("\t")
            id_i = header.index("entity_id")
            name_i = header.index("business_name")
            addr_i = header.index("business_address")
            c_i = header.index("country")
            for line in f:
                parts = line.strip("\n").split("\t")
                if len(parts) > c_i and parts[c_i].strip() == "India":
                    cid = parts[id_i]
                    b_name = parts[name_i] if len(parts) > name_i else ""
                    b_addr = parts[addr_i] if len(parts) > addr_i else ""
                    raw_txt = ((b_name or "") + " " + (b_addr or "")).lower()
                    txt = raw_txt if raw_txt.isascii() else romanize_text(raw_txt)

                    if cid in all_true_cands and cid not in found_true:
                        cand_rows.append((cid, txt))
                        found_true.add(cid)
                    elif n_distractors < distractor_limit:
                        cand_rows.append((cid, txt))
                        n_distractors += 1

    cand_ids = np.array([r[0] for r in cand_rows])
    cand_texts = [r[1] for r in cand_rows]
    n_cands = len(cand_ids)
    print(f"  Pool assembled: {n_cands:,} candidates ({len(found_true):,} true + {n_distractors:,} distractors)")

    # 4. Fit TF-IDF Vectorizer
    print("\n[Step 4] Fitting TF-IDF Vectorizer (3,4, min_df=5, max_df=0.05)...")
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=5,
        max_df=0.05,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )
    vec.fit(cand_texts[:100000])
    vocab_size = len(vec.vocabulary_)
    print(f"  Vectorizer fitted in {time.time()-t0:.2f}s | Vocab: {vocab_size:,}")

    # 5. Transform to TF-IDF sparse matrices
    print("\n[Step 5] Transforming to TF-IDF sparse matrices...")
    t0 = time.time()
    cand_tfidf = vec.transform(cand_texts)
    query_tfidf = vec.transform(s1_texts)
    print(f"  Transform time: {time.time()-t0:.2f}s | Cand matrix: {cand_tfidf.shape}, Query: {query_tfidf.shape}")

    # 6. Dimensionality Reduction: TruncatedSVD to 128 components
    dim = 128
    print(f"\n[Step 6] Fitting TruncatedSVD(n_components={dim})...")
    t0 = time.time()
    svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=3, random_state=42)
    # Fit on sample of candidates
    svd.fit(cand_tfidf[:50000])
    svd_fit_time = time.time() - t0
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"  SVD fitted in {svd_fit_time:.2f}s | Cumulative Explained Variance: {explained_var*100:.2f}%")

    print(f"  Transforming candidates & queries to {dim}-dim dense vectors...")
    t0 = time.time()
    cand_dense = svd.transform(cand_tfidf).astype(np.float32)
    query_dense = svd.transform(query_tfidf).astype(np.float32)
    print(f"  Dense transform done in {time.time()-t0:.2f}s")

    # L2 normalize for cosine similarity via inner product (IndexFlatIP)
    faiss.normalize_L2(cand_dense)
    faiss.normalize_L2(query_dense)

    # 7. Check VRAM & Build FAISS-GPU Index
    vram_est_mb = estimate_vram_requirement_mb(n_cands, dim=dim)
    print(f"\n[Step 7] VRAM Check for {n_cands:,} candidates (dim={dim}):")
    print(f"  Estimated VRAM needed: {vram_est_mb:.1f} MB")
    print(f"  Total Available VRAM: {total_vram_mb:.1f} MB")

    use_gpu = vram_est_mb < (total_vram_mb - 500)
    print(f"  Safe for GPU execution? {use_gpu}")

    vram_before = get_gpu_vram_mb()
    t0 = time.time()
    if use_gpu:
        try:
            print("  Building faiss.IndexFlatIP on GPU...")
            res = faiss.StandardGpuResources()
            index_cpu = faiss.IndexFlatIP(dim)
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            index.add(cand_dense)
            index_type = "GPU IndexFlatIP"
        except Exception as e:
            print(f"  GPU indexing failed ({e}), falling back gracefully to CPU IndexFlatIP...")
            index = faiss.IndexFlatIP(dim)
            index.add(cand_dense)
            index_type = "CPU Fallback IndexFlatIP"
    else:
        print("  Candidate pool exceeds safe VRAM limit! Gracefully using CPU IndexFlatIP...")
        index = faiss.IndexFlatIP(dim)
        index.add(cand_dense)
        index_type = "CPU Fallback IndexFlatIP"

    index_time = time.time() - t0
    vram_after = get_gpu_vram_mb()
    print(f"  Index built: {index_type} in {index_time:.3f}s")
    print(f"  VRAM allocated by PyTorch/Faiss: {vram_after - vram_before:.2f} MB")

    # 8. Query retrieval on GPU
    k = 30
    print(f"\n[Step 8] Batch-querying {len(query_dense):,} queries (k={k}) on {index_type}...")
    t0 = time.time()
    D, I = index.search(query_dense, k)
    search_time = time.time() - t0
    q_per_sec = len(query_dense) / search_time
    print(f"  Search completed in {search_time:.3f}s ({q_per_sec:.1f} queries/sec)!")

    # 9. Format results in drop-in schema: (source1_entity_id, candidate_entity_id, score, rank, country)
    print("\n[Step 9] Formatting candidates in standard schema:")
    print("  Schema: (source1_entity_id, candidate_entity_id, score, rank, country)")
    predictions_by_s1: Dict[str, List[Tuple[str, float]]] = {qid: [] for qid in s1_ids}
    schema_sample = []

    for row_i, qid in enumerate(s1_ids):
        for rank in range(k):
            cand_idx = I[row_i, rank]
            score = float(D[row_i, rank])
            if cand_idx >= 0:
                cid = str(cand_ids[cand_idx])
                predictions_by_s1[qid].append((cid, score))
                if len(schema_sample) < 5:
                    schema_sample.append((qid, cid, round(score, 4), rank + 1, "India"))

    print("  Sample rows:")
    for row in schema_sample:
        print(f"    {row}")

    # 10. Evaluate Accuracy
    print("\n[Step 10] Evaluating Accuracy against Ground Truth:")
    metrics = compute_macro_f05(predictions_by_s1, gt_map, s1_ids, k=k, threshold=0.1)
    print(f"  pair_completeness: {metrics['pair_completeness']*100:.2f}%")
    print(f"  macro_precision:   {metrics['macro_precision']*100:.2f}%")
    print(f"  macro_recall:      {metrics['macro_recall']*100:.2f}%")
    print(f"  macro_f05:         {metrics['macro_f05']:.4f}")

    # 11. VRAM Scaling Analysis for Full Dataset
    print("\n" + "=" * 80)
    print("VRAM SCALING ANALYSIS FOR ACTUAL TEST & TRAIN PARTITIONS")
    print("=" * 80)
    partitions = [
        ("India (Train S2+S3)", 4133346),
        ("US (Train S2+S3)", 6186873),
        ("India (Test S2+S3)", 4717565),
        ("US (Test S2+S3)", 3817031),
        ("France (Test S2+S3)", 1434993),
        ("India (Test Single Pass S2)", 2312565),
        ("India (Test Single Pass S3)", 2405000),
    ]
    print(f"{'Partition':<30} | {'Candidates':<12} | {'Est VRAM (d=128)':<18} | {'Fits 6GB GPU?':<14}")
    print("-" * 80)
    for p_name, n_cand in partitions:
        est_mb = estimate_vram_requirement_mb(n_cand, dim=128)
        fits = "YES (Direct)" if est_mb < 5000 else "NO (Use CPU/Chunk)"
        print(f"{p_name:<30} | {n_cand:<12,} | {est_mb:>7.1f} MB ({est_mb/1024:.2f}GB) | {fits:<14}")

    print("=" * 80)


if __name__ == "__main__":
    run_gpu_benchmark()
