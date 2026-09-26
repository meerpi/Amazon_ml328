"""Benchmark Persistent joblib pool vs Non-Persistent joblib vs Single-Process.

Simulates the real streaming ingestion pattern:
Streams multiple chunks of text (e.g., 4 chunks of 50,000 texts = 200,000 texts),
and benchmarks:
  1. Single-process: vectorizer.transform(chunk)
  2. Non-persistent joblib: Parallel(n_jobs=8) instantiated fresh per chunk
  3. Persistent joblib context: with Parallel(n_jobs=8) as parallel reused across all chunks
  4. Persistent ProcessPoolExecutor / loky comparison

Reports:
  - Total transform time (seconds)
  - Throughput (texts / second)
  - Pool initialization / teardown overhead
  - Memory / process behavioral analysis
"""

import gc
import os
import sys
import time
from typing import List

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tfidf_blocker import romanize_text


def load_candidate_texts(source_path: str, max_texts: int = 200000) -> List[str]:
    texts = []
    with open(source_path, "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        c_idx = header.index("country")

        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) > c_idx and parts[c_idx].strip() == "India":
                b_name = parts[name_idx] if len(parts) > name_idx else ""
                b_addr = parts[addr_idx] if len(parts) > addr_idx else ""
                raw_txt = ((b_name or "") + " " + (b_addr or "")).lower()
                txt = raw_txt if raw_txt.isascii() else romanize_text(raw_txt)
                texts.append(txt)
                if len(texts) >= max_texts:
                    break
    return texts


def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: PERSISTENT JOBLIB POOL VS NON-PERSISTENT VS SINGLE-PROCESS")
    print("=" * 80)

    data_path = "student_resource/dataset/train/train_source2.tsv"
    total_target = 200000
    chunk_size = 50000
    n_jobs = 8

    print(f"Loading {total_target:,} real candidate texts from {data_path}...")
    t0 = time.time()
    all_texts = load_candidate_texts(data_path, max_texts=total_target)
    print(f"Loaded {len(all_texts):,} texts in {time.time()-t0:.2f}s")

    # Fit a standard (3,4) char_wb vectorizer on 100k sample
    print("\nFitting TfidfVectorizer (char_wb, 3-4, min_df=5, max_df=0.05)...")
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
    vec.fit(all_texts[:100000])
    vocab_size = len(vec.vocabulary_)
    print(f"Vectorizer fitted in {time.time()-t0:.2f}s | Vocab size: {vocab_size:,}")

    # Split into chunks of 50,000 (simulating streaming candidate batches)
    chunks = [all_texts[i:i + chunk_size] for i in range(0, len(all_texts), chunk_size)]
    n_chunks = len(chunks)
    print(f"\nSimulating streaming of {n_chunks} chunks ({chunk_size:,} texts each, total {len(all_texts):,} texts)\n")

    # ---------------------------------------------------------
    # METHOD 1: SINGLE-PROCESS
    # ---------------------------------------------------------
    print("--- 1. Single-Process vec.transform(chunk) ---")
    gc.collect()
    t_start = time.time()
    res1_chunks = []
    chunk_times_1 = []
    for i, ch in enumerate(chunks):
        tc0 = time.time()
        m = vec.transform(ch)
        dt = time.time() - tc0
        chunk_times_1.append(dt)
        res1_chunks.append(m)
        print(f"  Chunk {i+1}/{n_chunks}: {dt:.2f}s ({len(ch)/dt:.0f} texts/s)")

    time_1 = time.time() - t_start
    rate_1 = len(all_texts) / time_1
    mat_1 = sp.vstack(res1_chunks, format="csr")
    print(f"  TOTAL Time: {time_1:.2f}s | Average Rate: {rate_1:.0f} texts/s\n")

    # ---------------------------------------------------------
    # METHOD 2: NON-PERSISTENT JOBLIB (Fresh Parallel per chunk)
    # ---------------------------------------------------------
    print("--- 2. Non-Persistent joblib: Parallel(n_jobs=8) recreated per chunk ---")
    gc.collect()
    t_start = time.time()
    res2_chunks = []
    chunk_times_2 = []
    for i, ch in enumerate(chunks):
        tc0 = time.time()
        sub_chunks = np.array_split(ch, n_jobs)
        # Fresh Parallel instance every chunk (pool startup + shutdown per chunk)
        m_list = Parallel(n_jobs=n_jobs)(
            delayed(vec.transform)(sc.tolist()) for sc in sub_chunks
        )
        m = sp.vstack(m_list, format="csr")
        dt = time.time() - tc0
        chunk_times_2.append(dt)
        res2_chunks.append(m)
        print(f"  Chunk {i+1}/{n_chunks}: {dt:.2f}s ({len(ch)/dt:.0f} texts/s)")

    time_2 = time.time() - t_start
    rate_2 = len(all_texts) / time_2
    mat_2 = sp.vstack(res2_chunks, format="csr")
    print(f"  TOTAL Time: {time_2:.2f}s | Average Rate: {rate_2:.0f} texts/s\n")

    # ---------------------------------------------------------
    # METHOD 3: PERSISTENT JOBLIB (Reused Parallel context)
    # ---------------------------------------------------------
    print("--- 3. Persistent joblib: with Parallel(n_jobs=8) as parallel (reused pool) ---")
    gc.collect()
    t_start = time.time()
    res3_chunks = []
    chunk_times_3 = []
    with Parallel(n_jobs=n_jobs) as parallel:
        for i, ch in enumerate(chunks):
            tc0 = time.time()
            sub_chunks = np.array_split(ch, n_jobs)
            m_list = parallel(
                delayed(vec.transform)(sc.tolist()) for sc in sub_chunks
            )
            m = sp.vstack(m_list, format="csr")
            dt = time.time() - tc0
            chunk_times_3.append(dt)
            res3_chunks.append(m)
            print(f"  Chunk {i+1}/{n_chunks}: {dt:.2f}s ({len(ch)/dt:.0f} texts/s)")

    time_3 = time.time() - t_start
    rate_3 = len(all_texts) / time_3
    mat_3 = sp.vstack(res3_chunks, format="csr")
    print(f"  TOTAL Time: {time_3:.2f}s | Average Rate: {rate_3:.0f} texts/s\n")

    # ---------------------------------------------------------
    # METHOD 4: PERSISTENT JOBLIB WITH THREADS (backend='threading')
    # ---------------------------------------------------------
    print("--- 4. Persistent joblib with Threads: Parallel(n_jobs=8, prefer='threads') ---")
    gc.collect()
    t_start = time.time()
    res4_chunks = []
    chunk_times_4 = []
    with Parallel(n_jobs=n_jobs, prefer="threads") as parallel_threads:
        for i, ch in enumerate(chunks):
            tc0 = time.time()
            sub_chunks = np.array_split(ch, n_jobs)
            m_list = parallel_threads(
                delayed(vec.transform)(sc.tolist()) for sc in sub_chunks
            )
            m = sp.vstack(m_list, format="csr")
            dt = time.time() - tc0
            chunk_times_4.append(dt)
            res4_chunks.append(m)
            print(f"  Chunk {i+1}/{n_chunks}: {dt:.2f}s ({len(ch)/dt:.0f} texts/s)")

    time_4 = time.time() - t_start
    rate_4 = len(all_texts) / time_4
    mat_4 = sp.vstack(res4_chunks, format="csr")
    print(f"  TOTAL Time: {time_4:.2f}s | Average Rate: {rate_4:.0f} texts/s\n")

    # ---------------------------------------------------------
    # SUMMARY & COMPARISON
    # ---------------------------------------------------------
    print("=" * 80)
    print("FINAL BENCHMARK COMPARISON (200,000 texts, 4 chunks x 50,000)")
    print("=" * 80)
    print(f"{'Method':<35} | {'Time (s)':<10} | {'Throughput (texts/s)':<22} | {'Speedup vs Single':<18}")
    print("-" * 80)
    print(f"{'1. Single-Process':<35} | {time_1:<10.2f} | {rate_1:<22.0f} | 1.00x (baseline)")
    print(f"{'2. Non-Persistent joblib (fresh pool)':<35} | {time_2:<10.2f} | {rate_2:<22.0f} | {time_1/time_2:.2f}x")
    print(f"{'3. Persistent joblib (processes)':<35} | {time_3:<10.2f} | {rate_3:<22.0f} | {time_1/time_3:.2f}x")
    print(f"{'4. Persistent joblib (threads)':<35} | {time_4:<10.2f} | {rate_4:<22.0f} | {time_1/time_4:.2f}x")
    print("-" * 80)

    # Verification of matrix identity
    diff2 = (mat_1 != mat_2).nnz
    diff3 = (mat_1 != mat_3).nnz
    diff4 = (mat_1 != mat_4).nnz
    print(f"Matrix verification: diff vs M2={diff2}, vs M3={diff3}, vs M4={diff4} (0 means exact match)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
