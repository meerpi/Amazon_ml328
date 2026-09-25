"""Sparkly-style character n-gram TF-IDF top-k blocking module for large-scale entity resolution.

This module implements memory-bounded, country-partitioned candidate generation
for business entity matching across multi-million record datasets.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import uroman as ur
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn


_uroman_instance: Optional[ur.Uroman] = None
_roman_word_cache: Dict[str, str] = {}


def get_uroman() -> ur.Uroman:
    """Lazily initializes the universal offline romanizer instance."""
    global _uroman_instance
    if _uroman_instance is None:
        _uroman_instance = ur.Uroman()
    return _uroman_instance


def romanize_text(s: str) -> str:
    """Normalizes and romanizes non-ASCII scripts (Devanagari, Bengali, French accents)
    to standard Latin characters using uroman with word-level caching.
    """
    if not s or s.isascii():
        return s
    u = get_uroman()
    words = s.split()
    rom_words = []
    for w in words:
        if w.isascii():
            rom_words.append(w)
        else:
            cached = _roman_word_cache.get(w)
            if cached is None:
                cached = u.romanize_string_core(w, None, ur.RomFormat.STR, 0)
                _roman_word_cache[w] = cached
            rom_words.append(cached)
    return " ".join(rom_words)


def build_text_representation(
    df: pd.DataFrame,
    name_col: str = "business_name",
    address_col: str = "business_address",
    romanize: bool = True,
) -> List[str]:
    """Combines business name and address into a single normalized text representation.

    Ensures missing values are replaced with empty strings to prevent string-coercion
    artifacts (e.g. float 'nan'). Optionally applies universal offline romanization (uroman)
    to convert native Indic and accented scripts to Latin characters.
    """
    names = df[name_col].fillna("").astype(str)
    addresses = df[address_col].fillna("").astype(str)
    combined = (names + " " + addresses).tolist()
    if romanize:
        return [romanize_text(s) for s in combined]
    return combined


def block_country_partition(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    country: str,
    k: int = 30,
    lower_bound: float = 0.1,
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 2,
    sublinear_tf: bool = True,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    address_col: str = "business_address",
) -> pd.DataFrame:
    """Performs character n-gram TF-IDF top-k blocking for a single country partition.

    Fits a dedicated TfidfVectorizer on the combined vocabulary of reference and
    candidate entities in this partition, transforms both sets into L2-normalized
    sparse CSR matrices (float32), and uses `sparse_dot_topn.awesome_cossim_topn`
    to compute top-k cosine similarity candidates above `lower_bound`.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Query entities (Source 1) belonging to `country`.
    candidate_df : pd.DataFrame
        Pool of candidate entities (Source 2 and Source 3) belonging to `country`.
    country : str
        Country partition identifier (e.g. 'US', 'India', 'France').
    k : int
        Maximum number of candidate records to retrieve per reference record.
    lower_bound : float
        Minimum cosine similarity threshold required to retain a candidate pair.
    ngram_range : Tuple[int, int]
        Character boundary n-gram range for TfidfVectorizer (default: (3, 5)).
    min_df : int
        Minimum document frequency cutoff to prune idiosyncratic n-grams (default: 2).
    sublinear_tf : bool
        Applies sublinear scaling (1 + log(tf)) to dampen high-frequency tokens.
    id_col : str
        Column name holding the unique entity ID.
    name_col : str
        Column name holding the business name.
    address_col : str
        Column name holding the business address.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        - `source1_entity_id`: ID of the reference query entity.
        - `candidate_entity_id`: ID of the retrieved candidate entity.
        - `similarity_score`: Cosine similarity score (float32).
        - `rank`: Rank of the candidate for this query (1 to k).
        - `country`: Country partition label.
    """
    if reference_df.empty or candidate_df.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "similarity_score",
                "rank",
                "country",
            ]
        )

    ref_texts = build_text_representation(reference_df, name_col, address_col)
    cand_texts = build_text_representation(candidate_df, name_col, address_col)

    # Fit country-specific vectorizer on pooled texts to avoid OOV bias
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        dtype=np.float32,
        norm="l2",
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    vectorizer.fit(ref_texts + cand_texts)

    # Transform without densifying; matrices remain sparse float32 CSR
    mat_ref = vectorizer.transform(ref_texts)
    mat_cand = vectorizer.transform(cand_texts)

    # Sparse dot product + top-n selection per query
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        sim_matrix: sp.csr_matrix = awesome_cossim_topn(
            mat_ref,
            mat_cand.T,
            ntop=k,
            lower_bound=lower_bound,
        )

    # Vectorized unpacking of CSR row-pointers and column indices
    counts = np.diff(sim_matrix.indptr)
    if len(counts) == 0 or sim_matrix.nnz == 0:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "similarity_score",
                "rank",
                "country",
            ]
        )

    row_indices = np.repeat(np.arange(sim_matrix.shape[0]), counts)
    col_indices = sim_matrix.indices
    scores = sim_matrix.data

    # Generate 1-indexed rank within each reference query
    ranks = np.concatenate([np.arange(1, c + 1) for c in counts])

    ref_id_arr = reference_df[id_col].to_numpy()
    cand_id_arr = candidate_df[id_col].to_numpy()

    return pd.DataFrame(
        {
            "source1_entity_id": ref_id_arr[row_indices],
            "candidate_entity_id": cand_id_arr[col_indices],
            "similarity_score": scores,
            "rank": ranks,
            "country": country,
        }
    )


def run_partitioned_blocking(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    k: int = 30,
    lower_bound: float = 0.1,
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 2,
    country_col: str = "country",
    id_col: str = "entity_id",
    name_col: str = "business_name",
    address_col: str = "business_address",
) -> pd.DataFrame:
    """Iterates through country partitions sequentially to execute memory-bounded blocking.

    Processes one country partition at a time to prevent high cumulative RAM spikes
    when scaling across millions of rows.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Complete reference dataset (Source 1).
    candidate_df : pd.DataFrame
        Pooled candidate dataset (Source 2 and Source 3).
    k : int
        Top-k candidates per reference query.
    lower_bound : float
        Cosine similarity cutoff.
    ngram_range : Tuple[int, int]
        Character boundary n-gram range.
    min_df : int
        Minimum document frequency cutoff.
    country_col : str
        Column name holding the country string label.
    id_col : str
        Entity ID column name.
    name_col : str
        Business name column name.
    address_col : str
        Business address column name.

    Returns
    -------
    pd.DataFrame
        Combined candidate pairs across all country partitions with similarity scores and ranks.
    """
    country_partitions = reference_df[country_col].dropna().unique()
    all_results: List[pd.DataFrame] = []

    for country in country_partitions:
        sub_ref = reference_df[reference_df[country_col] == country]
        sub_cand = candidate_df[candidate_df[country_col] == country]

        if sub_cand.empty:
            continue

        part_res = block_country_partition(
            reference_df=sub_ref,
            candidate_df=sub_cand,
            country=str(country),
            k=k,
            lower_bound=lower_bound,
            ngram_range=ngram_range,
            min_df=min_df,
            id_col=id_col,
            name_col=name_col,
            address_col=address_col,
        )
        all_results.append(part_res)

    if not all_results:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "similarity_score",
                "rank",
                "country",
            ]
        )

    return pd.concat(all_results, ignore_index=True)


def extract_ground_truth_pairs(
    ground_truth_df: pd.DataFrame,
    s1_id_col: str = "source1_entity_id",
    matches_col: str = "matched_entity_ids",
) -> Set[Tuple[str, str]]:
    """Normalizes ground-truth data into a set of (source1_id, candidate_id) true-positive tuples.

    Handles comma-separated match strings (e.g. 'S2-01,S3-04') as well as already
    unpivoted pairwise DataFrames. Filters out empty strings and NaN singletons.
    """
    true_pairs: Set[Tuple[str, str]] = set()

    for _, row in ground_truth_df.iterrows():
        s1 = str(row[s1_id_col]).strip()
        raw_matches = row.get(matches_col, "")
        if pd.isna(raw_matches):
            continue

        raw_str = str(raw_matches).strip()
        if not raw_str or raw_str.lower() == "nan":
            continue

        for m in raw_str.split(","):
            m_clean = m.strip()
            if m_clean:
                true_pairs.add((s1, m_clean))

    return true_pairs


def pair_completeness(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    k: Optional[int] = None,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    gt_s1_col: str = "source1_entity_id",
    gt_matches_col: str = "matched_entity_ids",
) -> float:
    """Computes Pair Completeness (blocking recall) against labeled ground truth.

    Pair Completeness measures the fraction of true equivalent pairs retained
    after the blocking stage:

        Pair Completeness = |Candidates ∩ True Matches| / |True Matches|

    Singletons (entities with 0 matches) are excluded from the denominator
    because they do not form positive candidate links.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate pairs output from blocking, containing `s1_col` and `cand_col`.
        If `k` is specified and a `rank` column exists, filtered by `rank <= k`.
    ground_truth_df : pd.DataFrame
        Ground-truth labels (with `gt_s1_col` and `gt_matches_col`).
    k : Optional[int]
        If provided, limits evaluation to the top-k candidate pairs per query entity.
    s1_col : str
        Reference query entity ID column name in `candidates_df`.
    cand_col : str
        Candidate entity ID column name in `candidates_df`.
    gt_s1_col : str
        Reference query entity ID column name in `ground_truth_df`.
    gt_matches_col : str
        Matched entity IDs column name in `ground_truth_df`.

    Returns
    -------
    float
        Pair Completeness score in [0.0, 1.0]. Returns 1.0 if ground truth has no positive links.
    """
    true_pairs = extract_ground_truth_pairs(
        ground_truth_df, s1_id_col=gt_s1_col, matches_col=gt_matches_col
    )
    if not true_pairs:
        return 1.0

    eval_df = candidates_df
    if k is not None:
        if "rank" in eval_df.columns:
            eval_df = eval_df[eval_df["rank"] <= k]
        else:
            eval_df = eval_df.groupby(s1_col).head(k)

    retrieved_pairs = set(zip(eval_df[s1_col].astype(str), eval_df[cand_col].astype(str)))
    matched_hits = len(true_pairs.intersection(retrieved_pairs))

    return matched_hits / len(true_pairs)


def compute_macro_f05(
    predictions_by_s1: Dict[str, List[Tuple[str, float]]],
    ground_truth_by_s1: Dict[str, Set[str]],
    all_s1_ids: Iterable[str],
    k: int = 30,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Computes exact official competition metrics: Macro F_0.5, Macro Precision, Macro Recall,
    and Micro Pair Completeness.

    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
        - True empty & Pred empty: Precision=1.0, Recall=1.0, F0.5=1.0
        - True empty & Pred non-empty: Precision=0.0, Recall=0.0, F0.5=0.0
    """
    total_true_links = 0
    total_retained_hits = 0

    f05_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []

    for s1_id in all_s1_ids:
        true_set = ground_truth_by_s1.get(s1_id, set())
        cands_with_scores = predictions_by_s1.get(s1_id, [])

        filtered = [cid for cid, sc in cands_with_scores[:k] if sc >= threshold]
        pred_set = set(filtered)

        total_true_links += len(true_set)
        hits = len(true_set.intersection(pred_set))
        total_retained_hits += hits

        if len(true_set) == 0:
            if len(pred_set) == 0:
                p, r, f05 = 1.0, 1.0, 1.0
            else:
                p, r, f05 = 0.0, 0.0, 0.0
        else:
            if len(pred_set) == 0:
                p, r, f05 = 0.0, 0.0, 0.0
            else:
                p = hits / len(pred_set)
                r = hits / len(true_set)
                denom = 0.25 * p + r
                f05 = (1.25 * p * r) / denom if denom > 0 else 0.0

        f05_list.append(f05)
        prec_list.append(p)
        rec_list.append(r)

    pair_comp = (total_retained_hits / total_true_links) if total_true_links > 0 else 1.0

    return {
        "macro_f05": float(np.mean(f05_list)),
        "macro_precision": float(np.mean(prec_list)),
        "macro_recall": float(np.mean(rec_list)),
        "pair_completeness": float(pair_comp),
        "total_true_links": total_true_links,
        "total_hits": total_retained_hits,
    }


# ==============================================================================
# Runnable Benchmark & Accuracy Evaluation
# ==============================================================================
if __name__ == "__main__":
    import os
    import time
    from collections import defaultdict

    real_data_path = "student_resource/dataset/train/train_source1.tsv"
    if os.path.exists(real_data_path):
        print("=" * 95)
        print("RUNNING TF-IDF BLOCKER COMPLETE ACCURACY & RECALL BENCHMARK ON REAL DATA")
        print("Zero simulation: Evaluating real Source 1 queries with Ground Truth matches & singletons.")
        print("=" * 95)

        SAMPLE_SIZE = 5000
        print(f"\n[Step 1] Loading {SAMPLE_SIZE:,} real S1 queries from {real_data_path}...")
        s1_df = pd.read_csv(real_data_path, sep="\t", nrows=SAMPLE_SIZE)
        s1_ids = s1_df["entity_id"].tolist()
        s1_id_set = set(s1_ids)

        print("[Step 2] Loading Ground Truth links and identifying singletons...")
        gt_map = {}
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
        print(f"  Loaded Ground Truth for {len(gt_map):,} S1 entities:")
        print(f"    Entities with matches: {len(gt_map) - singletons:,}")
        print(f"    Singletons (0 matches): {singletons:,} ({singletons/len(gt_map):.2%})")
        print(f"    Total true positive links: {total_true_matches:,}")

        print("\n[Step 3] Assembling candidate pool (all true candidates + 100,000 real distractors)...")
        cand_records = []
        EXTRA_DISTRACTORS = 100000
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

                    if eid in needed_cand_ids:
                        cand_records.append({
                            "entity_id": eid,
                            "business_name": name,
                            "business_address": addr,
                            "country": country,
                        })
                    elif distractors_loaded < EXTRA_DISTRACTORS:
                        cand_records.append({
                            "entity_id": eid,
                            "business_name": name,
                            "business_address": addr,
                            "country": country,
                        })
                        distractors_loaded += 1

        cand_df = pd.DataFrame(cand_records)
        print(f"  Candidate pool assembled: {len(cand_df):,} total records")
        print("  Candidate breakdown by country:")
        print(cand_df["country"].value_counts())

        print("\n[Step 4] Running run_partitioned_blocking() (k=100, lower_bound=0.01)...")
        t0 = time.time()
        candidates_df = run_partitioned_blocking(
            reference_df=s1_df,
            candidate_df=cand_df,
            k=100,
            lower_bound=0.01,
            ngram_range=(3, 5),
            min_df=2,
        )
        print(f"  Blocking completed in {time.time()-t0:.2f}s! Retrieved {len(candidates_df):,} pairs.")

        pred_by_s1 = defaultdict(list)
        for _, row in candidates_df.sort_values(["source1_entity_id", "rank"]).iterrows():
            pred_by_s1[str(row["source1_entity_id"])].append(
                (str(row["candidate_entity_id"]), float(row["similarity_score"]))
            )

        s1_us_ids = s1_df[s1_df["country"] == "US"]["entity_id"].tolist()
        s1_in_ids = s1_df[s1_df["country"] == "India"]["entity_id"].tolist()
        k_list = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

        print("\n" + "=" * 95)
        print(f"{'k':>4} | {'US Recall':>10} | {'India Recall':>12} | {'Overall Recall':>15} | {'Macro Precision':>16} | {'Macro F_0.5':>12}")
        print("=" * 95)

        for k in k_list:
            m_us = compute_macro_f05(pred_by_s1, gt_map, s1_us_ids, k=k)
            m_in = compute_macro_f05(pred_by_s1, gt_map, s1_in_ids, k=k)
            m_all = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=k)
            print(
                f"{k:4d} | {m_us['pair_completeness']*100:9.2f}% | "
                f"{m_in['pair_completeness']*100:11.2f}% | "
                f"{m_all['pair_completeness']*100:14.2f}% | "
                f"{m_all['macro_precision']*100:15.2f}% | "
                f"{m_all['macro_f05']:12.4f}"
            )
        print("=" * 95)

        print("\n[Step 5] Similarity Score Thresholding Sweep (at k=50)")
        print("Shows how downstream classification/thresholding maximizes the competition Macro F_0.5 metric:")
        print("-" * 75)
        print(f"{'Threshold':>10} | {'Pair Recall':>12} | {'Macro Precision':>16} | {'Macro F_0.5':>12}")
        print("-" * 75)
        for tau in [0.01, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
            m_thresh = compute_macro_f05(pred_by_s1, gt_map, s1_ids, k=50, threshold=tau)
            print(
                f"{tau:10.2f} | {m_thresh['pair_completeness']*100:11.2f}% | "
                f"{m_thresh['macro_precision']*100:15.2f}% | "
                f"{m_thresh['macro_f05']:12.4f}"
            )
        print("-" * 75)
        print("\nBenchmark Finished Successfully.")
    else:
        print("Dataset not found at default path. Please provide training files.")

