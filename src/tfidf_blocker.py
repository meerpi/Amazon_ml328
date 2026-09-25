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
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn


def build_text_representation(
    df: pd.DataFrame,
    name_col: str = "business_name",
    address_col: str = "business_address",
) -> List[str]:
    """Combines business name and address into a single normalized text representation.

    Ensures missing values are replaced with empty strings to prevent string-coercion
    artifacts (e.g. float 'nan').
    """
    names = df[name_col].fillna("").astype(str)
    addresses = df[address_col].fillna("").astype(str)
    return (names + " " + addresses).tolist()


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


# ==============================================================================
# Self-contained runnable sanity check
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING STANDALONE SANITY CHECK: SPARKLY-STYLE TF-IDF BLOCKER")
    print("=" * 80)

    # 1. Tiny synthetic reference dataset (Source 1)
    ref_records = pd.DataFrame(
        [
            {
                "entity_id": "S1-01",
                "business_name": "orelee barbershop",
                "business_address": "1795 westchester drive high point nc",
                "country": "US",
            },
            {
                "entity_id": "S1-02",
                "business_name": "prime money",
                "business_address": "17560 ellis road tahlequah ok",
                "country": "US",
            },
            {
                "entity_id": "S1-03",
                "business_name": "lakshmi media private limited",
                "business_address": "154 bhugaon pune maharashtra",
                "country": "India",
            },
            {
                "entity_id": "S1-04",
                "business_name": "societe generale",
                "business_address": "29 boulevard haussmann paris",
                "country": "France",
            },
            {
                "entity_id": "S1-05",
                "business_name": "delta logistics",
                "business_address": "100 airport road atlanta ga",
                "country": "US",
            },
        ]
    )

    # 2. Synthetic candidates dataset (Source 2 and Source 3 pooled)
    cand_records = pd.DataFrame(
        [
            # True match for S1-01 (US) with abbreviation & missing state
            {
                "entity_id": "S2-01",
                "business_name": "orelees barber shop",
                "business_address": "1795 westchester dr high point",
                "country": "US",
            },
            # True match for S1-02 (US) with typo & suffix variant
            {
                "entity_id": "S3-02",
                "business_name": "prime mony inc",
                "business_address": "17560 elis rd tahlequah",
                "country": "US",
            },
            # True match for S1-03 (India) romanized from Devanagari
            {
                "entity_id": "S2-03",
                "business_name": "lakssmii miiddiyaa praaivett limittedd",
                "business_address": "154 bhugaon pune maharashtra",
                "country": "India",
            },
            # True match for S1-04 (France) with suffix variant & street abbrev
            {
                "entity_id": "S3-04",
                "business_name": "societe generale sa",
                "business_address": "29 bd haussmann paris france",
                "country": "France",
            },
            # Same name as S1-05, but in India -> MUST NOT MATCH across country!
            {
                "entity_id": "S2-05",
                "business_name": "delta logistics",
                "business_address": "100 airport road chennai",
                "country": "India",
            },
            # Distractor candidate in US
            {
                "entity_id": "S3-06",
                "business_name": "unrelated cafe",
                "business_address": "45 main street boston ma",
                "country": "US",
            },
        ]
    )

    # 3. Labeled Ground Truth Mapping
    ground_truth = pd.DataFrame(
        [
            {"source1_entity_id": "S1-01", "matched_entity_ids": "S2-01"},
            {"source1_entity_id": "S1-02", "matched_entity_ids": "S3-02"},
            {"source1_entity_id": "S1-03", "matched_entity_ids": "S2-03"},
            {"source1_entity_id": "S1-04", "matched_entity_ids": "S3-04"},
            {"source1_entity_id": "S1-05", "matched_entity_ids": ""},  # True Singleton
        ]
    )

    print("\nRunning partitioned blocking (k=3, lower_bound=0.1, char_wb=(3,5))...\n")
    candidates = run_partitioned_blocking(
        reference_df=ref_records,
        candidate_df=cand_records,
        k=3,
        lower_bound=0.1,
        ngram_range=(3, 5),
        min_df=1,  # min_df=1 for tiny synthetic test
    )

    print("BLOCKING CANDIDATE PAIRS RETRIEVED:")
    print("-" * 80)
    print(candidates.to_string(index=False))
    print("-" * 80)

    # Sanity Check 1: Cross-country non-match verification
    cross_country_leak = candidates[
        (candidates["source1_entity_id"] == "S1-05")
        & (candidates["candidate_entity_id"] == "S2-05")
    ]
    assert cross_country_leak.empty, "FAIL: Cross-country candidate leak detected!"
    print("\n[PASS] Sanity Check 1: S1-05 (US) did NOT match S2-05 (India).")

    # Sanity Check 2: Pair Completeness evaluation across k
    print("\nPAIR COMPLETENESS SWEEP:")
    for test_k in [1, 2, 3]:
        pc = pair_completeness(candidates, ground_truth, k=test_k)
        print(f"  Pair Completeness @ k={test_k}: {pc:.4f} ({pc * 100:.1f}%)")

    pc_top1 = pair_completeness(candidates, ground_truth, k=1)
    assert pc_top1 == 1.0, f"FAIL: Expected Pair Completeness 1.0, got {pc_top1}"
    print("\n[PASS] Sanity Check 2: 100% of true ground-truth pairs captured at k=1.")
    print("=" * 80)
