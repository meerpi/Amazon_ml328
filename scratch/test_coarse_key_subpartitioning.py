"""Comprehensive Benchmark & A/B Test: Sub-Partition Blocking for India Entity Resolution.

Compares:
1. Baseline: Global Country-Level TF-IDF Blocking (current approach).
2. Strategy A: Name-only Coarse Sub-partitioning (First 2 chars of normalized name).
3. Strategy B: Disjunctive Name Coarse Sub-partitioning (Prefix of Word 1 OR Word 2).
4. Strategy C: Address Coarse Sub-partitioning (City / State Token extraction).
5. Strategy D: Combined Name + Address Coarse Key (First Char of Name + State/City).
6. Strategy E: Hybrid Disjunctive Scheme (Name Prefix OR Address City Token).

Measures exact Recall (Pair Completeness), Macro Precision, Macro F0.5, Wall-clock Time, and Block Statistics.
"""

import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn

sys.path.insert(0, ".")
from src.tfidf_blocker import romanize_text, compute_macro_f05

# -----------------------------------------------------------------------------
# City & State Dictionaries for India
# -----------------------------------------------------------------------------
INDIAN_STATES = {
    "andhra pradesh": "AP", "andhra": "AP", "ap": "AP",
    "arunachal pradesh": "AR",
    "assam": "AS", "bihar": "BR", "chhattisgarh": "CG", "goa": "GA",
    "gujarat": "GJ", "gujrat": "GJ", "haryana": "HR",
    "himachal pradesh": "HP", "jharkhand": "JH",
    "karnataka": "KA", "kerala": "KL",
    "madhya pradesh": "MP", "maharashtra": "MH",
    "manipur": "MN", "meghalaya": "ML", "mizoram": "MZ", "nagaland": "NL",
    "odisha": "OD", "orissa": "OD", "punjab": "PB",
    "rajasthan": "RJ", "sikkim": "SK",
    "tamil nadu": "TN", "tamilnadu": "TN",
    "telangana": "TS", "telengana": "TS", "tg": "TS",
    "tripura": "TR", "uttar pradesh": "UP", "uttarakhand": "UK",
    "west bengal": "WB", "delhi": "DL", "new delhi": "DL", "chandigarh": "CH",
}

TOP_INDIAN_CITIES = [
    "mumbai", "delhi", "bengaluru", "bangalore", "hyderabad", "ahmedabad",
    "chennai", "kolkata", "surat", "pune", "jaipur", "lucknow", "kanpur",
    "nagpur", "indore", "thane", "bhopal", "visakhapatnam", "pimpri",
    "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik",
    "faridabad", "meerut", "rajkot", "varanasi", "srinagar", "aurangabad",
    "dhanbad", "amritsar", "navi mumbai", "allahabad", "prayagraj",
    "ranchi", "howrah", "coimbatore", "jabalpur", "gwalior", "vijayawada",
    "jodhpur", "madurai", "raipur", "kota", "guwahati", "chandigarh",
    "solapur", "hubli", "dharwad", "bareilly", "moradabad", "mysore",
    "mysuru", "gurgaon", "gurugram", "aligarh", "jalandhar", "tiruchirappalli",
    "bhubaneswar", "salem", "warangal", "mira bhayandar", "thiruvananthapuram",
    "bhiwandi", "saharanpur", "guntur", "amravati", "bikaner", "noida",
    "jamshedpur", "bhilai", "cuttack", "firozabad", "kochi", "cochin",
    "bhavnagar", "dehradun", "durgapur", "asansol", "nanded", "kolhapur",
    "ajmer", "gulbarga", "jamnagar", "ujjain", "loni", "siliguri", "jhansi",
    "ulhasnagar", "jammu", "sangli", "mangalore", "mangaluru", "erode",
    "belgaum", "belagavi", "ambattur", "tirunelveli", "malegaon", "gaya"
]

TOP_CITIES_SET = set(TOP_INDIAN_CITIES)

NAME_PREFIX_STOPWORDS = {
    "the", "m/s", "ms", "shri", "shree", "sri", "dr", "hotel", "om",
    "jai", "new", "all", "national", "indian", "india", "royal", "golden", "sai"
}


def extract_city(addr: str) -> Optional[str]:
    """Extracts known top Indian city from normalized address."""
    addr_lower = romanize_text(addr).lower()
    tokens = set(re.findall(r"\b[a-z0-9]+\b", addr_lower))
    
    # Check multi-word cities first
    if "navi mumbai" in addr_lower:
        return "mumbai"
    if "new delhi" in addr_lower:
        return "delhi"
        
    common = tokens.intersection(TOP_CITIES_SET)
    if common:
        # Standardize synonyms
        c = next(iter(common))
        if c == "bangalore": return "bengaluru"
        if c == "prayagraj": return "allahabad"
        if c == "mysuru": return "mysore"
        if c == "gurugram": return "gurgaon"
        if c == "mangaluru": return "mangalore"
        if c == "cochin": return "kochi"
        return c
    return None


def extract_state_code(addr: str) -> Optional[str]:
    """Extracts 2-letter state code."""
    addr_lower = romanize_text(addr).lower()
    tokens = set(re.findall(r"\b[a-z&]+\b", addr_lower))
    for state_name, code in INDIAN_STATES.items():
        if state_name in tokens or f" {state_name} " in f" {addr_lower} ":
            return code
    return None


def extract_name_tokens(name: str) -> List[str]:
    """Romanized tokens for business name."""
    rom = romanize_text(name).lower()
    return re.findall(r"\b[a-z0-9]+\b", rom)


def get_sig_token(tokens: List[str]) -> str:
    if not tokens:
        return ""
    for t in tokens:
        if t not in NAME_PREFIX_STOPWORDS and len(t) >= 2:
            return t
    return tokens[0]


# -----------------------------------------------------------------------------
# Key Extractors for Testing
# -----------------------------------------------------------------------------
def get_coarse_key(name: str, addr: str, strategy: str) -> str:
    """Computes a single coarse partition key."""
    tokens = extract_name_tokens(name)
    sig = get_sig_token(tokens)
    
    if strategy == "name_pref2":
        # First 2 chars of significant token
        return sig[:2] if len(sig) >= 2 else (sig[:1] if sig else "UNK")
        
    elif strategy == "name_pref3":
        # First 3 chars of significant token
        return sig[:3] if len(sig) >= 3 else (sig if sig else "UNK")

    elif strategy == "city":
        # City token, fallback to state, fallback to UNK
        city = extract_city(addr)
        if city: return f"c_{city}"
        state = extract_state_code(addr)
        if state: return f"s_{state}"
        return "loc_UNK"

    elif strategy == "name1_state":
        # Combined: 1st char of name + State
        n1 = sig[:1] if sig else "u"
        st = extract_state_code(addr) or "unk"
        return f"{n1}_{st}"

    elif strategy == "name2_city":
        # Combined: 2 chars of name + City/State
        n2 = sig[:2] if len(sig) >= 2 else (sig[:1] if sig else "u")
        city = extract_city(addr)
        if city:
            return f"{n2}_{city}"
        st = extract_state_code(addr)
        if st:
            return f"{n2}_{st}"
        return f"{n2}_unk"

    raise ValueError(f"Unknown strategy: {strategy}")


def get_multi_keys(name: str, addr: str, strategy: str) -> Set[str]:
    """Computes a set of coarse keys for disjunctive (multi-index) blocking."""
    tokens = extract_name_tokens(name)
    sig = get_sig_token(tokens)
    sec = tokens[1] if len(tokens) > 1 else ""
    city = extract_city(addr)
    st = extract_state_code(addr)
    
    keys = set()
    if strategy == "disj_w1_w2":
        # Prefix of Word 1 OR Prefix of Word 2
        if len(sig) >= 2: keys.add(f"w_{sig[:2]}")
        if len(sec) >= 2: keys.add(f"w_{sec[:2]}")
        if not keys: keys.add("w_unk")
        
    elif strategy == "disj_name2_or_city":
        # First 2 chars of name OR City token
        if len(sig) >= 2: keys.add(f"n_{sig[:2]}")
        if city: keys.add(f"c_{city}")
        if not keys: keys.add("unk")

    return keys


# -----------------------------------------------------------------------------
# Fast TF-IDF Matching Engine (Within a Block)
# -----------------------------------------------------------------------------
def run_tfidf_matching(
    ref_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    k: int = 50,
    lower_bound: float = 0.05,
) -> pd.DataFrame:
    """Runs vectorized TF-IDF top-k cosine similarity on a single partition."""
    if ref_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "similarity_score", "rank"])

    ref_texts = (ref_df["business_name"].fillna("") + " " + ref_df["business_address"].fillna("")).tolist()
    cand_texts = (cand_df["business_name"].fillna("") + " " + cand_df["business_address"].fillna("")).tolist()

    # Pre-romanize
    ref_rom = [romanize_text(t) for t in ref_texts]
    cand_rom = [romanize_text(t) for t in cand_texts]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        dtype=np.float32,
        norm="l2",
        min_df=1,
        sublinear_tf=True,
    )
    vec.fit(ref_rom + cand_rom)
    mat_ref = vec.transform(ref_rom)
    mat_cand = vec.transform(cand_rom)

    sim = awesome_cossim_topn(mat_ref, mat_cand.T, ntop=k, lower_bound=lower_bound)

    counts = np.diff(sim.indptr)
    if len(counts) == 0 or sim.nnz == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "similarity_score", "rank"])

    rows = np.repeat(np.arange(sim.shape[0]), counts)
    cols = sim.indices
    data = sim.data
    ranks = np.concatenate([np.arange(1, c + 1) for c in counts])

    ref_ids = ref_df["entity_id"].to_numpy()
    cand_ids = cand_df["entity_id"].to_numpy()

    return pd.DataFrame({
        "source1_entity_id": ref_ids[rows],
        "candidate_entity_id": cand_ids[cols],
        "similarity_score": data,
        "rank": ranks,
    })


# -----------------------------------------------------------------------------
# Main Experiment Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 95)
    print("SUB-PARTITION BLOCKING BENCHMARK: INDIA ENTITY RESOLUTION")
    print("Evaluating Speedup, Recall (Pair Completeness), and Macro F0.5")
    print("=" * 95)

    # 1. Load real India queries & ground truth
    N_TEST_QUERIES = 2500
    N_DISTRACTORS = 150000

    print(f"\n[Step 1] Loading {N_TEST_QUERIES:,} real India queries from train_source1.tsv...")
    s1_df = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t")
    s1_india = s1_df[s1_df["country"] == "India"].head(N_TEST_QUERIES).reset_index(drop=True).copy()
    s1_ids = s1_india["entity_id"].tolist()
    s1_id_set = set(s1_ids)

    print("[Step 2] Loading Ground Truth matches...")
    gt_map = {}
    needed_cand_ids = set()
    singletons = 0
    with open("student_resource/dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in s1_id_set:
                raw = parts[1] if len(parts) > 1 else ""
                if raw:
                    cands = set(c.strip() for c in raw.split(",") if c.strip())
                    gt_map[s1_id] = cands
                    needed_cand_ids.update(cands)
                else:
                    gt_map[s1_id] = set()
                    singletons += 1
            if len(gt_map) >= len(s1_ids):
                break

    for s1 in s1_ids:
        if s1 not in gt_map:
            gt_map[s1] = set()
            singletons += 1

    total_true = sum(len(c) for c in gt_map.values())
    print(f"  Loaded GT for {len(gt_map):,} queries: {total_true:,} true positive candidate links.")

    print(f"\n[Step 3] Assembling India candidate pool (all {len(needed_cand_ids):,} true candidates + {N_DISTRACTORS:,} distractors)...")
    cand_records = []
    distractors_loaded = 0

    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{fname}"
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\n").split("\t")
                if len(parts) < 4 or parts[3] != "India":
                    continue
                cid = parts[0]
                name = parts[1]
                addr = parts[2]
                
                if cid in needed_cand_ids:
                    cand_records.append({"entity_id": cid, "business_name": name, "business_address": addr, "country": "India"})
                elif distractors_loaded < N_DISTRACTORS:
                    cand_records.append({"entity_id": cid, "business_name": name, "business_address": addr, "country": "India"})
                    distractors_loaded += 1

    cand_df = pd.DataFrame(cand_records).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    print(f"  Total candidate pool assembled: {len(cand_df):,} records.")

    # -------------------------------------------------------------------------
    # Baseline: Country-Level Global TF-IDF (Current Pipeline)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING BENCHMARKS...")
    print("=" * 80)

    results = []

    # 1. Baseline
    print("\n--- Running Baseline: Global Country-Level TF-IDF (No Sub-Partitioning) ---")
    t0 = time.time()
    base_preds_df = run_tfidf_matching(s1_india, cand_df, k=50, lower_bound=0.05)
    base_time = time.time() - t0
    
    base_pred_dict = defaultdict(list)
    for _, r in base_preds_df.iterrows():
        base_pred_dict[str(r["source1_entity_id"])].append((str(r["candidate_entity_id"]), float(r["similarity_score"])))
    
    m_base = compute_macro_f05(base_pred_dict, gt_map, s1_ids, k=30, threshold=0.30)
    results.append({
        "Method": "1. Baseline (Global Country TF-IDF)",
        "Time (s)": base_time,
        "Speedup": 1.0,
        "Blocks": 1,
        "Pair Completeness (Recall)": m_base["pair_completeness"],
        "Macro Precision": m_base["macro_precision"],
        "Macro F0.5": m_base["macro_f05"],
    })
    print(f"  Completed in {base_time:.2f}s | Recall: {m_base['pair_completeness']*100:.2f}% | F0.5: {m_base['macro_f05']:.4f}")

    # -------------------------------------------------------------------------
    # Test Sub-partitioning Strategies
    # -------------------------------------------------------------------------
    strategies = [
        ("2. Name Prefix(2) Partitioning", "name_pref2", False),
        ("3. Name Prefix(3) Partitioning", "name_pref3", False),
        ("4. Combined Name(1) + State", "name1_state", False),
        ("5. Combined Name(2) + City/State", "name2_city", False),
        ("6. City/State Partitioning", "city", False),
        ("7. Disjunctive: Word1(2) OR Word2(2)", "disj_w1_w2", True),
        ("8. Disjunctive: Name(2) OR City", "disj_name2_or_city", True),
    ]

    for label, strat_name, is_disj in strategies:
        print(f"\n--- Running: {label} ---")
        t0 = time.time()
        
        if not is_disj:
            # Single key partitioning
            s1_keys = [get_coarse_key(str(r["business_name"]), str(r["business_address"]), strat_name) for _, r in s1_india.iterrows()]
            cand_keys = [get_coarse_key(str(r["business_name"]), str(r["business_address"]), strat_name) for _, r in cand_df.iterrows()]
            
            s1_with_key = s1_india.copy()
            s1_with_key["_block_key"] = s1_keys
            cand_with_key = cand_df.copy()
            cand_with_key["_block_key"] = cand_keys

            cand_by_key = {k: v for k, v in cand_with_key.groupby("_block_key")}
            
            sub_preds = []
            distinct_blocks = s1_with_key["_block_key"].nunique()

            for key, s1_sub in s1_with_key.groupby("_block_key"):
                cand_sub = cand_by_key.get(key)
                if cand_sub is not None and not cand_sub.empty:
                    df_sub = run_tfidf_matching(s1_sub, cand_sub, k=50, lower_bound=0.05)
                    sub_preds.append(df_sub)

            all_preds = pd.concat(sub_preds, ignore_index=True) if sub_preds else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "similarity_score", "rank"])

        else:
            # Disjunctive (multi-key) partitioning: iterate over distinct block keys
            # Build inverted index for candidates and queries
            cand_index = defaultdict(list)
            for idx, r in cand_df.iterrows():
                for k in get_multi_keys(str(r["business_name"]), str(r["business_address"]), strat_name):
                    cand_index[k].append(idx)

            s1_index = defaultdict(list)
            for idx, r in s1_india.iterrows():
                for k in get_multi_keys(str(r["business_name"]), str(r["business_address"]), strat_name):
                    s1_index[k].append(idx)

            distinct_blocks = len(cand_index)
            sub_preds = []

            for key, s1_indices in s1_index.items():
                cand_indices = cand_index.get(key)
                if cand_indices:
                    q_sub = s1_india.iloc[s1_indices]
                    cand_sub = cand_df.iloc[cand_indices]
                    df_sub = run_tfidf_matching(q_sub, cand_sub, k=50, lower_bound=0.05)
                    sub_preds.append(df_sub)

            if sub_preds:
                raw_preds = pd.concat(sub_preds, ignore_index=True)
                # Deduplicate by (source1_entity_id, candidate_entity_id) keeping highest similarity score
                all_preds = (
                    raw_preds.sort_values("similarity_score", ascending=False)
                    .drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
                    .reset_index(drop=True)
                )
            else:
                all_preds = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "similarity_score", "rank"])

        elapsed = time.time() - t0
        speedup = base_time / elapsed if elapsed > 0 else 0.0

        pred_dict = defaultdict(list)
        for _, r in all_preds.iterrows():
            pred_dict[str(r["source1_entity_id"])].append((str(r["candidate_entity_id"]), float(r["similarity_score"])))

        m = compute_macro_f05(pred_dict, gt_map, s1_ids, k=30, threshold=0.30)
        results.append({
            "Method": label,
            "Time (s)": elapsed,
            "Speedup": speedup,
            "Blocks": distinct_blocks,
            "Pair Completeness (Recall)": m["pair_completeness"],
            "Macro Precision": m["macro_precision"],
            "Macro F0.5": m["macro_f05"],
        })
        print(f"  Completed in {elapsed:.2f}s (Speedup: {speedup:.2f}x) | Blocks: {distinct_blocks:,} | Recall: {m['pair_completeness']*100:.2f}% | F0.5: {m['macro_f05']:.4f}")

    # -------------------------------------------------------------------------
    # Final Summary Table
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 105)
    print("FINAL RIGOROUS COMPARISON TABLE (N=2,500 India Queries vs. 150,000 Real Candidates)")
    print("=" * 105)
    print(f"{'Blocking Strategy':<38} | {'Time (s)':<9} | {'Speedup':<8} | {'Blocks':<8} | {'Recall':<8} | {'Precision':<10} | {'Macro F0.5':<10}")
    print("-" * 105)
    for _, r in summary_df.iterrows():
        print(
            f"{r['Method']:<38} | "
            f"{r['Time (s)']:9.2f} | "
            f"{r['Speedup']:7.2f}x | "
            f"{r['Blocks']:8,d} | "
            f"{r['Pair Completeness (Recall)']*100:7.2f}% | "
            f"{r['Macro Precision']*100:9.2f}% | "
            f"{r['Macro F0.5']:10.4f}"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()
