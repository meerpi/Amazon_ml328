"""Research and Empirical Evaluation of Coarse Blocking Keys for India Entity Resolution.

Tests:
1. Exact Recall / Pair Completeness on thousands of real Ground-Truth pairs in India.
2. Missingness analysis (e.g. how many addresses lack PIN code or State).
3. Blocking key distribution, candidate reduction ratio, and bucket skew.
4. Composite keys combining Business Name and Address components.
5. Disjunctive (Multi-Key / Multi-Index) schemes to retain high recall.
"""

import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from src.tfidf_blocker import romanize_text

# -------------------------------------------------------------------------
# Indian Address Dictionaries & Normalization Helpers
# -------------------------------------------------------------------------
INDIAN_STATES = {
    "andhra pradesh": "AP", "andhra": "AP", "ap": "AP",
    "arunachal pradesh": "AR", "arunachal": "AR",
    "assam": "AS",
    "bihar": "BR",
    "chhattisgarh": "CG", "chattisgarh": "CG",
    "goa": "GA",
    "gujarat": "GJ", "gujrat": "GJ",
    "haryana": "HR",
    "himachal pradesh": "HP", "himachal": "HP",
    "jharkhand": "JH",
    "karnataka": "KA",
    "kerala": "KL",
    "madhya pradesh": "MP",
    "maharashtra": "MH",
    "manipur": "MN",
    "meghalaya": "ML",
    "mizoram": "MZ",
    "nagaland": "NL",
    "odisha": "OD", "orissa": "OD",
    "punjab": "PB",
    "rajasthan": "RJ",
    "sikkim": "SK",
    "tamil nadu": "TN", "tamilnadu": "TN",
    "telangana": "TS", "telengana": "TS",
    "tripura": "TR",
    "uttar pradesh": "UP", "uttarpradesh": "UP",
    "uttarakhand": "UK", "uttaranchal": "UK",
    "west bengal": "WB", "bengal": "WB",
    "delhi": "DL", "new delhi": "DL",
    "chandigarh": "CH",
    "jammu and kashmir": "JK", "jammu & kashmir": "JK", "jammu": "JK", "kashmir": "JK",
    "ladakh": "LA",
    "puducherry": "PY", "pondicherry": "PY",
}

# Common Indian business stopwords / prefixes that create false mismatches in name prefix blocking
NAME_PREFIX_STOPWORDS = {
    "the", "m/s", "ms", "shri", "shree", "sri", "dr", "dr.", "prof", "hotel", "om",
    "jai", "new", "all", "national", "indian", "india", "royal", "golden", "sai"
}

PINCODE_REGEX = re.compile(r"\b([1-9][0-9]{5})\b")


def extract_pincode(addr: str) -> Optional[str]:
    """Extracts 6-digit Indian PIN code."""
    m = PINCODE_REGEX.search(addr)
    return m.group(1) if m else None


def extract_state(addr: str) -> Optional[str]:
    """Extracts standardized 2-letter state code from address."""
    addr_lower = addr.lower()
    # Check word tokens / phrases
    # Search for multi-word states first, then single words
    tokens = re.findall(r"\b[a-z&]+\b", addr_lower)
    token_str = " " + " ".join(tokens) + " "
    
    for state_name, code in INDIAN_STATES.items():
        if f" {state_name} " in token_str:
            return code
    return None


def clean_name_tokens(name: str) -> List[str]:
    """Romanizes, lowercases, removes punctuation, and splits into significant words."""
    rom = romanize_text(name).lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", rom)
    return tokens


def get_first_significant_token(tokens: List[str]) -> str:
    """Returns first token skipping leading stopwords if alternatives exist."""
    if not tokens:
        return ""
    for t in tokens:
        if t not in NAME_PREFIX_STOPWORDS and len(t) >= 2:
            return t
    return tokens[0]


def soundex(token: str) -> str:
    """Computes basic Soundex code for a word."""
    if not token or not token[0].isalpha():
        return ""
    token = token.upper()
    first_letter = token[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }
    encoded = [first_letter]
    prev = mapping.get(first_letter, "")
    for char in token[1:]:
        code = mapping.get(char, "")
        if code and code != prev:
            encoded.append(code)
            prev = code
        elif not code and char not in "HW":
            prev = ""
        if len(encoded) == 4:
            break
    while len(encoded) < 4:
        encoded.append("0")
    return "".join(encoded)


# -------------------------------------------------------------------------
# Candidate Blocking Key Functions
# -------------------------------------------------------------------------
def generate_blocking_keys(name: str, addr: str) -> Dict[str, str]:
    """Generates all single-key candidates for an entity."""
    tokens = clean_name_tokens(name)
    raw_norm = re.sub(r"[^a-z0-9]", "", romanize_text(name).lower())
    first_sig = get_first_significant_token(tokens)
    second_sig = tokens[1] if len(tokens) > 1 else ""
    
    pin = extract_pincode(addr)
    state = extract_state(addr)
    
    keys = {}
    
    # 1. Raw Character Prefixes
    keys["raw_prefix_1"] = raw_norm[:1] if len(raw_norm) >= 1 else ""
    keys["raw_prefix_2"] = raw_norm[:2] if len(raw_norm) >= 2 else ""
    keys["raw_prefix_3"] = raw_norm[:3] if len(raw_norm) >= 3 else ""
    
    # 2. Significant Token Character Prefixes (Stopword-aware)
    keys["clean_sig_prefix_1"] = first_sig[:1] if len(first_sig) >= 1 else ""
    keys["clean_sig_prefix_2"] = first_sig[:2] if len(first_sig) >= 2 else ""
    keys["clean_sig_prefix_3"] = first_sig[:3] if len(first_sig) >= 3 else ""
    
    # 3. Soundex of first significant word
    keys["name_soundex"] = soundex(first_sig)
    
    # 4. Address Keys
    keys["addr_pincode_6"] = pin if pin else ""
    keys["addr_pincode_circle_2"] = pin[:2] if pin else ""
    keys["addr_pincode_zone_1"] = pin[:1] if pin else ""
    keys["addr_state"] = state if state else ""
    
    # 5. Composite Keys: Name + Address
    # Name Prefix + State
    keys["comp_sig1_state"] = (first_sig[:1] + "_" + state) if (first_sig and state) else ""
    keys["comp_sig2_state"] = (first_sig[:2] + "_" + state) if (len(first_sig) >= 2 and state) else ""
    
    # Name Prefix + PIN Zone (1-digit)
    keys["comp_sig1_pinzone"] = (first_sig[:1] + "_z" + pin[:1]) if (first_sig and pin) else ""
    keys["comp_sig2_pinzone"] = (first_sig[:2] + "_z" + pin[:1]) if (len(first_sig) >= 2 and pin) else ""
    
    # Name Prefix + PIN Circle (2-digit)
    keys["comp_sig1_pincircle"] = (first_sig[:1] + "_c" + pin[:2]) if (first_sig and pin) else ""
    
    # Soundex + State
    keys["comp_soundex_state"] = (soundex(first_sig) + "_" + state) if (first_sig and state) else ""
    
    # Extra helper tokens for disjunctive sets
    keys["_second_sig_prefix_2"] = second_sig[:2] if len(second_sig) >= 2 else ""
    
    return keys


def generate_disjunctive_keys(name: str, addr: str) -> Dict[str, Set[str]]:
    """Generates sets of keys for multi-index / disjunctive blocking schemas.
    An entity matches if ANY of its keys match a candidate's keys!
    """
    single = generate_blocking_keys(name, addr)
    sig1 = single["clean_sig_prefix_1"]
    sig2 = single["clean_sig_prefix_2"]
    sig3 = single["clean_sig_prefix_3"]
    sec2 = single["_second_sig_prefix_2"]
    pin2 = single["addr_pincode_circle_2"]
    state = single["addr_state"]
    sndx = single["name_soundex"]
    
    disj = {}
    
    # Disjunctive Schema 1: First word prefix(2) OR Second word prefix(2)
    s1 = set()
    if sig2: s1.add(f"w1_{sig2}")
    if sec2: s1.add(f"w2_{sec2}")
    disj["disj_w1_2_OR_w2_2"] = s1
    
    # Disjunctive Schema 2: First word prefix(2) OR PIN circle(2)
    s2 = set()
    if sig2: s2.add(f"nm_{sig2}")
    if pin2: s2.add(f"pin_{pin2}")
    disj["disj_name2_OR_pincircle2"] = s2
    
    # Disjunctive Schema 3: (First word prefix(2) + State) OR (First word prefix(2) + PIN circle(2)) OR Soundex
    s3 = set()
    if sig2 and state: s3.add(f"ns_{sig2}_{state}")
    if sig2 and pin2: s3.add(f"np_{sig2}_{pin2}")
    if sndx and state: s3.add(f"ss_{sndx}_{state}")
    disj["disj_composite_robust"] = s3

    # Disjunctive Schema 4: First word prefix(2) OR First word Soundex
    s4 = set()
    if sig2: s4.add(f"pref_{sig2}")
    if sndx: s4.add(f"sndx_{sndx}")
    disj["disj_name2_OR_soundex"] = s4
    
    # Disjunctive Schema 5: First word prefix(3) OR (First word prefix(2) + State)
    s5 = set()
    if sig3: s5.add(f"pref3_{sig3}")
    if sig2 and state: s5.add(f"p2st_{sig2}_{state}")
    disj["disj_name3_OR_name2state"] = s5

    return disj


def main():
    print("=" * 95)
    print("RESEARCH EXPERIMENT: COARSE BLOCKING KEYS FOR INDIA ENTITY RESOLUTION")
    print("=" * 95)

    # 1. Load real India Source 1 queries with Ground Truth
    N_SAMPLES = 8000
    print(f"\n[Phase 1] Loading {N_SAMPLES:,} India Source 1 entities and Ground Truth matches...")
    
    s1_df = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t")
    s1_india = s1_df[s1_df["country"] == "India"].head(N_SAMPLES)
    s1_in_ids = set(s1_india["entity_id"])
    print(f"  Loaded {len(s1_india):,} India S1 reference records.")

    # Load Ground Truth
    gt_pairs = []
    needed_cands = set()
    with open("student_resource/dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in s1_in_ids:
                matches = parts[1] if len(parts) > 1 else ""
                if matches:
                    for cid in matches.split(","):
                        cid = cid.strip()
                        if cid:
                            gt_pairs.append((s1_id, cid))
                            needed_cands.add(cid)

    print(f"  Found {len(gt_pairs):,} true positive India match pairs across {len(needed_cands):,} candidate entities.")

    # 2. Load candidate details for all needed candidates
    print("\n[Phase 2] Fetching candidate details (Source 2 and Source 3)...")
    cand_dict = {}
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = f"student_resource/dataset/train/{src_file}"
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip("\n").split("\t")
                cid = parts[0]
                if cid in needed_cands:
                    name = parts[1] if len(parts) > 1 else ""
                    addr = parts[2] if len(parts) > 2 else ""
                    country = parts[3] if len(parts) > 3 else ""
                    cand_dict[cid] = (name, addr, country)
                if len(cand_dict) >= len(needed_cands):
                    break

    print(f"  Loaded entity details for all {len(cand_dict):,} true matching candidates.")

    # Build reference lookup
    s1_dict = {
        row["entity_id"]: (str(row["business_name"]), str(row["business_address"]))
        for _, row in s1_india.iterrows()
    }

    # 3. Analyze Missingness & Field Coverage in India Data
    print("\n[Phase 3] Field & Feature Presence in India (S1 and Candidates):")
    pincode_present_s1 = sum(1 for _, addr in s1_dict.values() if extract_pincode(addr))
    state_present_s1 = sum(1 for _, addr in s1_dict.values() if extract_state(addr))
    
    pincode_present_cand = sum(1 for name, addr, _ in cand_dict.values() if extract_pincode(addr))
    state_present_cand = sum(1 for name, addr, _ in cand_dict.values() if extract_state(addr))

    print(f"  Source 1 PIN Code present: {pincode_present_s1 / len(s1_dict):.2%}")
    print(f"  Source 1 State present:    {state_present_s1 / len(s1_dict):.2%}")
    print(f"  Candidate PIN Code present: {pincode_present_cand / len(cand_dict):.2%}")
    print(f"  Candidate State present:    {state_present_cand / len(cand_dict):.2%}")

    # 4. Evaluate Recall (Pair Completeness) on Ground Truth Pairs
    print("\n[Phase 4] Testing Single-Key Retention / Recall on True Matches:")
    print("-" * 80)
    print(f"{'Key Strategy':<28} | {'Retained Hits':<14} | {'Total True Pairs':<16} | {'Recall':<8}")
    print("-" * 80)

    key_hits = Counter()
    disj_hits = Counter()

    for s1_id, cid in gt_pairs:
        s1_name, s1_addr = s1_dict[s1_id]
        cand_name, cand_addr, _ = cand_dict[cid]
        
        # Single keys
        k1 = generate_blocking_keys(s1_name, s1_addr)
        k2 = generate_blocking_keys(cand_name, cand_addr)
        for key_name in k1:
            if k1[key_name] and k2[key_name] and k1[key_name] == k2[key_name]:
                key_hits[key_name] += 1

        # Disjunctive keys
        d1 = generate_disjunctive_keys(s1_name, s1_addr)
        d2 = generate_disjunctive_keys(cand_name, cand_addr)
        for d_name in d1:
            if not d1[d_name].isdisjoint(d2[d_name]):
                disj_hits[d_name] += 1

    total_pairs = len(gt_pairs)
    for key_name, count in sorted(key_hits.items(), key=lambda x: -x[1]):
        if key_name.startswith("_"):
            continue
        rec = count / total_pairs
        print(f"{key_name:<28} | {count:<14,d} | {total_pairs:<16,d} | {rec*100:6.2f}%")

    print("\n[Phase 5] Testing Disjunctive (Multi-Key) Retention / Recall on True Matches:")
    print("-" * 80)
    print(f"{'Disjunctive Strategy':<28} | {'Retained Hits':<14} | {'Total True Pairs':<16} | {'Recall':<8}")
    print("-" * 80)
    for d_name, count in sorted(disj_hits.items(), key=lambda x: -x[1]):
        rec = count / total_pairs
        print(f"{d_name:<28} | {count:<14,d} | {total_pairs:<16,d} | {rec*100:6.2f}%")

    # 5. Measure Candidate Pool Block Size & Reduction Ratio
    print("\n[Phase 6] Measuring Block Size Distribution & Reduction Ratio on 100,000 Real India Candidates:")
    sample_cand_pool = []
    with open("student_resource/dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == "India":
                sample_cand_pool.append((parts[0], parts[1], parts[2]))
                if len(sample_cand_pool) >= 100000:
                    break

    print(f"  Loaded {len(sample_cand_pool):,} real India candidate records.")
    
    # Evaluate reduction ratio and bucket skew for top strategies
    eval_strategies = [
        "raw_prefix_1", "raw_prefix_2", "raw_prefix_3",
        "clean_sig_prefix_1", "clean_sig_prefix_2", "clean_sig_prefix_3",
        "name_soundex", "addr_state", "addr_pincode_circle_2",
        "comp_sig1_state", "comp_sig2_state", "comp_sig1_pinzone",
        "comp_soundex_state",
    ]

    print("-" * 95)
    print(f"{'Strategy':<22} | {'Distinct Blocks':<15} | {'Max Block':<10} | {'Median Block':<12} | {'Theoretical RR':<15} | {'True Recall':<10}")
    print("-" * 95)

    for strat in eval_strategies:
        blocks = defaultdict(int)
        for cid, name, addr in sample_cand_pool:
            k = generate_blocking_keys(name, addr).get(strat, "")
            if k:
                blocks[k] += 1
            else:
                blocks["__MISSING__"] += 1

        sizes = list(blocks.values())
        n_blocks = len(blocks)
        max_size = max(sizes) if sizes else 0
        med_size = np.median(sizes) if sizes else 0
        
        # Reduction ratio estimate: 1 - sum(size^2) / N^2
        N = len(sample_cand_pool)
        pair_comparisons = sum(s * s for s in sizes)
        total_possible = N * N
        rr = 1.0 - (pair_comparisons / total_possible)
        
        rec = (key_hits[strat] / total_pairs) * 100
        print(f"{strat:<22} | {n_blocks:<15,d} | {max_size:<10,d} | {med_size:<12.1f} | {rr*100:13.2f}% | {rec:8.2f}%")

    print("=" * 95)


if __name__ == "__main__":
    main()
