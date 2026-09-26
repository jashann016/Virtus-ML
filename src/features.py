"""
VIRTUS-ML: High-Speed Pairwise Feature Engineering Engine
Extracts 17 high-signal similarity features for candidate pairs.
Optimized for LightGBM tabular classification under Macro-F0.5.
"""

import re
from typing import Dict, List, Set, Any
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.text_preprocessing import (
    normalize_business_name,
    normalize_address,
    has_non_latin,
    extract_numbers,
    extract_plot_codes,
    RE_DOMAIN
)

FEATURE_NAMES = [
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_ratio",
    "name_exact_clean",
    "name_first_word_match",
    "name_len_diff_ratio",
    "addr_token_set_ratio",
    "addr_token_sort_ratio",
    "addr_jaccard",
    "number_match",
    "plot_code_match",
    "domain_match",
    "country_match",
    "non_latin_flag",
    "reciprocal_rank",
    "has_matching_street_token"
]


def is_domain_link(name_a: str, name_b: str, norm_a: str, norm_b: str) -> float:
    """Returns 1.0 if one record is a web domain corresponding to the other business name."""
    dom_a = RE_DOMAIN.match(name_a.strip())
    dom_b = RE_DOMAIN.match(name_b.strip())
    
    if dom_a and not dom_b:
        core_a = dom_a.group(1).lower().replace('-', ' ')
        if core_a in norm_b or fuzz.token_sort_ratio(core_a, norm_b) >= 75:
            return 1.0
    elif dom_b and not dom_a:
        core_b = dom_b.group(1).lower().replace('-', ' ')
        if core_b in norm_a or fuzz.token_sort_ratio(core_b, norm_a) >= 75:
            return 1.0
    elif any(ext in name_a.lower() or ext in name_b.lower() for ext in ['.com', '.in', '.org', '.net']):
        if fuzz.token_sort_ratio(norm_a, norm_b) >= 70:
            return 1.0
            
    return 0.0


def compute_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Computes Jaccard token overlap between two sets of tokens."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return intersection / union if union > 0 else 0.0


def extract_pairwise_features(
    rec_a: Dict[str, Any],
    rec_b: Dict[str, Any],
    rank: int = 1,
    norm_name_a: str = None,
    norm_name_b: str = None,
    norm_addr_a: str = None,
    norm_addr_b: str = None
) -> Dict[str, float]:
    """
    Extracts 17 numerical similarity features for a pair of entities.
    Pre-normalized strings can be passed in to maximize throughput.
    """
    raw_name_a = rec_a.get('business_name', '')
    raw_name_b = rec_b.get('business_name', '')
    raw_addr_a = rec_a.get('business_address', '')
    raw_addr_b = rec_b.get('business_address', '')
    country_a = rec_a.get('country', '').strip().upper()
    country_b = rec_b.get('country', '').strip().upper()

    if norm_name_a is None:
        norm_name_a = normalize_business_name(raw_name_a)
    if norm_name_b is None:
        norm_name_b = normalize_business_name(raw_name_b)
    if norm_addr_a is None:
        norm_addr_a = normalize_address(raw_addr_a)
    if norm_addr_b is None:
        norm_addr_b = normalize_address(raw_addr_b)

    # 1-7: Name Similarities
    token_sort = fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0
    token_set = fuzz.token_set_ratio(norm_name_a, norm_name_b) / 100.0
    ratio = fuzz.ratio(norm_name_a, norm_name_b) / 100.0
    jaro_winkler = float(JaroWinkler.similarity(norm_name_a, norm_name_b))
    exact_clean = 1.0 if (norm_name_a and norm_name_a == norm_name_b) else 0.0
    
    words_a = norm_name_a.split()
    words_b = norm_name_b.split()
    first_word_match = 1.0 if (words_a and words_b and words_a[0] == words_b[0]) else 0.0
    
    max_len = max(len(norm_name_a), len(norm_name_b), 1)
    len_diff = abs(len(norm_name_a) - len(norm_name_b)) / max_len

    # 8-10: Address Similarities
    addr_token_set = fuzz.token_set_ratio(norm_addr_a, norm_addr_b) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0
    
    addr_words_a = set(norm_addr_a.split())
    addr_words_b = set(norm_addr_b.split())
    addr_jaccard = compute_jaccard(addr_words_a, addr_words_b)

    # 11: Number Match
    nums_a = set(extract_numbers(raw_addr_a))
    nums_b = set(extract_numbers(raw_addr_b))
    if nums_a and nums_b:
        number_match = 1.0 if nums_a.intersection(nums_b) else 0.0
    else:
        number_match = 0.5  # Neutral when address lacks street numbers

    # 12: Indian Plot Code Match
    plots_a = set(extract_plot_codes(raw_addr_a))
    plots_b = set(extract_plot_codes(raw_addr_b))
    if plots_a and plots_b:
        plot_code_match = 1.0 if plots_a.intersection(plots_b) else 0.0
    else:
        plot_code_match = 0.5

    # 13: Domain Link Match
    domain_match = is_domain_link(raw_name_a, raw_name_b, norm_name_a, norm_name_b)

    # 14: Country Match
    country_match = 1.0 if (country_a and country_a == country_b) else 0.0

    # 15: Cross-Script / Non-Latin Flag
    is_non_latin = 1.0 if (has_non_latin(raw_name_a) or has_non_latin(raw_name_b) or 
                           has_non_latin(raw_addr_a) or has_non_latin(raw_addr_b)) else 0.0

    # 16: Reciprocal Rank from Blocker
    reciprocal_rank = 1.0 / max(rank, 1)

    # 17: Street Token Overlap (excluding generic terms like 'street', 'road', etc.)
    stopwords = {'street', 'road', 'avenue', 'boulevard', 'lane', 'drive', 'court', 'highway', 'parkway', 'rue', 'suite', 'unit', 'floor', 'apt', 'building'}
    meaningful_a = {w for w in addr_words_a if len(w) > 3 and w not in stopwords}
    meaningful_b = {w for w in addr_words_b if len(w) > 3 and w not in stopwords}
    street_token_match = 1.0 if meaningful_a.intersection(meaningful_b) else 0.0

    return {
        "name_token_sort_ratio": token_sort,
        "name_token_set_ratio": token_set,
        "name_jaro_winkler": jaro_winkler,
        "name_ratio": ratio,
        "name_exact_clean": exact_clean,
        "name_first_word_match": first_word_match,
        "name_len_diff_ratio": len_diff,
        "addr_token_set_ratio": addr_token_set,
        "addr_token_sort_ratio": addr_token_sort,
        "addr_jaccard": addr_jaccard,
        "number_match": number_match,
        "plot_code_match": plot_code_match,
        "domain_match": domain_match,
        "country_match": country_match,
        "non_latin_flag": is_non_latin,
        "reciprocal_rank": reciprocal_rank,
        "has_matching_street_token": street_token_match
    }
