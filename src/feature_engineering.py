from typing import Dict, Any
from functools import lru_cache

from rapidfuzz.fuzz import token_sort_ratio, token_set_ratio
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.text_preprocessing import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
    RE_DOMAIN,
)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


@lru_cache(maxsize=100000)
def cached_business_name(value: str) -> str:
    return normalize_business_name(value)


@lru_cache(maxsize=100000)
def cached_address(value: str) -> str:
    return normalize_address(value)


@lru_cache(maxsize=100000)
def cached_numbers(value: str):
    return tuple(extract_numbers(value))


@lru_cache(maxsize=100000)
def cached_domain(value: str) -> str:
    if not value:
        return ""
    match = RE_DOMAIN.match(value.strip())
    return match.group(1).lower() if match else ""


def number_overlap_ratio(numbers_a, numbers_b) -> float:
    a = set(numbers_a)
    b = set(numbers_b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def address_token_jaccard(addr_a: str, addr_b: str) -> float:
    a = set(addr_a.split())
    b = set(addr_b.split())

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def token_count_ratio(a: str, b: str) -> float:
    a_count = len(a.split())
    b_count = len(b.split())

    if a_count == 0 and b_count == 0:
        return 1.0
    if a_count == 0 or b_count == 0:
        return 0.0

    return min(a_count, b_count) / max(a_count, b_count)


def compute_pair_features(
    source1: Dict[str, Any],
    candidate: Dict[str, Any]
) -> Dict[str, float]:

    raw_name_a = safe_text(source1.get("business_name"))
    raw_name_b = safe_text(candidate.get("business_name"))

    raw_addr_a = safe_text(source1.get("business_address"))
    raw_addr_b = safe_text(candidate.get("business_address"))

    # Cached normalization
    name_a = cached_business_name(raw_name_a)
    name_b = cached_business_name(raw_name_b)

    addr_a = cached_address(raw_addr_a)
    addr_b = cached_address(raw_addr_b)

    # Cached numbers/domains
    numbers_a = cached_numbers(raw_addr_a)
    numbers_b = cached_numbers(raw_addr_b)

    domain_a = cached_domain(raw_name_a)
    domain_b = cached_domain(raw_name_b)

    # Name similarities
    levenshtein_ratio = Levenshtein.normalized_similarity(name_a, name_b)
    jaro_winkler = JaroWinkler.normalized_similarity(name_a, name_b)
    token_sort = token_sort_ratio(name_a, name_b) / 100.0
    token_set = token_set_ratio(name_a, name_b) / 100.0

    # Exact name
    exact_name_match = float(
        bool(name_a) and bool(name_b) and name_a == name_b
    )

    # Domain
    domain_match_flag = float(
        bool(domain_a) and bool(domain_b) and domain_a == domain_b
    )

    # Number features
    number_overlap = number_overlap_ratio(numbers_a, numbers_b)

    set_numbers_a = set(numbers_a)
    set_numbers_b = set(numbers_b)

    exact_number_match = float(
        bool(set_numbers_a)
        and bool(set_numbers_b)
        and set_numbers_a == set_numbers_b
    )

    number_conflict_flag = float(
        bool(set_numbers_a)
        and bool(set_numbers_b)
        and not (set_numbers_a & set_numbers_b)
    )

    # Address features
    address_jaccard = address_token_jaccard(addr_a, addr_b)

    address_token_sort = (
        token_sort_ratio(addr_a, addr_b) / 100.0
        if addr_a and addr_b
        else 0.0
    )

    missing_address_flag = float(not addr_a or not addr_b)

    # Name length
    max_name_length = max(len(name_a), len(name_b), 1)
    name_length_diff = (
        abs(len(name_a) - len(name_b)) / max_name_length
    )

    # Token count
    token_ratio = token_count_ratio(name_a, name_b)

    return {
        "levenshtein_ratio": levenshtein_ratio,
        "jaro_winkler": jaro_winkler,
        "token_sort_ratio": token_sort,
        "token_set_ratio": token_set,
        "exact_name_match": exact_name_match,
        "domain_match_flag": domain_match_flag,
        "number_overlap_ratio": number_overlap,
        "exact_number_match": exact_number_match,
        "number_conflict_flag": number_conflict_flag,
        "address_token_jaccard": address_jaccard,
        "address_token_sort_ratio": address_token_sort,
        "missing_address_flag": missing_address_flag,
        "name_length_diff": name_length_diff,
        "token_count_ratio": token_ratio,
    }


FEATURE_COLUMNS = [
    "levenshtein_ratio",
    "jaro_winkler",
    "token_sort_ratio",
    "token_set_ratio",
    "exact_name_match",
    "domain_match_flag",
    "number_overlap_ratio",
    "exact_number_match",
    "number_conflict_flag",
    "address_token_jaccard",
    "address_token_sort_ratio",
    "missing_address_flag",
    "name_length_diff",
    "token_count_ratio",
]