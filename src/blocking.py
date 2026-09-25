import os
import sys
import gc
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.text_preprocessing import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
    extract_plot_codes,
    has_non_latin
)

STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'by', 'with', 'from', 
    'de', 'du', 'des', 'la', 'le', 'et', 'near', 'opp', 'opposite', 'behind'
}

COMMON_CITY_TOKENS = {
    'delhi', 'mumbai', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 
    'kolkata', 'pune', 'ahmedabad', 'jaipur', 'india', 'new', 'south', 'north',
    'east', 'west', 'paris', 'lyon', 'bordeaux', 'ny', 'ca', 'tx', 'fl'
}

GENERIC_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'lane', 'drive', 'floor', 
    'building', 'rue', 'allee', 'route', 'court', 'plaza', 'parkway'
}


def extract_compound_numbers(text: str) -> List[str]:
    """Extracts complex municipal plot/house numbers like 6-3-712, 32/2, 1/522."""
    if not text:
        return []
    matches = re.findall(r'\b\d+(?:[-/]\d+)+\b', text)
    cleaned = []
    for m in matches:
        cleaned.append(m.replace('/', '_').replace('-', '_'))
        # Also include the first two segments
        parts = re.split(r'[-/]', m)
        if len(parts) >= 2:
            cleaned.append(f"{parts[0]}_{parts[1]}")
    return list(set(cleaned))


def extract_address_keys(address_norm: str, raw_address: str = "") -> List[str]:
    """Extracts high-precision blocking keys from an address."""
    if not address_norm:
        return []
    
    tokens = address_norm.split()
    numbers = [t for t in tokens if any(c.isdigit() for c in t)]
    sig_words = [
        t for t in tokens 
        if len(t) >= 4 and not any(c.isdigit() for c in t) and t not in GENERIC_ADDR_WORDS and t not in STOP_WORDS
    ]
    
    keys = []
    
    # 1. Alphanumeric plot codes (B-148 -> b148)
    plot_codes = extract_plot_codes(raw_address)
    for p in plot_codes:
        keys.append(f"plot_{p}")
        for w in sig_words[:2]:
            keys.append(f"{p}_{w}")
            
    # 2. Compound house numbers (e.g. 6-3-712 -> cnum_6_3_712)
    compound_nums = extract_compound_numbers(raw_address)
    for cnum in compound_nums:
        keys.append(f"cnum_{cnum}")
        for w in sig_words[:2]:
            keys.append(f"{cnum}_{w}")
            
    # 3. Building numbers combined with street words
    if numbers and sig_words:
        for num in numbers[:3]:
            for w in sig_words[:3]:
                keys.append(f"{num}_{w}")
                
    # 4. Location tokens (length >= 4)
    for w in sig_words:
        keys.append(f"loc_{w}")
                
    # 5. Distinctive street & locality word-pairs
    if len(sig_words) >= 2:
        for i in range(min(5, len(sig_words) - 1)):
            pair = "_".join(sorted([sig_words[i], sig_words[i + 1]]))
            keys.append(f"pair_{pair}")
            
    return keys


def get_char_trigrams(word: str) -> List[str]:
    """Extracts character 3-grams from a word to capture typo variations."""
    if len(word) < 3:
        return [word]
    return [word[i:i+3] for i in range(len(word) - 2)]


def load_tsv_records(file_path: str, max_rows: int = None) -> List[dict]:
    """Fast TSV reader."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_map = {col.strip(): i for i, col in enumerate(header)}
        id_idx = col_map.get('entity_id', 0)
        name_idx = col_map.get('business_name', 1)
        addr_idx = col_map.get('business_address', 2)
        country_idx = col_map.get('country', 3)
        
        for count, line in enumerate(f):
            if max_rows and count >= max_rows:
                break
            parts = line.strip('\n').split('\t')
            if len(parts) <= max(id_idx, name_idx, addr_idx, country_idx):
                continue
                
            records.append({
                'entity_id': parts[id_idx].strip(),
                'business_name': parts[name_idx].strip(),
                'business_address': parts[addr_idx].strip(),
                'country': parts[country_idx].strip()
            })
    return records


class InvertedIndexBlocker:
    """
    Precision-Engineered Multi-Angle Candidate Generation Engine:
    - Partitioned strictly by country (US, India, France).
    - Multi-angle indexing:
      1. Exact full-name match index (+40 boost)
      2. First-word brand token index (+15 boost)
      3. Compound house numbers (6-3-712, 32/2)
      4. Alphanumeric plot codes (B148, A42)
      5. Character trigrams for typos
      6. Cross-script sensitivity
    """
    def __init__(self):
        self.exact_name_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.first_word_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.name_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.sorted_name_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.trigram_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.address_key_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def index_candidates(self, candidates: List[dict]):
        """Indexes candidate records from Source 2 and Source 3."""
        for cand in candidates:
            c_id = cand['entity_id']
            country = cand['country']
            raw_name = cand['business_name']
            raw_addr = cand['business_address']
            
            norm_name = normalize_business_name(raw_name)
            norm_addr = normalize_address(raw_addr)
            
            # 1. Exact full normalized name
            if norm_name:
                self.exact_name_index[country][norm_name].append(c_id)
            
            # 2. Name tokens & first-word brand index
            name_words = [w for w in norm_name.split() if len(w) >= 2 and w not in STOP_WORDS]
            if name_words:
                first_w = name_words[0]
                if first_w not in COMMON_CITY_TOKENS and len(first_w) >= 3:
                    self.first_word_index[country][first_w].append(c_id)
                    
            for w in name_words:
                self.name_token_index[country][w].append(c_id)
                if len(w) >= 4:
                    for tri in get_char_trigrams(w):
                        self.trigram_index[country][tri].append(c_id)
            
            # 3. Token-sorted name key
            if len(name_words) >= 2:
                sorted_key = "_".join(sorted(name_words[:4]))
                self.sorted_name_index[country][sorted_key].append(c_id)
                    
            # 4. Address keys (numbers, compound numbers, plot codes, locations)
            addr_keys = extract_address_keys(norm_addr, raw_address=raw_addr)
            for k in addr_keys:
                self.address_key_index[country][k].append(c_id)

    def retrieve_candidates(
        self, 
        s1_record: dict, 
        max_candidates: int = 90
    ) -> List[str]:
        """Finds top candidate entity_ids for a single Source 1 entity."""
        country = s1_record['country']
        raw_name = s1_record['business_name']
        raw_addr = s1_record['business_address']
        
        norm_name = normalize_business_name(raw_name)
        norm_addr = normalize_address(raw_addr)
        
        is_cross_script = has_non_latin(raw_name) or country == "India"
        candidate_scores = defaultdict(int)
        
        # 1. Exact name match boost (+40 points)
        if norm_name:
            for c_id in self.exact_name_index[country].get(norm_name, []):
                candidate_scores[c_id] += 40
                
        # 2. Name words & First-word brand token matching
        name_words = [w for w in norm_name.split() if len(w) >= 2 and w not in STOP_WORDS]
        if name_words:
            first_w = name_words[0]
            if first_w not in COMMON_CITY_TOKENS and len(first_w) >= 3:
                for c_id in self.first_word_index[country].get(first_w, []):
                    candidate_scores[c_id] += 15
                    
        for w in name_words:
            # Common city names in business names carry lower weight
            w_weight = 2 if w in COMMON_CITY_TOKENS else 6
            for c_id in self.name_token_index[country].get(w, []):
                candidate_scores[c_id] += w_weight
            if len(w) >= 4 and w not in COMMON_CITY_TOKENS:
                for tri in get_char_trigrams(w):
                    for c_id in self.trigram_index[country].get(tri, []):
                        candidate_scores[c_id] += 1
                        
        # 3. Token-sorted name matching
        if len(name_words) >= 2:
            sorted_key = "_".join(sorted(name_words[:4]))
            for c_id in self.sorted_name_index[country].get(sorted_key, []):
                candidate_scores[c_id] += 12
                    
        # 4. Address key matching
        addr_keys = extract_address_keys(norm_addr, raw_address=raw_addr)
        for k in addr_keys:
            if k.startswith('cnum_') or k.startswith('plot_') or k.startswith('pair_'):
                weight = 15
            elif is_cross_script:
                weight = 8
            else:
                weight = 4
            for c_id in self.address_key_index[country].get(k, []):
                candidate_scores[c_id] += weight
                
        if not candidate_scores:
            return []
            
        sorted_candidates = sorted(
            candidate_scores.keys(), 
            key=lambda cid: candidate_scores[cid], 
            reverse=True
        )
        return sorted_candidates[:max_candidates]


def write_candidate_pairs_file(
    s1_records: List[dict],
    blocker: InvertedIndexBlocker,
    output_path: str,
    max_candidates: int = 90
):
    """Generates output/candidate_pairs.tsv formatted according to competition rules."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for record in s1_records:
            s1_id = record['entity_id']
            candidates = blocker.retrieve_candidates(record, max_candidates=max_candidates)
            cand_str = ",".join(candidates) if candidates else ""
            f.write(f"{s1_id}\t{cand_str}\n")
