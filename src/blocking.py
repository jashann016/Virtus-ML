import os
import sys
import gc
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

GENERIC_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'lane', 'drive', 'floor', 
    'building', 'block', 'phase', 'sector', 'nagar', 'colony', 'city', 
    'rue', 'allee', 'place', 'route', 'court', 'plaza', 'parkway'
}


def extract_address_keys(address_norm: str, raw_address: str = "") -> List[str]:
    """
    Extracts high-precision blocking keys from an address:
    1. Alphanumeric plot codes (e.g. 'b148', 'g3')
    2. Numbers combined with significant street/city tokens
    3. Distinctive street/city word-pairs (e.g. 'puri_station', 'kingsport_claremont')
    """
    if not address_norm:
        return []
    
    tokens = address_norm.split()
    numbers = [t for t in tokens if any(c.isdigit() for c in t)]
    sig_words = [
        t for t in tokens 
        if len(t) >= 4 and not any(c.isdigit() for c in t) and t not in GENERIC_ADDR_WORDS and t not in STOP_WORDS
    ]
    
    keys = []
    
    # 1. Alphanumeric plot codes (e.g. B-148 -> b148)
    plot_codes = extract_plot_codes(raw_address)
    for p in plot_codes:
        keys.append(f"plot_{p}")
        for w in sig_words[:2]:
            keys.append(f"{p}_{w}")
            
    # 2. Building numbers combined with street words
    if numbers and sig_words:
        for num in numbers[:2]:
            for w in sig_words[:3]:
                keys.append(f"{num}_{w}")
                
    # 3. Distinctive street & locality word-pairs (handles records with missing numbers)
    if len(sig_words) >= 2:
        for i in range(min(3, len(sig_words) - 1)):
            pair = "_".join(sorted([sig_words[i], sig_words[i + 1]]))
            keys.append(f"loc_{pair}")
            
    return keys


def get_char_trigrams(word: str) -> List[str]:
    """Extracts character 3-grams from a word to capture typo variations."""
    if len(word) < 3:
        return [word]
    return [word[i:i+3] for i in range(len(word) - 2)]


def load_tsv_records(file_path: str, max_rows: int = None) -> List[dict]:
    """Fast TSV reader for entity_id, business_name, business_address, country."""
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
    Advanced Multi-Angle Candidate Generation Engine:
    - Partitioned strictly by country (US, India, France).
    - Multi-angle indexing:
      1. Word tokens & trigrams (names)
      2. Token-sorted name keys
      3. Alphanumeric plot/block codes (B-148, G-3)
      4. Building-number + street keys
      5. Locality/landmark word-pairs
      6. Cross-script sensitivity (Indic scripts <-> English)
    """
    def __init__(self):
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
            
            # 1. Name tokens & trigrams
            name_words = [w for w in norm_name.split() if len(w) >= 2 and w not in STOP_WORDS]
            for w in name_words:
                self.name_token_index[country][w].append(c_id)
                if len(w) >= 4:
                    for tri in get_char_trigrams(w):
                        self.trigram_index[country][tri].append(c_id)
            
            # 2. Token-sorted name key
            if len(name_words) >= 2:
                sorted_key = "_".join(sorted(name_words[:4]))
                self.sorted_name_index[country][sorted_key].append(c_id)
                    
            # 3. Address keys (numbers, plot codes, locality pairs)
            addr_keys = extract_address_keys(norm_addr, raw_address=raw_addr)
            for k in addr_keys:
                self.address_key_index[country][k].append(c_id)

    def retrieve_candidates(
        self, 
        s1_record: dict, 
        max_candidates: int = 65
    ) -> List[str]:
        """Finds top candidate entity_ids for a single Source 1 entity."""
        country = s1_record['country']
        raw_name = s1_record['business_name']
        raw_addr = s1_record['business_address']
        
        norm_name = normalize_business_name(raw_name)
        norm_addr = normalize_address(raw_addr)
        
        is_cross_script = has_non_latin(raw_name) or country == "India"
        candidate_scores = defaultdict(int)
        name_words = [w for w in norm_name.split() if len(w) >= 2 and w not in STOP_WORDS]
        
        # 1. Exact word token matching
        for w in name_words:
            for c_id in self.name_token_index[country].get(w, []):
                candidate_scores[c_id] += 5
            if len(w) >= 4:
                for tri in get_char_trigrams(w):
                    for c_id in self.trigram_index[country].get(tri, []):
                        candidate_scores[c_id] += 1
                        
        # 2. Token-sorted name matching
        if len(name_words) >= 2:
            sorted_key = "_".join(sorted(name_words[:4]))
            for c_id in self.sorted_name_index[country].get(sorted_key, []):
                candidate_scores[c_id] += 10
                    
        # 3. Address key matching
        addr_keys = extract_address_keys(norm_addr, raw_address=raw_addr)
        addr_weight = 12 if is_cross_script else 8
        
        for k in addr_keys:
            for c_id in self.address_key_index[country].get(k, []):
                # Extra weight for plot codes and exact building numbers
                candidate_scores[c_id] += addr_weight
                
        if not candidate_scores:
            return []
            
        # Return top-K candidates
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
    max_candidates: int = 65
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
