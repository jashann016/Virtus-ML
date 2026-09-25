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
    extract_numbers
)


def extract_address_keys(address_norm: str) -> List[str]:
    """
    Extracts blocking keys from an address:
    Combines numeric tokens (building numbers, postal codes) with significant street/city words.
    """
    if not address_norm:
        return []
    
    tokens = address_norm.split()
    numbers = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if len(t) >= 3 and not any(c.isdigit() for c in t)]
    
    keys = []
    if numbers and words:
        for num in numbers[:2]:
            for w in words[:3]:
                keys.append(f"{num}_{w}")
    return keys


def get_char_trigrams(word: str) -> List[str]:
    """Extracts character 3-grams from a word to capture typo variations."""
    if len(word) < 3:
        return [word]
    return [word[i:i+3] for i in range(len(word) - 2)]


def load_tsv_records(file_path: str, max_rows: int = None) -> List[dict]:
    """
    Fast TSV reader without external dependencies.
    Reads entity_id, business_name, business_address, country.
    """
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


STOP_WORDS = {'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'by', 'with', 'from', 'de', 'du', 'des', 'la', 'le', 'et'}


class InvertedIndexBlocker:
    """
    High-Recall Candidate Generation Engine:
    - Partitioned strictly by country (US, India, France).
    - Multi-angle inverted indexing:
      1. Exact word tokens (weight 5)
      2. Token-sorted name key (weight 10)
      3. Character 3-grams for typos (weight 1)
      4. Building-number + street keys for DBAs / trade names (weight 8)
    """
    def __init__(self):
        self.name_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.sorted_name_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.trigram_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.address_key_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def index_candidates(self, candidates: List[dict]):
        """Indexes records from Source 2 and Source 3."""
        for cand in candidates:
            c_id = cand['entity_id']
            country = cand['country']
            raw_name = cand['business_name']
            raw_addr = cand['business_address']
            
            norm_name = normalize_business_name(raw_name)
            norm_addr = normalize_address(raw_addr)
            
            # 1. Full word tokens (words with length >= 2, excluding stopwords)
            name_words = [w for w in norm_name.split() if len(w) >= 2 and w not in STOP_WORDS]
            for w in name_words:
                self.name_token_index[country][w].append(c_id)
                
                # 2. Character trigrams for typo resilience
                if len(w) >= 4:
                    for tri in get_char_trigrams(w):
                        self.trigram_index[country][tri].append(c_id)
            
            # 3. Token-sorted name key
            if len(name_words) >= 2:
                sorted_key = "_".join(sorted(name_words[:4]))
                self.sorted_name_index[country][sorted_key].append(c_id)
                    
            # 4. Address keys (number + street)
            addr_keys = extract_address_keys(norm_addr)
            for k in addr_keys:
                self.address_key_index[country][k].append(c_id)

    def retrieve_candidates(
        self, 
        s1_record: dict, 
        max_candidates: int = 50
    ) -> List[str]:
        """Finds top candidate entity_ids for a single Source 1 entity."""
        country = s1_record['country']
        norm_name = normalize_business_name(s1_record['business_name'])
        norm_addr = normalize_address(s1_record['business_address'])
        
        candidate_scores = defaultdict(int)
        name_words = [w for w in norm_name.split() if len(w) >= 2 and w not in STOP_WORDS]
        
        # 1. Exact word token matching
        for w in name_words:
            for c_id in self.name_token_index[country].get(w, []):
                candidate_scores[c_id] += 5
                
            # Trigram matching for typos
            if len(w) >= 4:
                for tri in get_char_trigrams(w):
                    for c_id in self.trigram_index[country].get(tri, []):
                        candidate_scores[c_id] += 1
                        
        # 2. Token-sorted name matching
        if len(name_words) >= 2:
            sorted_key = "_".join(sorted(name_words[:4]))
            for c_id in self.sorted_name_index[country].get(sorted_key, []):
                candidate_scores[c_id] += 10
                    
        # 3. Address key matching (Very strong signal for DBAs / trade names)
        addr_keys = extract_address_keys(norm_addr)
        for k in addr_keys:
            for c_id in self.address_key_index[country].get(k, []):
                candidate_scores[c_id] += 8
                
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
    max_candidates: int = 35
):
    """
    Generates output/candidate_pairs.tsv formatted according to competition rules.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for record in s1_records:
            s1_id = record['entity_id']
            candidates = blocker.retrieve_candidates(record, max_candidates=max_candidates)
            cand_str = ",".join(candidates) if candidates else ""
            f.write(f"{s1_id}\t{cand_str}\n")


if __name__ == "__main__":
    print("Testing InvertedIndexBlocker...")
    blocker = InvertedIndexBlocker()
    mock_candidates = [
        {'entity_id': 'S2-001', 'business_name': 'Maure Wilblims Colombier Inc', 'business_address': '', 'country': 'US'},
        {'entity_id': 'S3-002', 'business_name': 'Dréxkor', 'business_address': '85 Wanye Avenue, Ticonderoga', 'country': 'US'},
        {'entity_id': 'S2-003', 'business_name': 'Totally Unrelated', 'business_address': '100 Main St', 'country': 'US'}
    ]
    blocker.index_candidates(mock_candidates)
    query = {'entity_id': 'S1-001', 'business_name': 'Maure Williams Colombier Inc', 'business_address': '85 Wayne Avenue', 'country': 'US'}
    retrieved = blocker.retrieve_candidates(query, max_candidates=10)
    print(f"Retrieved: {retrieved}")
    assert 'S2-001' in retrieved
    assert 'S3-002' in retrieved
    print("Tests passed successfully!")
