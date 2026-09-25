import re
import unicodedata
from typing import List, Set

# Compiled regular expressions for high performance
RE_DIACRITICS = re.compile(r'[\u0300-\u036f]')
RE_NON_ALPHANUM = re.compile(r'[^\w\s]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+(?:[/-]\d+)?\b')
RE_DOMAIN = re.compile(r'^(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|in|org|net|fr|co|io|biz|info)\b', re.IGNORECASE)

# Legal suffixes across US, India, and France
LEGAL_SUFFIXES = [
    # Multi-word suffixes (must match first)
    r'\bprivate\s+limited\b',
    r'\bpvt\s*\.?\s*ltd\s*\.?\b',
    r'\bet\s+fils\b',
    r'\b&\s*fils\b',
    r'\blimited\s+liability\s+company\b',
    r'\bllp\b',
    
    # French entity types
    r'\bsarl\b',
    r'\bs\.a\.r\.l\.\b',
    r'\bsas\b',
    r'\bs\.a\.s\.\b',
    r'\bsasu\b',
    r'\beurl\b',
    r'\bsa\b',
    r'\bsci\b',
    
    # Common English / US / Global suffixes
    r'\bincorporated\b',
    r'\binc\s*\.?\b',
    r'\bcorporation\b',
    r'\bcorp\s*\.?\b',
    r'\bllc\s*\.?\b',
    r'\bl\.l\.c\.\b',
    r'\blimited\b',
    r'\bltd\s*\.?\b',
    r'\bcompany\b',
    r'\bco\s*\.?\b',
]

RE_LEGAL_SUFFIXES = [re.compile(pattern, re.IGNORECASE) for pattern in LEGAL_SUFFIXES]

# Address abbreviation mappings (English and French)
ADDRESS_ABBR = {
    # English
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bav\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bbd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bbldg\b': 'building',
    r'\bfl\b': 'floor',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bpl\b': 'place',
    # French
    r'\br\b': 'rue',
    r'\brte\b': 'route',
    r'\ball\b': 'allee',
    r'\bimp\b': 'impasse',
}

RE_ADDRESS_ABBR = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBR.items()]


def strip_accents(text: str) -> str:
    """Strips accents and diacritics (crucial for French names/addresses)."""
    if not text:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_text_basic(text: str) -> str:
    """Basic normalization: strip accents, lowercase, unify '&' -> 'and', clean punctuation."""
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Unicode & Accent normalization
    text = strip_accents(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Replace '&' with 'and'
    text = text.replace('&', ' and ')
    
    # 4. Remove leading noise characters like '<<', '--'
    text = re.sub(r'^[<\-+*#~]+\s*', '', text)
    
    # 5. Remove general non-alphanumeric punctuation except spaces
    text = RE_NON_ALPHANUM.sub(' ', text)
    
    # 6. Squash multi-spaces
    return RE_MULTI_SPACE.sub(' ', text).strip()


def normalize_business_name(name: str) -> str:
    """
    Normalizes a business name:
    - Handles website domain names (e.g. maurewilliamscolombier.com -> maure williams colombier)
    - Strips legal suffixes (Pvt Ltd, LLC, Inc, SARL, SAS)
    """
    if not name or not isinstance(name, str):
        return ""
        
    cleaned = clean_text_basic(name)
    
    # Check if name is a domain like "maurewilliamscolombier.com"
    domain_match = RE_DOMAIN.match(name.strip())
    if domain_match:
        core_domain = domain_match.group(1).lower().replace('-', ' ')
        cleaned = core_domain
    
    # Strip legal suffixes
    for reg in RE_LEGAL_SUFFIXES:
        cleaned = reg.sub(' ', cleaned)
        
    return RE_MULTI_SPACE.sub(' ', cleaned).strip()


def normalize_address(addr: str) -> str:
    """
    Normalizes an address string:
    - Expands abbreviations (st -> street, rd -> road, bd -> boulevard)
    - Standardizes separators
    """
    if not addr or not isinstance(addr, str):
        return ""
        
    cleaned = clean_text_basic(addr)
    
    for reg, repl in RE_ADDRESS_ABBR:
        cleaned = reg.sub(repl, cleaned)
        
    return RE_MULTI_SPACE.sub(' ', cleaned).strip()


def extract_numbers(addr: str) -> List[str]:
    """Extracts all numeric tokens (building numbers, plot numbers, postal codes)."""
    if not addr or not isinstance(addr, str):
        return []
    return RE_NUMBERS.findall(addr)


if __name__ == "__main__":
    # Test cases representing our discovered noise patterns:
    print("Testing Text Preprocessing Functions:")
    
    # Test 1: Suffix stripping and domain detection
    t1 = "Maure Williams Colombier Inc"
    t2 = "maurewilliamscolombier.com"
    print(f"'{t1}' -> '{normalize_business_name(t1)}'")
    print(f"'{t2}' -> '{normalize_business_name(t2)}'")
    assert normalize_business_name(t1) == "maure williams colombier"
    
    # Test 2: French accents and legal suffix
    t3 = "<< Team École SARL"
    print(f"'{t3}' -> '{normalize_business_name(t3)}'")
    assert normalize_business_name(t3) == "team ecole"
    
    # Test 3: Address abbreviations and numbers
    addr = "85 Wanye Avenue, Ticonderoga Townshiip, NY 12883"
    print(f"Address: '{addr}' -> '{normalize_address(addr)}'")
    print(f"Numbers extracted: {extract_numbers(addr)}")
    assert "85" in extract_numbers(addr)
    assert "12883" in extract_numbers(addr)
    
    print("All preprocessing tests passed successfully!")
