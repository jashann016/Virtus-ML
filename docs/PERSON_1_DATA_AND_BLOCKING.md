# Person 1: Data Ingestion & Candidate Blocking Lead

## 1. Role Mission & Strategic Importance
* **Your Title:** Data & Candidate Generation Lead (Stage 1 Master)
* **Your Core Mission:** Find every plausible match across sources while discarding 99.9% of obvious non-matches.
* **Why You Are Critical:**
  * **You define the Recall Ceiling.** If you miss a true match in Stage 1, Person 2 and Person 3 can never find it. Your target candidate recall is **$\ge 95\%$**.
  * **You own the official submission format.** You ensure both `output/candidate_pairs.tsv` and `output/matching_results.tsv` pass the official `utils/validate_submission.py` script with zero errors.

---

## 2. Your Detailed Responsibilities & Tasks

### Task 1: Robust Text Normalization (`src/text_preprocessing.py`)
Real-world data contains accents, punctuation noise, legal terms, and typos.
* **Accent & Diacritics Stripping (For France):** Convert `é, è, ê, ç, à` $\rightarrow$ `e, e, e, c, a` using Unicode normalization (`unicodedata.normalize('NFKD', ...)`).
* **Legal Suffix Canonicalization:**
  * US: `inc, llc, corp, corporation, co, company, ltd, limited`.
  * India: `pvt ltd, private limited, llp, limited`.
  * France: `sarl, sas, sasu, sa, eurl, & fils, et fils`.
* **Address Token Standardization:**
  * English: `st -> street`, `rd -> road`, `ave -> avenue`, `blvd -> boulevard`, `bldg -> building`, `fl -> floor`.
  * French: `bd -> boulevard`, `av -> avenue`, `r -> rue`, `pl -> place`.
* **Country Partitioning Rule:** Businesses in the US will never match businesses in India or France. Blocking MUST be partitioned strictly within the same `country` label!

### Task 2: Candidate Blocking Engine (`src/blocking.py`)
Comparing all 1.7M test entities against 9.8M records is 16 trillion pairs. Your engine narrows this to ~20 to 40 candidates per Source 1 entity.
* **Technique A: Sublinear Character N-Gram TF-IDF:**
  * Use 3-to-4 character n-grams on the cleaned `business_name`.
  * Allows matching through typos (e.g., `Wilblims` vs `Williams`).
* **Technique B: Token-Sorted Inverted Index:**
  * Sort words alphabetically (e.g., `Bakery Sharma Sons` == `Sharma Sons Bakery`).
* **Technique C: Address Core Street / Number Key:**
  * Extracts street numbers + first 3 letters of street name to catch trade-name variations (`Dréxkor` matching `Maure Williams Colombier Inc`).
* **Output:** A combined, deduplicated candidate list per Source 1 entity.

### Task 3: Official Candidate File Generation
* Generate `output/candidate_pairs.tsv` adhering strictly to competition rules:
  * Tab-separated: `source1_entity_id \t candidate_entity_ids`
  * Exactly one row per test Source 1 entity.
  * Empty candidate list when no plausible candidate is found.
  * IDs in comma-separated list must be from Source 2 or 3 only, no duplicates, no self-matches.

---

## 3. Git Branch & Deliverables

* **Working Branch:** `feat/p1-blocking-engine`
* **Target PR Branch:** `dev`
* **Primary Code Files:**
  * `src/text_preprocessing.py`
  * `src/blocking.py`
  * `src/run_blocking_pipeline.py`
* **Output Deliverable:** `output/candidate_pairs.tsv`
* **Success Metric:** Candidate Recall on validation split $\ge 95\%$, candidate count $\le 50$ per entity.

---

## 4. Daily Checklist & PR Protocol
1. Pull latest `dev` and rebase your branch: `git checkout dev && git pull && git checkout feat/p1-blocking-engine && git rebase dev`.
2. Run test on validation slice to measure recall.
3. Push branch and open PR to `dev`.
4. Post PR link in the WhatsApp group for Team Leader approval.
