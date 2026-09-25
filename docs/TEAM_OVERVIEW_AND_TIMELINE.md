# Team Strategy, Problem Analysis & Milestone Timeline

## 1. Challenge Overview
* **Objective:** Given business records from 3 independent sources ($S_1$, $S_2$, $S_3$) with noise, typos, abbreviations, and missing fields, determine which records in $S_2$ and $S_3$ match each deduplicated reference record in $S_1$.
* **Cardinality:** 1-to-0 (singleton), 1-to-1, or 1-to-many.
* **Country Shift:** Train has `US` and `India`. Test adds `France` (zero-shot generalization).
* **Data Scale:**
  * Training: 2.2M $S_1$, 5.0M $S_2$, 5.2M $S_3$ (Total ~12.5M records).
  * Test: 1.7M $S_1$, 4.8M $S_2$, 5.0M $S_3$ (Total ~11.5M records).
  * Grand total: ~26.4 Million rows across all files.

---

## 2. Real-World Ground Truth Discoveries
From our inspection of real clusters (e.g., `S1-965667`), matches exhibit specific real-world patterns:

| Pattern Type | Example in Ground Truth | How Our Pipeline Handles It |
| :--- | :--- | :--- |
| **Typo in Name** | `Williams` vs `Wilblims` | Character n-gram TF-IDF + Levenshtein distance |
| **Dropped Legal Suffix** | `Colombier Inc` vs `Colombier` | Suffix canonicalization (`Inc`, `Corp`, `Pvt Ltd`, `SARL`) |
| **Website as Name** | `maurewilliamscolombier.com` | Domain name extractor & string-containment |
| **DBA / Trade Name** | `Maure Williams Colombier Inc` vs `Dréxkor` | Building number `85` + street `Wayne Avenue` match |
| **Missing Address** | Address is completely blank | Fallback to ultra-high name confidence |
| **Multilingual Records** | Hindi Devanagari script vs English | Multilingual text normalization + Qwen 2.5 7B |
| **French Zero-Shot** | `<< Team Ecole`, `175 Boulevard...`, `SARL` | International suffix strip + accent removal |

---

## 3. Evaluation Metric: Macro-$F_{0.5}$ & Singletons

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

* **Precision is weighted 2× over Recall.**
* **Penalizing False Merges:** Linking two different stores together drops your score twice as fast as missing a link.
* **The Singleton Goldmine:**
  * If a Source 1 entity has no true matches, predicting an empty list gives **1.0 (100%)**.
  * Predicting even 1 wrong candidate drops it to **0.0 (0%)**.
* **Core Takeaway:** *Never guess blindly. A high confidence threshold $\tau \approx 0.75 - 0.85$ maximizes macro-$F_{0.5}$.*

---

## 4. End-to-End System Architecture

```
[ Raw Test Data (S1, S2, S3) ]
             │
             ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Blocking & Candidate Generation     │ ──> output/candidate_pairs.tsv
│ (Person 1: Recall Ceiling Target > 96%)      │
└──────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Feature Extraction & LightGBM Trees │ ──> Fast pairwise scoring
│ (Person 2: Scores all candidate pairs)       │
└──────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Qwen 2.5 7B High-Precision Judge    │ ──> Resolves hard/borderline cases
│ (Person 3: Fine-tuned on AWS GPU)            │
└──────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ Stage 4: F_0.5 Threshold Gating & Validator  │ ──> output/matching_results.tsv
│ (All: Zero-error validation check)           │
└──────────────────────────────────────────────┘
```

---

## 5. Milestone & Delivery Timeline

| Milestone | Target Output | Lead |
| :--- | :--- | :--- |
| **Milestone 1: Foundations** | • Cleaned text pipeline<br>• Local validation split (50k records)<br>• Official validator working | Person 1 & Person 2 |
| **Milestone 2: Stage 1 Candidate Blocking** | • High-recall candidate generator<br>• `output/candidate_pairs.tsv` verified (>95% recall) | Person 1 |
| **Milestone 3: Fast Classifier (LightGBM)** | • Feature engineering module<br>• LightGBM baseline trained and evaluated locally ($F_{0.5} > 0.80$) | Person 2 |
| **Milestone 4: Cloud LLM Tuning (AWS)** | • EC2 `g5.xlarge` instance spun up<br>• `Qwen2.5-7B` QLoRA fine-tuning completed<br>• Hard ambiguous cases evaluated | Person 3 |
| **Milestone 5: Final Submission Package** | • `matching_results.tsv` + `candidate_pairs.tsv`<br>• `validate_submission.py` PASS<br>• `Documentation_template.md` filled | All (Team Leader signs off) |
