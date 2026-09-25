# Person 3: AWS Cloud GPU, 7B LLM Tuning & Documentation Lead

## 1. Role Mission & Strategic Importance
* **Your Title:** Cloud GPU & LLM Lead + Documentation Owner (The Judge & Scribe)
* **Your Core Mission:** Manage the AWS GPU infrastructure, fine-tune **Qwen 2.5 7B** using QLoRA to resolve hard edge cases, and author the official **`Documentation_template.md`**.
* **Why You Are Critical:**
  * **You hold the key to the hard cases:** When name spellings look completely different (DBAs, trade names, foreign scripts), your fine-tuned 7B model uses deep reasoning to make the call.
  * **You own the final submission review:** Even if a model scores #1 on the leaderboard, Amazon disqualifies teams if their `Documentation_template.md` or code reproduction is incomplete. You ensure the final package is prize-worthy.

---

## 2. Your Detailed Responsibilities & Tasks

### Task 1: AWS Environment Setup & Cost Management
You have the **AWS Builder Pro Plan ($549 credits)**.
* **Instance Type:** `g5.xlarge` (1x NVIDIA A10G with 24 GB VRAM, 4 vCPUs, 16 GB RAM).
* **Cost:** ~$1.006 / hour (training will take ~1.5 hours $\approx \$1.50$ total!).
* **Operating System / AMI:** Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04).
* **Safety Rule:** Always shut down / stop the EC2 instance immediately when training/inference completes to avoid burning credits.

### Task 2: Qwen 2.5 7B QLoRA Fine-Tuning (`aws/qwen_finetune.py`)
* **Base Model:** `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0 license).
* **Quantization:** 4-bit NormalFloat (`bitsandbytes`) $\rightarrow$ uses only ~6 GB of VRAM.
* **Adapter (PEFT / LoRA):**
  * Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
  * LoRA Rank: $r = 16$, Alpha: $\alpha = 32$.
  * Trainable parameters: ~25 Million (less than 0.5% of the model).
* **Training Dataset:** 50,000 curated hard pairs:
  * Difficult true matches (different trade names, website domains, Hindi/French transliterations).
  * Tricky negative pairs (shops on the same street with similar names).
* **Output:** Saves the lightweight ~100 MB LoRA adapter (`adapter_model.safetensors`).

### Task 3: Borderline Case Inference (`aws/qwen_infer_borderline.py`)
* Receive the ambiguous candidate pairs flagged by Person 2 ($0.35 \le P \le 0.88$).
* Run batch inference with Qwen on the AWS GPU:
  ```json
  {"is_match": true, "confidence": 0.95, "reason": "Identical building 85 and street in Ticonderoga"}
  ```
* Send refined probability adjustments back to Person 1 and Person 2.

### Task 4: Competition Methodology Documentation (`Documentation_template.md`)
Amazon requires every team to submit a filled `Documentation_template.md` in the final zip archive. Top teams are audited on this write-up!
* **Section 1: Approach Overview:** Architecture diagram (Blocking $\rightarrow$ LightGBM $\rightarrow$ Qwen 7B $\rightarrow$ $F_{0.5}$ Gating).
* **Section 2: Candidate Generation / Blocking:** Explain sublinear TF-IDF character n-grams and country partitioning.
* **Section 3: Model Architecture & Features:** Detail the string similarity metrics, number overlap logic, and Qwen QLoRA adapter.
* **Section 4: Validation & $F_{0.5}$ Tuning:** Document local validation split results, threshold selection, and singleton handling.

---

## 3. Git Branch & Deliverables

* **Working Branch:** `feat/p3-aws-qwen-doc`
* **Target PR Branch:** `dev`
* **Primary Code & Doc Files:**
  * `aws/qwen_finetune.py`
  * `aws/qwen_infer_borderline.py`
  * `aws/setup_aws_instance.sh`
  * `Documentation_template.md`
* **Deliverables:**
  * Trained LoRA adapter (`adapter_model.safetensors`).
  * Predictions on ambiguous pairs (`output/borderline_predictions.tsv`).
  * Completely filled `Documentation_template.md`.

---

## 4. Daily Checklist & PR Protocol
1. Pull latest `dev` and rebase your branch: `git checkout dev && git pull && git checkout feat/p3-aws-qwen-doc && git rebase dev`.
2. Ensure AWS GPU instance is stopped whenever not in active use.
3. Push branch and open PR to `dev`.
4. Post PR link in the WhatsApp group for Team Leader approval.
