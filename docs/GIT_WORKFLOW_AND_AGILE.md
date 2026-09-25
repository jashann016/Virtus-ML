# Team Git Workflow & Agile Operating Guidelines

## 1. Core Principles & Philosophy
To win this competition as a 3-person team, we must operate like an elite engineering unit. Speed, reproducibility, and avoiding code collisions are paramount.

* **Single Source of Truth:** `main` always reflects stable, validated, submission-ready code.
* **Strict Rule on `main`:** **ABSOLUTELY NO DIRECT COMMITS TO `main`!** Direct pushes to `main` are strictly blocked.
* **Leader Gated Integration:** All changes enter `main` only through reviewed, tested Pull Requests (PRs) approved by the Team Leader.
* **Fast Async Communication:** Use our team WhatsApp group for instantaneous PR alerts and blockers.

---

## 2. The 3-Tier Branching System

```
[main] --------------------------------------------● (Submission Tag / Leader Only)
   ▲                                              │
   │ PR + Review + Validator Check Passed        │
   │                                              ▼
[dev] ------------------------------------------------ (Integration Branch)
   ▲                    ▲                    ▲
   │ PR + Rebase        │ PR + Rebase        │ PR + Rebase
   │                    │                    │
[feat/p1-blocking]   [feat/p2-lightgbm]   [feat/p3-aws-qwen]
(Person 1: Data)     (Person 2: ML)       (Person 3: Cloud)
```

### Branch Definitions:
1. **`main` (Production / Leaderboard Ready):**
   * Contains only code that has passed local validation (`validate_submission.py`) and is ready to generate official leaderboard files.
   * Merges into `main` happen **only from `dev`**, and **only by the Team Leader**.
2. **`dev` (Active Integration):**
   * Where all tested features come together.
   * All members base their work off `dev` and open PRs targeting `dev`.
3. **`feat/<person>-<feature-name>` (Individual Feature Branches):**
   * Every team member works strictly in their own feature branch.
   * Examples:
     * `feat/p1-text-cleaning`
     * `feat/p1-candidate-blocking`
     * `feat/p2-address-features`
     * `feat/p2-lightgbm-model`
     * `feat/p3-aws-setup`
     * `feat/p3-qwen-finetuning`

---

## 3. Daily Git Workflow (Step-by-Step)

### Step A: Starting a New Task
Always branch out from the latest `dev`:
```bash
# 1. Switch to dev and pull latest changes
git checkout dev
git pull origin dev

# 2. Create your dedicated feature branch
git checkout -b feat/p1-blocking-engine
```

### Step B: Committing Your Work
Make small, atomic commits with clear commit messages:
```bash
git add src/blocking.py
git commit -m "feat(blocking): add character n-gram TF-IDF candidate generation"
```

### Step C: Rebasing Before Creating a Pull Request
Before submitting your work, ensure your branch incorporates any updates your teammates pushed to `dev`:
```bash
# 1. Fetch latest dev
git checkout dev
git pull origin dev

# 2. Switch back to your feature branch and rebase
git checkout feat/p1-blocking-engine
git rebase dev

# (If merge conflicts arise, resolve them cleanly, then git rebase --continue)

# 3. Push your feature branch to GitHub
git push -u origin feat/p1-blocking-engine
```

---

## 4. Pull Request (PR) & WhatsApp Review Protocol

Once your code is pushed:

### 1. Open a Pull Request on GitHub:
* **Base branch:** `dev`
* **Compare branch:** `feat/<your-feature-name>`
* **Title:** Clear and descriptive (e.g., `[Person 1] Implement Stage 1 Blocking & Candidate Generation`)
* **Description:** Mention:
  * What changes were made.
  * Local test results (e.g., "Candidate recall on validation set: 96.2%").
  * Output files verified (if applicable).

### 2. WhatsApp Group Alert:
Immediately after creating the PR, copy the PR link and send an alert in the team **WhatsApp group** using this exact template:

```text
🚨 PR Ready for Review!
• Author: [Your Name]
• Feature: Stage 1 Candidate Blocking Engine
• Target: dev
• PR Link: https://github.com/<org>/<repo>/pull/XX
• Summary: Implemented TF-IDF n-gram search with 96.2% recall on validation split. Verified memory usage < 4GB.
👉 @TeamLeader Please review and merge into dev!
```

### 3. Review & Approval:
* The Team Leader (or designated reviewer) reviews the code, checks for formatting, runs a quick sanity check, and clicks **Approve & Merge** (Squash and Merge or Rebase and Merge).
* The feature branch can then be deleted.

---

## 5. Merging from `dev` to `main` (Leader Only)

When a complete milestone is reached:
1. Team Leader verifies that `output/matching_results.tsv` and `output/candidate_pairs.tsv` pass:
   ```bash
   python3 utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir dataset/test
   ```
2. Team Leader creates a PR from `dev` to `main`.
3. Merges into `main` and tags the release (e.g., `git tag v1.0-submission-baseline`).

---

## 6. Agile Team Cadence

| Cadence | When | Purpose | Output |
| :--- | :--- | :--- | :--- |
| **Morning Sync** | 10:00 AM (15 mins via WP Call / Voice Notes) | • What did you finish yesterday?<br>• What is your goal today?<br>• Are you blocked by anything? | Clear daily task alignment |
| **Midday Integration** | 3:00 PM | • Rebase feature branches on `dev`<br>• Resolve any interface mismatches | Friction-free merging |
| **Evening Review** | 8:00 PM | • Review local $F_{0.5}$ score progression<br>• Validate submission candidates<br>• Plan next sprint | Measurable progress tracking |
