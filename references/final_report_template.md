# AI6127 Final Report Template

> Keep the main content within 8 pages (excluding cover page and references).  
> Submit as PDF. One report per team.

## Cover Information
- **Project Title:**  
- **Team Members (Name + NTU Email):**
  - Member 1:
  - Member 2:
  - Member 3:

---

## Abstract (max 250 words)
**What to include:** problem motivation, core contribution, key method idea, main quantitative result, and one-line significance.

**Draft starter:**
This project studies robust protein language modeling under structural uncertainty. We extend ProSST with confidence-aware sequence-structure integration using residue-level confidence signals (e.g., pLDDT). We compare hard masking, soft weighting, and learned gating strategies in disentangled attention. On ProteinGym, our best variant improves robustness in low-confidence structure regimes while maintaining competitive overall performance against ProSST baselines. These results suggest that uncertainty-aware structure fusion is important for practical deployment of structure-aware PLMs.

---

## 1. Introduction
### 1.1 Problem and Motivation
- What task are you solving?
- Why is this important?
- Why are existing methods insufficient?

### 1.2 Challenges
- Data quality challenge (non-uniform structure confidence)
- Modeling challenge (how to fuse sequence + structure reliably)

### 1.3 Contributions
List 2-4 concrete contributions. Example:
1. We propose confidence-aware ProSST variants for structure reliability handling.  
2. We provide controlled ablations (hard/soft/learned gating).  
3. We show stratified robustness gains by confidence bins.

---

## 2. Related Work / Background
### 2.1 Protein Language Models
- Sequence-only PLMs (e.g., ESM family)
- Structure-aware PLMs (e.g., ProSST, SaProt)

### 2.2 Confidence and Uncertainty in Deep Models
- Briefly cover confidence-aware fusion or reliability-aware attention ideas.

### 2.3 Positioning of Our Work
- Clearly state how your method differs from and improves over ProSST.

---

## 3. Approach / Method
### 3.1 Baseline: ProSST
- One short paragraph describing structure tokens + disentangled attention.

### 3.2 Proposed Confidence-Aware ProSST
#### V1: Hard Confidence Masking
- Mask low-confidence structure tokens.

#### V2: Soft Confidence Weighting
- Down-weight structure-related attention terms by confidence score.

#### V3: Learned Confidence Gate (Main)
- Learn per-residue gate to balance sequence vs structure contribution.

### 3.3 Model Formulation (Optional equations)
- Define confidence feature.
- Show how attention terms are modified.

### 3.4 Implementation Details
- Architecture changes
- Hyperparameters specific to your method
- Any simplifications due to compute constraints

---

## 4. Experiments
### 4.1 Data
- Dataset name(s), source links, versions, split details
- Task definition and exact input/output format

### 4.2 Evaluation Metrics
- Primary metric (e.g., Spearman rho for ProteinGym)
- Secondary metrics (NDCG, Top-recall, etc.)
- Explain why each metric is used

### 4.3 Experimental Settings
- Hardware and runtime
- Training settings (batch size, LR, epochs/steps, optimizer)
- Reproducibility settings (seed, checkpoint selection)

### 4.4 Baselines
- ProSST original
- Sequence-only ablation
- Optional: SaProt / ESM-2

### 4.5 Main Results

**Table 1 placeholder: Overall performance**

| Model | Spearman (ProteinGym) | NDCG | Top-Recall | Notes |
|---|---:|---:|---:|---|
| ProSST baseline |  |  |  |  |
| CA-ProSST V1 (hard mask) |  |  |  |  |
| CA-ProSST V2 (soft weight) |  |  |  |  |
| CA-ProSST V3 (learned gate) |  |  |  |  |

**Commentary required by instructor:**  
- Are results better/worse than expected?  
- Why?  
- What do they imply about your method?

### 4.6 Ablation Studies
- Effect of threshold choices
- Effect of confidence feature type
- Component removal tests

### 4.7 Robustness by Confidence Strata

**Table 2 placeholder: Performance by pLDDT quartiles**

| Model | Q1 (low) | Q2 | Q3 | Q4 (high) |
|---|---:|---:|---:|---:|
| ProSST baseline |  |  |  |  |
| Best CA-ProSST |  |  |  |  |

### 4.8 Statistical Significance
- Bootstrap CI and/or paired testing
- Mention sample unit and number of bootstrap runs

---

## 5. Analysis (Qualitative)
This section is explicitly expected in the instruction.

### 5.1 Success Cases
- Show examples where confidence-aware modeling helps.

### 5.2 Failure Cases
- Analyze where your model still fails and why.

### 5.3 Mechanistic Insight (Optional)
- Attention visualization / gate values vs pLDDT

---

## 6. Conclusion
- Main findings
- What you learned
- Primary limitations
- Future work

---

## References
- Include all cited papers and datasets.

---

## Appendix (Optional)
Useful for extra results, visualizations, hyperparameter tables, additional examples.  
Do not rely on appendix for core grading points.

---

## Team Contributions (Required)
Provide 1-2 sentences per member + percentage contribution.

- **Member 1 (xx%)**:  
- **Member 2 (xx%)**:  
- **Member 3 (xx%)**:  

Percentages should sum to 100%.

---

## Final Checklist (Before PDF Submission)
- [ ] Main content within 8 pages  
- [ ] Abstract under 250 words  
- [ ] Contains: data, metrics, model settings, baselines, quantitative results, qualitative analysis  
- [ ] Results include commentary (expected/unexpected and why)  
- [ ] Conclusion includes limitations and future work  
- [ ] Team contributions included with percentages  
- [ ] References complete and consistent  
- [ ] Exported to PDF and uploaded to NTU Learn
