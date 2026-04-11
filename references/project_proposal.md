# Project Proposal

## Title
Confidence-Aware ProSST: Improving Robustness of Structure-Aware Protein Language Models Under Structural Uncertainty

## 1) Background and Motivation
Protein language models (PLMs) have achieved strong performance in mutation effect prediction and protein function tasks. ProSST (NeurIPS 2024) improves over sequence-only PLMs by incorporating structure tokens and sequence-structure disentangled attention.  
However, structure quality is not uniform across proteins and residues (e.g., low-confidence AlphaFold regions). When structure is noisy, directly injecting structure tokens can hurt predictions.

This project proposes a confidence-aware extension of ProSST that explicitly models structural reliability during training and inference, with the goal of improving robustness and generalization.

## 2) Problem Statement
Current structure-aware PLMs treat all structure tokens similarly, even though structural confidence varies by residue.

**Research problem:** How can we integrate residue-level structural confidence (e.g., pLDDT) into ProSST so that the model uses structure when reliable and falls back to sequence when structure is uncertain?

## 3) Research Questions
- **RQ1:** Does confidence-aware structure integration improve zero-shot mutation effect prediction over vanilla ProSST?
- **RQ2:** Are gains concentrated in proteins/residues with low-to-medium structural confidence?
- **RQ3:** Which strategy is most effective: hard masking, soft weighting, or learned confidence gating?

## 4) Proposed Method
We build **CA-ProSST (Confidence-Aware ProSST)** with three variants:

- **V1: Hard Confidence Masking**
  - Mask structure tokens for residues below a confidence threshold.
- **V2: Soft Confidence Weighting**
  - Scale structure-related attention terms by normalized confidence scores.
- **V3: Learned Confidence Gate (main model)**
  - Add a lightweight gating module that learns how much to use structure vs sequence per residue:
  - `gate = sigmoid(MLP([residue_hidden, confidence_feature]))`
  - `structure_contribution = gate * structure_attention_component`

### Confidence Features
- Primary: per-residue pLDDT (binned token or normalized scalar).
- Optional: missing-structure indicator and local confidence statistics.

### Integration Point
Modify ProSST disentangled attention terms involving structure (`R2S`, `S2R`) using confidence-aware scaling/gating.

## 5) Novelty and Contribution
- Introduces uncertainty-aware sequence-structure fusion for protein language modeling.
- Provides a systematic comparison of confidence-integration strategies.
- Delivers practical guidance for deployment when predicted structures are imperfect.
- Adds robustness analysis by confidence strata.

## 6) Experimental Plan

### Datasets
- **Primary:** ProteinGym (zero-shot mutation effect prediction)
- **Secondary (optional):** DeepLoc or Thermostability for supervised validation
- Structure source: AlphaFoldDB (with pLDDT), optionally ESMFold for stress tests

### Baselines
- ProSST (original)
- Sequence-only ProSST ablation (`K=0` equivalent)
- Optional external context baselines: SaProt, ESM-2

### Metrics
- ProteinGym: Spearman's rho (main), NDCG, Top-recall
- Supervised tasks: Accuracy or F1-max (task-dependent)

### Analysis
- Overall performance comparison
- Performance by pLDDT quartiles
- Sensitivity to confidence thresholds
- Ablations: no confidence signal vs hard/soft/learned strategies
- Attention/case-study analysis (optional)

### Statistical Significance
- Bootstrap confidence intervals across proteins/assays
- Paired testing against ProSST baseline

## 7) Expected Outcomes
- Confidence-aware ProSST improves robustness in low-confidence structure regimes.
- Learned gating provides best tradeoff between robustness and overall accuracy.
- Results provide actionable recommendations on when to trust structure cues.

## 8) Risks and Mitigation
- **Compute constraints:** Use ProSST checkpoint adaptation instead of full pretraining.
- **Small overall gains:** Emphasize stratified robustness gains and significance tests.
- **Data alignment errors:** Add strict residue indexing and sanity-check scripts.

## 9) Timeline (6 Weeks)
- **Week 1:** Reproduce baseline pipeline and set up datasets.
- **Week 2:** Implement V1 (hard masking) and run sanity checks.
- **Week 3:** Implement V2 (soft weighting) and pilot experiments.
- **Week 4:** Implement V3 (learned gate) and ablation experiments.
- **Week 5:** Full evaluation, statistical testing, and error analysis.
- **Week 6:** Final report, figures/tables, and presentation prep.

## 10) Deliverables
- CA-ProSST implementation (all variants)
- Reproducible scripts and config files
- Result tables and confidence-stratified plots
- Final report and presentation slides

## 11) Short Abstract
This project proposes Confidence-Aware ProSST (CA-ProSST), an extension of ProSST that integrates residue-level structural confidence into sequence-structure attention. While structure-aware PLMs improve protein prediction, they are vulnerable to low-confidence predicted structures. We evaluate hard masking, soft weighting, and learned gating mechanisms to control structure usage based on confidence signals (e.g., pLDDT). Experiments on ProteinGym assess zero-shot mutation effect prediction overall and across confidence strata. We expect confidence-aware fusion to improve robustness and deliver statistically significant gains in low-confidence regimes while preserving strong overall performance.
