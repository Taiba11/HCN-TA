# Detailed Experimental Results

## Performance Metrics (Table 1)

| Metric | ASVspoof 2019 (LA) | FoR Dataset | ASVspoof 2021 (cross) |
|--------|-------------------|-------------|----------------------|
| Accuracy | **98.5%** | **99.2%** | 96.8% |
| Precision | 97.8% | 98.9% | 95.5% |
| Recall | 98.0% | 99.0% | 96.0% |
| F1-Score | 97.9% | 98.95% | 95.75% |
| EER (%) | **0.42** | **0.11** | 1.45 |

---

## Ablation Study — ASVspoof 2019 LA (Table 2)

| Component | Accuracy | EER (%) |
|-----------|----------|---------|
| HCN without Temporal Attention | 95.2% | 3.15 |
| HCN without Temporal Locality Awareness | 96.1% | 2.75 |
| **HCN-TA (Full Model)** | **98.5%** | **0.42** |

**Key Finding:** Temporal attention contributes the largest accuracy gain (+3.3%), followed by locality awareness (+2.4%).

---

## Ablation Study — FoR Dataset (Table 3)

| Component | Accuracy | EER (%) |
|-----------|----------|---------|
| HCN without Temporal Attention | 96.7% | 2.90 |
| HCN without Temporal Locality Awareness | 97.5% | 2.25 |
| **HCN-TA (Full Model)** | **99.2%** | **0.11** |

**Key Finding:** All three components (HCN, temporal attention, locality awareness) are essential. Removing any single component degrades EER by 20–26× on the FoR dataset.

---

## Comparison with State-of-the-Art (Table 4)

| Study | Model | Dataset | EER (%) |
|-------|-------|---------|---------|
| Luo et al. (ICASSP 2021) | Capsule Networks | ASVspoof 2019 | 1.07 |
| Mao et al. (FCS 2021) | CQCC Capsule | ASVspoof 2019 | 5.09 |
| Wani & Amerini (ICIAP 2023) | cCNN | FoR | 3.20 |
| Khochare et al. (2021) | TCN | FoR | 8.00 |
| **Proposed** | **HCN-TA** | **ASVspoof 2019** | **0.42** |
| **Proposed** | **HCN-TA** | **FoR** | **0.11** |

---

## Cross-Dataset Generalization

HCN-TA trained on ASVspoof 2019 and evaluated on ASVspoof 2021 (unseen attacks) achieves **96.8% accuracy** and **1.45% EER**, demonstrating strong generalization to new attack types without retraining.
