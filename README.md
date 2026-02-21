<div align="center">

# HCN-TA: Hierarchical Capsule Network with Temporal Attention for a Generalizable Approach to Audio Deepfake Detection

[![Paper](https://img.shields.io/badge/Paper-ACM%20SAC%202025-blue.svg)](https://doi.org/10.1145/3672608.3707761)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/ACM%20SAC-2025-purple.svg)](#)

**Official implementation of the paper accepted at the 40th ACM/SIGAPP Symposium on Applied Computing (SAC '25)**

[Taiba Majid Wani](mailto:majid@diag.uniroma1.it)<sup>1</sup>&nbsp;&nbsp;
[Madleen Uceker](mailto:mauecker@uni-osnabrueck.de)<sup>2</sup>&nbsp;&nbsp;
[Farooq Ahmad Wani](mailto:wani@diag.uniroma1.it)<sup>1</sup>&nbsp;&nbsp;
[Irene Amerini](mailto:amerini@diag.uniroma1.it)<sup>1</sup>

<sup>1</sup>Sapienza University of Rome, Italy &nbsp;&nbsp; <sup>2</sup>Osnabrück University, Germany

<br>

<img src="assets/architecture.png" alt="HCN-TA Architecture" width="850"/>

</div>

---

## 📋 Abstract

The increasing prevalence of audio deepfakes has raised serious concerns due to their potential misuse in identity theft, disinformation, and the compromise of voice authentication systems. We introduce **HCN-TA (Hierarchical Capsule Network with Temporal Attention)**, a novel architecture specifically designed for **scalable and generalizable** audio deepfake detection. The **hierarchical capsule networks** capture local and global audio patterns, while the **multi-resolution temporal attention** focuses on key segments with likely deepfake artifacts. **Temporal locality awareness** ensures prioritization of critical, rapidly changing regions.

### 🏆 Key Results

| Dataset | Accuracy | F1-Score | EER (%) |
|---------|----------|----------|---------|
| **ASVspoof 2019 (LA)** | **98.5%** | **97.9%** | **0.42** |
| **FoR** | **99.2%** | **98.95%** | **0.11** |
| **ASVspoof 2021** (cross-dataset) | 96.8% | 95.75% | 1.45 |

---

## 🔥 Highlights

- **Hierarchical Capsule Network (HCN)** — Lower-level capsules capture local patterns (phonetic transitions); higher-level capsules model global patterns (prosody, sentence coherence)
- **Multi-Resolution Temporal Attention** — Captures anomalies across different time scales, focusing on segments with likely deepfake artifacts
- **Temporal Locality Awareness** — Prioritizes rapidly changing audio regions where manipulations are most detectable
- **Cross-Dataset Generalization** — Validated on ASVspoof 2019, FoR, and ASVspoof 2021 (unseen attacks)
- **ResNet50 Backbone** — Efficient feature extraction from mel-spectrograms

---

## 🏗️ Architecture

```
Audio Input
    │
    ▼
┌──────────────────────┐
│   Preprocessing       │  Resample (16kHz) → Spectral Subtraction → Silence Removal
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Mel-Spectrogram     │  STFT + Mel Filter Bank → S(t, f)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   ResNet50 Backbone   │  F_res(t,f) = σ(W * S(t,f) + b) ∈ R^{T×F×C}
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Hierarchical        │  Lower Capsules: local patterns (phonetic transitions)
│   Capsule Network     │  Higher Capsules: global patterns (prosody, coherence)
│   (HCN)               │  Dynamic Routing between layers
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Multi-Resolution    │  e_t^(r) = W_e^(r) · C_low(t)
│   Temporal Attention  │
│   +                   │  L_t = ||C_low(t) - C_low(t-1)||  (locality score)
│   Temporal Locality   │
│   Awareness           │  α_t = softmax(e_t · L_t)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Classification      │  Class = argmax(||v_real||, ||v_fake||)
│   + Margin Loss       │  L_k = T_k·max(0, m+ - ||v_k||)² + λ(1-T_k)·max(0, ||v_k|| - m-)²
└──────────────────────┘
```

---

## 📁 Project Structure

```
HCN-TA/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── asvspoof2019.yaml
│   └── for_dataset.yaml
├── datasets/
│   ├── __init__.py
│   ├── asvspoof2019.py
│   ├── for_dataset.py
│   └── preprocessing.py
├── models/
│   ├── __init__.py
│   ├── hcn_ta.py                 # Full HCN-TA architecture
│   ├── resnet_backbone.py        # ResNet50 feature extractor
│   ├── hierarchical_capsule.py   # Hierarchical capsule network
│   ├── temporal_attention.py     # Multi-resolution temporal attention + locality
│   ├── capsule_layers.py         # Primary & higher capsule layers
│   └── losses.py                 # Margin loss
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── logger.py
│   └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── assets/
└── docs/
    └── RESULTS.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/HCN-TA.git
cd HCN-TA

conda create -n hcnta python=3.9 -y
conda activate hcnta

pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### ASVspoof 2019 (LA) — Primary Evaluation
```
data/ASVspoof2019/LA/
├── ASVspoof2019_LA_train/
├── ASVspoof2019_LA_dev/
├── ASVspoof2019_LA_eval/
└── ASVspoof2019_LA_cm_protocols/
```

### FoR Dataset — Primary Evaluation
```
data/FoR/
├── for-original/
├── for-norm/
├── for-2seconds/
└── for-rerecorded/
```

### ASVspoof 2021 — Cross-Dataset Evaluation
```
data/ASVspoof2021/LA/
└── ...
```

---

## 🚀 Training

```bash
# ASVspoof 2019
python scripts/train.py --config configs/asvspoof2019.yaml --data_dir data/ASVspoof2019/LA

# FoR Dataset
python scripts/train.py --config configs/for_dataset.yaml --data_dir data/FoR
```

## 📈 Evaluation

```bash
# Standard evaluation
python scripts/evaluate.py --checkpoint experiments/best_model.pth --data_dir data/ASVspoof2019/LA

# Cross-dataset evaluation (ASVspoof 2021)
python scripts/evaluate.py --checkpoint experiments/best_model.pth --data_dir data/ASVspoof2021/LA --dataset asvspoof2021

# Single file inference
python scripts/inference.py --checkpoint experiments/best_model.pth --audio_path path/to/audio.wav
```

---

## 📊 Results

### Performance Across Datasets (Table 1)

| Metric | ASVspoof 2019 (LA) | FoR Dataset | ASVspoof 2021 (cross) |
|--------|-------------------|-------------|----------------------|
| Accuracy | **98.5%** | **99.2%** | 96.8% |
| Precision | 97.8% | 98.9% | 95.5% |
| Recall | 98.0% | 99.0% | 96.0% |
| F1-Score | 97.9% | 98.95% | 95.75% |
| EER (%) | **0.42** | **0.11** | 1.45 |

### Ablation Study — ASVspoof 2019 (Table 2)

| Component | Accuracy | EER (%) |
|-----------|----------|---------|
| HCN without Temporal Attention | 95.2% | 3.15 |
| HCN without Temporal Locality Awareness | 96.1% | 2.75 |
| **HCN-TA (Full Model)** | **98.5%** | **0.42** |

### Ablation Study — FoR (Table 3)

| Component | Accuracy | EER (%) |
|-----------|----------|---------|
| HCN without Temporal Attention | 96.7% | 2.90 |
| HCN without Temporal Locality Awareness | 97.5% | 2.25 |
| **HCN-TA (Full Model)** | **99.2%** | **0.11** |

### Comparison with State-of-the-Art (Table 4)

| Method | Model | Dataset | EER (%) |
|--------|-------|---------|---------|
| Luo et al. | Capsule Networks | ASVspoof 2019 | 1.07 |
| Mao et al. | CQCC Capsule | ASVspoof 2019 | 5.09 |
| Wani & Amerini | cCNN | FoR | 3.20 |
| Khochare et al. | TCN | FoR | 8.00 |
| **Proposed** | **HCN-TA** | **ASVspoof 2019** | **0.42** |
| **Proposed** | **HCN-TA** | **FoR** | **0.11** |

---

## 📜 Citation

```bibtex
@inproceedings{wani2025hcnta,
    title     = {HCN-TA: Hierarchical Capsule Network with Temporal Attention for a Generalizable Approach to Audio Deepfake Detection},
    author    = {Wani, Taiba Majid and Uceker, Madleen and Wani, Farooq Ahmad and Amerini, Irene},
    booktitle = {The 40th ACM/SIGAPP Symposium on Applied Computing (SAC '25)},
    year      = {2025},
    pages     = {775--777},
    doi       = {10.1145/3672608.3707761},
    publisher = {ACM}
}
```

---

## 🙏 Acknowledgments

This study has been partially supported by:
- **SERICS** (PE00000014) under the MUR National Recovery and Resilience Plan funded by the European Union – NextGenerationEU
- **Sapienza University of Rome** project 2022–2024 "EV2" (003 009 22)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

</div>
