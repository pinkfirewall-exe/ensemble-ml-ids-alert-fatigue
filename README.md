# Ensemble ML Intrusion Detection with Confidence-Based SOC Triage

> Reducing alert fatigue in Security Operations Centres using ensemble machine learning,
> per-class evaluation, and a confidence threshold triage framework which is evaluated on
> CICIoT2023 (IoT) and UNSW-NB15 (enterprise) benchmark datasets.

---

## The Problem

SOC analysts are drowning in alerts. Signature-based IDS tools like Snort and Suricata generate thousands of notifications per shift. The majority are false positives. Analysts develop shortcuts, trust erodes, and real attacks get missed inside the noise.

Published ML research makes this worse by reporting **aggregate accuracy** : a single number that looks impressive but conceals how the model performs on individual attack types. A 99.6% accurate classifier can simultaneously miss 61% of web-based intrusion attempts. That failure is invisible until it is too late.

This project addresses three specific gaps:

- **Per-class evaluation** - break accuracy down to FPR and FNR for every attack category
- **Confidence-based routing** - use model uncertainty to route predictions to the right level of analyst scrutiny
- **Explainability** - SHAP feature-level explanations for every prediction, EU AI Act Article 13 compliant

---

## What This Project Does

Three ensemble classifiers: Random Forest, XGBoost, and a soft Voting Classifier - are trained and evaluated on two public IDS benchmarks under class weighting and SMOTE.

A **three-tier confidence triage framework** then routes each prediction based on how certain the model is:

| Tier | Confidence | Observed Accuracy | SOC Action |
|---|---|---|---|
| **Tier 1 - Auto-action** | ≥ 90% | ≥ 96% | Automated response, no analyst review |
| **Tier 2 - Second Review** | 50–90% | 40–92% | Second analyst verifies before action |
| **Tier 3 - Escalation** | < 50% | < 50% | Full manual investigation |

> **Important:** The threshold values of 90% and 50% are theoretically grounded
> starting points, not universal settings. Every organisation should calibrate
> these against their own traffic characteristics, analyst team capacity, and
> acceptable false positive/negative rates before production deployment.
> A high-volume financial SOC and a small healthcare security team have very
> different operational constraints, and the right thresholds will differ accordingly.

SHAP (TreeSHAP) provides feature-level explanations for every prediction for both global importance rankings and instance-level waterfall plots for misclassified predictions.

---

## Datasets

| Dataset | Year | Environment | Training Records | Classes | Max Imbalance |
|---|---|---|---|---|---|
| [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | 2023 | IoT (105 devices) | 5,357,406 | 8 | 6,058:1 |
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 2015 | Enterprise | 96,822 | 10 | 431:1 |

Both datasets are publicly available. Download links above.

---

## Key Results

### CICIoT2023

| Model | Accuracy | Macro F1 | Workload Reduction | Tier 1 Accuracy |
|---|---|---|---|---|
| Random Forest | 99.61% | 87.10% | 98.74% | 99.98% |
| XGBoost | 99.40% | 77.75% | 98.88% | 99.99% |
| Voting Classifier | 99.62% | 86.12% | 98.68% | 100.00% |

### UNSW-NB15

| Model | Accuracy | Macro F1 | Workload Reduction | Tier 1 Accuracy |
|---|---|---|---|---|
| Random Forest | 76.12% | 52.30% | 67.81% | 98.85% |
| XGBoost | 62.90% | 48.47% | 65.61% | 99.56% |
| Voting Classifier | 67.86% | 51.60% | 61.81% | 99.10% |

### Why aggregate accuracy is not enough

Random Forest on CICIoT2023: **99.61% accuracy** - looks perfect.
The same model misses **61.09% of Web attacks** and **44.48% of BruteForce attacks**.
Both figures describe the same classifier on the same test data. Only per-class evaluation reveals the gap.

### SMOTE vs Class Weighting (UNSW-NB15)

Class weighting outperforms SMOTE for Random Forest by **+5.85 percentage points** in accuracy. UNSW-NB15's overlapping class boundaries cause SMOTE to generate synthetic instances in adjacent class regions, making classification harder. Class weighting adjusts the loss function directly without touching the data — and it scales to any dataset size.

---

## SHAP Feature Importance

**Zero features overlap** between the two datasets' top-15 SHAP rankings.

| Rank | UNSW-NB15 (RF) | CICIoT2023 (XGB) |
|---|---|---|
| 1 | sbytes | IAT |
| 2 | service_- | Number |
| 3 | dbytes | Magnitue |
| 4 | smean | Protocol Type |
| 5 | tcprtt | rst_count |

A model trained on enterprise traffic and deployed in an IoT environment bases its predictions on features that carry **no signal** in the new environment. SHAP analysis in the deployment environment - not just the training environment - is the mechanism that catches this before production.

---

## Repository Structure

```
ensemble-ml-ids-alert-fatigue/
├── README.md
├── src/
│   ├── preprocessing.py          ← data cleaning, encoding, scaling
│   ├── model_training.py         ← RF, XGBoost, Voting Classifier training
│   ├── evaluation.py             ← per-class FPR, FNR, macro F1
│   ├── triage_framework.py       ← confidence threshold triage system
│   └── shap_analysis.py          ← global, beeswarm, waterfall SHAP plots
├── figures/
│   ├── fig_methodology_pipeline.png
│   ├── fig_triage_tiers.png
│   ├── fig_ciciot_cm_rf.png
│   ├── fig_unsw_cm_rf.png
│   ├── fig_ciciot_confidence_perclass.png
│   ├── fig_ciciot_threshold.png
│   ├── fig_unsw_shap_global.png
│   ├── fig_ciciot_shap_global.png
│   ├── fig_ciciot_shap_rf_vs_xgb.png
│   ├── fig_ciciot_shap_beeswarm_ddos.png
│   ├── fig_ciciot_shap_beeswarm_benign.png
│   ├── fig_unsw_shap_beeswarm_normal.png
│   ├── fig_unsw_shap_beeswarm_dos.png
│   ├── fig_ciciot_shap_waterfall.png
│   └── fig_unsw_shap_waterfall.png
└── report/
    └── thesis.pdf                ← full write-up (38 pages, IEEE format)
```

---

## Setup and Usage

```bash
git clone https://github.com/pinkfirewall-exe/ensemble-ml-ids-alert-fatigue.git
cd ensemble-ml-ids-alert-fatigue
pip install -r requirements.txt
```

**Step 1 — Preprocess the data**
```bash
python src/preprocessing.py --dataset ciciot2023 --input data/raw/ --output data/processed/
python src/preprocessing.py --dataset unsw_nb15 --input data/raw/ --output data/processed/
```

**Step 2 — Train the models**
```bash
python src/model_training.py --dataset ciciot2023 --balancing class_weight
python src/model_training.py --dataset unsw_nb15 --balancing class_weight
python src/model_training.py --dataset unsw_nb15 --balancing smote
```

**Step 3 — Evaluate per-class FPR and FNR**
```bash
python src/evaluation.py --dataset ciciot2023
python src/evaluation.py --dataset unsw_nb15
```

**Step 4 — Run the triage framework**
```bash
python src/triage_framework.py --dataset ciciot2023 --tier1-threshold 0.90 --tier3-threshold 0.50
python src/triage_framework.py --dataset unsw_nb15 --tier1-threshold 0.90 --tier3-threshold 0.50
```

**Step 5 — SHAP analysis**
```bash
python src/shap_analysis.py --dataset ciciot2023 --sample-size 1000
python src/shap_analysis.py --dataset unsw_nb15 --sample-size 1000
```

---

## Requirements

```
scikit-learn>=1.4
xgboost>=2.0
imbalanced-learn>=0.12
shap>=0.44
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
```

> Experiments were run on Kaggle Cloud (~13GB RAM, 4 CPU cores, Python 3.12).
> CICIoT2023 cross-validation used a stratified 20% subsample due to memory constraints.
> All random seeds fixed at 42 for reproducibility.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-337733?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-0.44-FF0000?style=flat)
![Kaggle](https://img.shields.io/badge/Experiments-Kaggle_Cloud-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## Related Projects

- [snort-ids-evasion-detection](https://github.com/pinkfirewall-exe/snort-ids-evasion-detection) — Snort IDS evasion testing and custom rule development
- [tpot-honeypot-analysis](https://github.com/pinkfirewall-exe/tpot-honeypot-analysis) — T-Pot honeypot deployment and attack analysis
- [cybersecurity-portfolio](https://github.com/pinkfirewall-exe/cybersecurity-portfolio) — Security practice writeups and lab projects

---

## References

- Geifman & El-Yaniv (2017) — Selective classification for deep neural networks. *NeurIPS*
- Hendrickx et al. (2024) — Machine learning with a reject option. *Machine Learning*
- Sommer & Paxson (2010) — Outside the closed world: On using ML for network IDS. *IEEE S&P*
- Neto et al. (2023) — CICIoT2023: A real-time dataset for IoT cybersecurity. *Sensors*
- Moustafa & Slay (2015) — UNSW-NB15: A comprehensive dataset for network intrusion detection. *MilCIS*
- Chen & Guestrin (2016) — XGBoost: A scalable tree boosting system. *KDD*
- Lundberg & Lee (2017) — A unified approach to interpreting model predictions. *NeurIPS*
