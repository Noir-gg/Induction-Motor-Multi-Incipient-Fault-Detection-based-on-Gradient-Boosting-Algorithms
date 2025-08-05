# Motor Fault Diagnosis via Hybrid Feature Extraction and Ensemble Learning

This repository provides an end‑to‑end Python pipeline for detecting **bearing** and **stator‑winding** faults in three‑phase induction motors using stator‑current signals.

## Project Motivation
The codebase accompanies our peer‑reviewed study published in *IEEE Transactions on Industrial Electronics* (DOI: 10.1109/TIE.2025.11027235). It demonstrates how classical signal‑processing and modern machine‑learning techniques can be fused to build a lightweight diagnostic tool suitable for embedded deployment.

## Repository Structure
| File | Purpose |
|------|---------|
| `feature_extraction.py` | Statistical feature generation and SMOTE balancing of raw current data |
| `feature_extraction_wavelet.py` | Multi‑wavelet analysis with MI‑based feature selection |
| `gridsearch.py` | Hyper‑parameter optimisation for several ensemble classifiers |
| `main.py` | Training, cross‑validation and visual analytics (ROC, confusion matrices) |
| `testing.py` | Quick data visualisation / sanity checks |

## Data Flow
1. **Input**: Excel files containing phase‑current time‑series (`datasets/*.xlsx`).
2. **Pre‑processing**  
   * Class balancing with *SMOTE*  
   * Per‑row statistics (mean, skew, variance, FFT peaks)
3. **Wavelet Feature Bank**  
   * `pywt.dwt` on three wavelet families (`sym`, `db`, `coif`)  
   * Information‑theoretic ranking → top‑10 % kept
4. **Model Selection** (`gridsearch.py`)  
   * CatBoost (default) + optional KNN / RF / XGBoost / LightGBM
5. **Training & Evaluation** (`main.py`)  
   * 5‑fold CV accuracy, precision, recall, macro‑F1  
   * Figures saved to `results/<model>/`
6. **Output**  
   * `results/metrics/*.csv` — aggregated metrics  
   * `results/<model>/*.png` — ROC & confusion matrix plots  
   * Trained model objects (optionally via `joblib`)

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your datasets
mkdir datasets
# add bearing_fault_dataset.xlsx, normal_dataset.xlsx, stator_winding_fault_dataset.xlsx

# 3. Feature engineering
python feature_extraction.py
python feature_extraction_wavelet.py

# 4. Model search & training
python gridsearch.py
python main.py
```

## Inputs
| Path | Description |
|------|-------------|
| `datasets/bearing_fault_dataset.xlsx` | Labeled fault currents |
| `datasets/normal_dataset.xlsx` | Healthy baseline currents |
| `datasets/stator_winding_fault_dataset.xlsx` | Winding short‑circuit currents |

## Outputs
| Artifact | Produced By | Description |
|----------|-------------|-------------|
| `datasets/dataset.csv` | `feature_extraction.py` | Balanced statistical‑feature table |
| `datasets/wavelet_undersamplified.csv` | `feature_extraction_wavelet.py` | Reduced wavelet‑feature table |
| `results/gridsearch/*.csv` | `gridsearch.py` | Parameter sweep log |
| `results/metrics/*.csv` | `main.py` | Final performance scores |
| `results/<model>/*.png` | `main.py` | ROC curves & confusion matrices |

## Reference
Please cite our paper if you use this code:

```
@article{hussain2025fault,
  title={Hybrid Feature Extraction and Ensemble Learning for Induction-Motor Fault Diagnosis},
  author={Hussain, R. and Khatri, S.},
  journal={IEEE Trans. Industrial Electronics},
  year={2025},
  doi={10.1109/TIE.2025.11027235}
}
```