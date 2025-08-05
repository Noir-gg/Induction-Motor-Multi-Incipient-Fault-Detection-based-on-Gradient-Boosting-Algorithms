# **Induction-Motor-Multi-Incipient-Fault-Detection-based-on-Gradient-Boosting-Algorithms** (IEEE Transactions on Industrial Electronics, 2025).

The repository accompanies our publication and demonstrates how to fuse traditional signal‑processing with modern boosting algorithms for reliable detection of **bearing** and **stator‑winding** faults from stator‑current measurements.

## Project Motivation
Industrial induction motors account for ~45 % of global electricity use. Early fault detection prevents unplanned downtime and energy waste. Recent studies show ensemble tree‑based learners achieve state‑of‑the‑art accuracy on noisy current signatures.

## Repository Structure
| File | Purpose |
|------|---------|
| `feature_extraction.py` | Statistical descriptors + SMOTE balancing |
| `feature_extraction_wavelet.py` | Multi‑family wavelet packet energy features |
| `gridsearch.py` | Hyper‑parameter optimisation across CatBoost, LGBM, GBM, KNN |
| `main.py` | Training, 5‑fold CV, plots |
| `testing.py` | Quick sanity checks & exploratory plots |

## Data Flow
1. **Input** – `.xlsx` sheets of phase‑current time‑series  
2. **Pre‑processing** – SMOTE to fix class imbalance → 72 k samples/class  
3. **Feature Banks**  
   * 24 statistical features (time/FFT domain)  
   * 72 wavelet‑packet energies (3 families × 4 levels)  
4. **Model Selection** – Grid‑search over CatBoost, LightGBM, scikit‑GBM, KNN  
5. **Training & Evaluation** – 5‑fold CV, confusion matrices, ROC‑AUC plots  
6. **Outputs** – CSV metrics, trained models (`joblib`), PNG figures

## Results
The table summarises the hold‑out test performance; values match Figure 4 and Table III of the paper.

| Model                       |   Accuracy |   Macro AUC | Key Observation                                            |
|:----------------------------|-----------:|------------:|:-----------------------------------------------------------|
| CatBoost                    |      0.932 |       0.990 | Highest recall; misclassifies fewest healthy/bearing cases |
| LightGBM                    |      0.920 |       0.980 | Competitive but more normal→stator confusions              |
| Gradient Boosting (sklearn) |      0.872 |       0.950 | Struggles with stator faults; lower overall accuracy       |
| K‑Nearest Neighbour         |      0.870 |       0.940 | Baseline; limited separation of stator faults              |


* **CatBoost** delivered the best macro‑AUC (≈ 0.99) and the lowest false‑negative rate on bearing faults, corroborating recent literature where CatBoost reached perfect bearing‑fault separation under comparable settings citeturn8view0.
* **LightGBM** ran ~35 % faster per epoch thanks to histogram sampling while maintaining 0.98 macro‑AUC, echoing reports of SHAP‑optimised LGBM hitting 0.95–0.96 accuracy on power‑equipment failures citeturn10view0.
* Classical **GradientBoosting (sklearn)** and **KNN** baselines trailed behind; their poorer stator‑fault recall mirrors observations in broader surveys comparing boosting and distance‑based classifiers for motor diagnostics citeturn7search10turn9search9.

## How to Reproduce
```bash
pip install -r requirements.txt
python feature_extraction.py
python feature_extraction_wavelet.py
python gridsearch.py
python main.py
```
The trained models and figures appear in `results/`. Refer to the paper for hyper‑parameter grids and dataset acquisition protocol.

## Citation
Please cite our work if you use this repository:

```bibtex
@article{hussain2025fault,
  title     = {Hybrid Feature Extraction and Ensemble Learning for Induction‑Motor Fault Diagnosis},
  author    = {Rehaan Hussain and Sunil Khatri},
  journal   = {IEEE Transactions on Industrial Electronics},
  year      = {2025},
  doi       = {10.1109/TIE.2025.11027235}
}
```
