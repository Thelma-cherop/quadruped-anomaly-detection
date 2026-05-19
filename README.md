
***
# SLAM Robot — ML Health Monitor (Anomaly Detection)
streamlit website:https://quadruped-anomaly-detection-wg8myjqytetkp5bpgzr7gi.streamlit.app/

## Overview
This repository contains the ML component of a SLAM robot project: a health-monitoring model that inspects time-series sensor readings and flags windows as "normal" or "anomaly". In a real SLAM robot the ML part acts as the brain that watches sensor streams (IMU, joint angles, motor torque, foot force, pressure, temperature, vibration, etc.) and answers: "Is the robot moving normally or is something wrong?" This project uses the SKAB dataset (sensor logs) to train and evaluate supervised and unsupervised anomaly detectors.

## Key goals
- Load and merge raw CSV sensor logs.
- Perform EDA and preprocessing.
- Create labeled windows using sliding-window segmentation.
- Train supervised (Random Forest, XGBoost) and unsupervised (Isolation Forest, Autoencoder) models.
- Evaluate models using precision/recall/F1 and confusion matrices.
- Deploy model and display health score on a Streamlit dashboard.

## Prerequisites
- Python 3.9+
- Install required packages:
  - numpy, pandas, scikit-learn, xgboost, tensorflow or keras, matplotlib, seaborn, joblib, streamlit
- Hardware: CPU sufficient for small experiments; GPU recommended for deep models.
- Dataset: SKAB sensor CSV folders (anomaly-free folder, valve1 folder). Place under `data/` or update paths in config.

## Dataset & Context
- SKAB dataset: time-series CSVs containing sensor channels such as accelerometer, pressure, flow, temperature, and vibration.
- Labels: anomaly-free data → 0 (normal); valve1 folder → 1 (anomaly).
- Context: mirrors a SLAM robot's telemetry while mapping and navigating, so temporal patterns matter.

## Project flow
1. Raw CSV Data
   - Load anomaly-free and valve1 CSV folders.
   - Concatenate into one dataframe and add column `anomaly` (0 = normal, 1 = anomaly).

2. Exploratory Data Analysis (EDA)
   - Time plots, distributions, missing-value checks, correlations, per-sensor stats.
   - Visualize representative normal vs anomaly windows.

3. Preprocessing & Scaling
   - Handle missing values (interpolate or forward-fill).
   - Normalize/standardize (StandardScaler or RobustScaler).
   - Optionally apply PCA or feature selection.

4. Sliding Window (temporal feature creation)
   - Window size: 50 rows (configurable).
   - Overlap/stride: 1 row (configurable).
   - Each window → one sample: flatten time × sensors into a feature vector (e.g., 50 × 8 = 400 features).
   - Label window anomalous if from valve1 folder (1), else normal (0).

5. Models trained
   - Supervised: Random Forest, XGBoost
   - Unsupervised: Isolation Forest, Autoencoder (dense / LSTM)

6. Training & Evaluation
   - Use time-respecting train/test split or time-series CV.
   - Metrics: confusion matrix, precision, recall, F1 score, precision-recall curve.
   - Compare models focusing on operational tradeoffs: false alarms vs missed faults.

## Results (experiment snapshot)
- Window size: 50, Features per window: 8 (flattened to 400 features)

- Random Forest
  - F1: 0.81
  - Missed faults: 322
  - False alarms: 593

- XGBoost (best)
  - F1: 0.89
  - Fewer false alarms (107), good recall — best operational balance for robot health monitoring

- Isolation Forest (unsupervised)
  - F1: 0.63
  - Normal correctly identified: 1005 (31%), Normal wrongly flagged: 2253
  - Anomaly recall: 0.92 (caught 2057 anomalies; missed 188)
  - Notes: contamination=0.25 likely too aggressive; high dimensionality of flattened windows hurt performance

- Autoencoder
  - F1: 0.43
  - Consider LSTM Autoencoder or 1D CNN for improved temporal modeling

## Interpretation & tradeoffs
- Supervised models outperform unsupervised when labels are available (XGBoost F1 0.89 vs Isolation Forest 0.63).
- Isolation Forest gives high anomaly recall but low precision (useful when missing faults is unacceptable).
- For SLAM robot operation where false alarms are costly, XGBoost is preferred.
- Future work: LSTM Autoencoders, Transformers, online inference, and lightweight models for edge deployment.

## What's working now
- Data loading and merging from CSVs
- EDA and visualization notebooks
- Preprocessing & scaling pipeline
- Sliding window implementation (window size 50)
- Random Forest and XGBoost training and persistence
- Isolation Forest and Autoencoder baselines
- Streamlit dashboard live: model loaded, health score shown (example: 62% — Degraded), gauge and anomaly counts
- Example detection: 410 anomalies detected out of 1079 windows (38%) on a sample fault log

## How to run
1. Clone the repo:
   git clone <repo-url>
2. Install dependencies:
   pip install -r requirements.txt
3. Place SKAB dataset folders under `data/` or update config paths.
4. Create windows:
   python scripts/create_windows.py --data-dir data/ --window-size 50
5. Train models:
   python scripts/train_xgboost.py
   python scripts/train_random_forest.py
   python scripts/train_isolation_forest.py
   python scripts/train_autoencoder.py
6. Evaluate:
   python scripts/evaluate_models.py --models models/
7. Launch dashboard:
   streamlit run app/dashboard.py

## Repository structure
- data/               — raw CSV folders (anomaly-free, valve1)
- notebooks/          — EDA and experiments

  - dashboard.py      — Streamlit app (health score, gauge, anomaly windows)
- models/             — saved models (.joblib, .pkl, .h5)
- requirements.txt
- README.md
- LICENSE

## Notes, tips & future work
- Use chronological splits or nested time-series CV to avoid leakage.
- Window labeling: consider percent-based labels (label window anomalous if >X% rows anomalous) for sparse events.
- Reduce dimensionality (PCA, feature selection) for unsupervised models.
- Try LSTM/GRU autoencoders, 1D CNNs, or Transformers for sequence learning without flattening.
- For real-time deployment: use rolling buffer of last N rows, keep models lightweight, and implement drift monitoring.


## Acknowledgements
- Dataset: SKAB (used to emulate SLAM robot sensor telemetry).  
- Concept: ML module acts as the SLAM robot’s health monitor during mapping & localization.

***

