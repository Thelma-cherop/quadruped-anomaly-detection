# quadruped-anomaly-detection

# Quadruped Robot Anomaly Detection System

Final Year Project — Mechatronics Engineering

## Project Overview
ML framework for detecting operational anomalies in quadruped 
robotic systems using sensor data.

## Models Implemented
- XGBoost (F1: 0.89) ← best model
- Random Forest (F1: 0.81)
- Isolation Forest (F1: 0.63)
- Autoencoder (F1: 0.43)

## How to Run Dashboard
pip install -r requirements.txt
streamlit run dashboard.py

## Dataset
SKAB — Skoltech Anomaly Benchmark
