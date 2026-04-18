# Jet Engine RUL Prediction System

A production-grade predictive maintenance system for aviation turbofan 
engines — predicting Remaining Useful Life (RUL) using a GRU-LSTM 
hybrid model with attention mechanism.

## Project Overview

Unplanned engine failures cost the aviation industry billions annually. 
This system predicts how many flight cycles remain before an engine 
requires maintenance, enabling proactive scheduling and preventing 
catastrophic failures.

## Dataset

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
- FD001: 100 training engines, 100 test engines
- Single operating condition, single fault mode
- 21 sensor measurements per cycle

## Model Architecture

- 4 × GRU layers
- 1 × LSTM layer  
- Scaled Dot-Product Attention mechanism
- RobustScaler preprocessing
- Validation RMSE: 24.62 | MAE: 18.21

## Tech Stack

| Layer | Tools |
|---|---|
| Modelling | TensorFlow/Keras (GRU-LSTM + Attention) |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI + Docker |
| Dashboard | Streamlit (Fleet + Individual engine) |
| CI/CD | GitHub Actions |
| Deployment | Render + HuggingFace Spaces |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Author

Haadhi Mohammed  
MSc Data Science — Coventry University (Distinction)  
Dissertation project — Academic Year 2024/25