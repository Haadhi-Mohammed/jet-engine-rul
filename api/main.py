# api/main.py
# FastAPI application for Jet Engine RUL Prediction

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pickle
import json
import shap
from pathlib import Path
import uvicorn

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization

# ---- paths ----
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

# ---- custom layer — must be defined before model load ----
class ScaledDotProductAttention(Layer):
    # same class as training — needed to deserialise best_model.keras
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x):
        d_k     = tf.cast(tf.shape(x)[-1], tf.float32)
        scale   = tf.math.sqrt(d_k)
        scores  = tf.matmul(x, x, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, x)
        return self.layer_norm(x + context)

# ---- load model config ----
with open(MODELS_DIR / 'model_config.pkl', 'rb') as f:
    config = pickle.load(f)

with open(MODELS_DIR / 'best_model_config.json', 'r') as f:
    best_config = json.load(f)

with open(MODELS_DIR / 'feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

SEQUENCE_LENGTH = config['sequence_length']
N_FEATURES      = config['n_features']
RUL_CAP         = config['rul_cap']

# ---- load scaler ----
with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ---- load model ----
model = tf.keras.models.load_model(
    MODELS_DIR / 'best_model.keras',
    custom_objects={'ScaledDotProductAttention': ScaledDotProductAttention}
)
print("model loaded successfully")

# ---- load SHAP explainer ----
print("loading SHAP explainer...")
shap_background = np.load(MODELS_DIR / 'shap_background.npy')
explainer = shap.GradientExplainer(model, shap_background)
print("SHAP explainer ready")

# ---- load precomputed fleet predictions ----
fleet_preds = np.load(MODELS_DIR / 'y_pred_test.npy')
print(f"fleet predictions loaded — {len(fleet_preds)} engines")

# ---- FastAPI app ----
app = FastAPI(
    title='Jet Engine RUL Prediction API',
    description='Predictive maintenance API for NASA CMAPSS turbofan engines. '
                'RMSE 14.07 | R² 0.877 | GRU-LSTM + Attention',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ---- request / response schemas ----
class SensorReading(BaseModel):
    # one cycle of sensor readings — 14 values in FEATURE_COLS order
    s_2:  float
    s_3:  float
    s_4:  float
    s_7:  float
    s_8:  float
    s_9:  float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_17: float
    s_20: float
    s_21: float


class PredictRequest(BaseModel):
    engine_id: int = Field(..., description="Engine unit number")
    readings: List[SensorReading] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 consecutive cycles of raw sensor readings"
    )


class ShapValue(BaseModel):
    sensor:     str
    importance: float


class PredictResponse(BaseModel):
    engine_id:     int
    predicted_rul: float
    alert_level:   str
    alert_message: str
    shap_values:   List[ShapValue]
    model_version: str


class EngineStatus(BaseModel):
    engine_id:     int
    predicted_rul: float
    alert_level:   str
    alert_message: str


class FleetResponse(BaseModel):
    total_engines: int
    red_count:     int
    amber_count:   int
    green_count:   int
    engines:       List[EngineStatus]


# ---- helpers ----
def get_alert_level(rul: float):
    # thresholds match dashboard colour coding
    if rul < 30:
        return 'RED',   'Immediate maintenance required'
    elif rul < 60:
        return 'AMBER', 'Schedule maintenance soon'
    return 'GREEN', 'Engine healthy'


def preprocess_readings(readings: List[SensorReading]) -> np.ndarray:
    # convert pydantic objects → numpy array → scale → reshape for model
    raw = np.array([[
        r.s_2, r.s_3, r.s_4, r.s_7, r.s_8, r.s_9,
        r.s_11, r.s_12, r.s_13, r.s_14, r.s_15,
        r.s_17, r.s_20, r.s_21
    ] for r in readings], dtype=np.float32)    # (30, 14)

    scaled = scaler.transform(raw)             # (30, 14)
    return scaled.reshape(1, 30, 14)           # (1, 30, 14)


def compute_shap(sequence: np.ndarray) -> List[ShapValue]:
    # returns per-sensor importance averaged across all 30 timesteps
    shap_vals = np.array(
        explainer.shap_values(sequence)
    ).squeeze(-1)                              # (1, 30, 14)

    mean_importance = np.abs(shap_vals).mean(axis=1)[0]  # (14,)

    return sorted([
        ShapValue(sensor=feat, importance=round(float(imp), 4))
        for feat, imp in zip(feature_cols, mean_importance)
    ], key=lambda x: x.importance, reverse=True)


# ---- endpoints ----
@app.get('/')
def root():
    return {
        'name':        'Jet Engine RUL Prediction API',
        'version':     '1.0.0',
        'status':      'running',
        'model':       '2GRU(64)+LSTM(32)+Attention',
        'performance': best_config['performance'],
        'dataset':     'NASA CMAPSS FD001',
    }


@app.get('/health')
def health():
    # Render pings this — must stay fast, no model inference here
    return {'status': 'healthy', 'model_loaded': model is not None}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.readings) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {SEQUENCE_LENGTH} cycles required. "
                   f"Got {len(request.readings)}."
        )

    try:
        sequence = preprocess_readings(request.readings)

        # RUL prediction
        raw_pred = model.predict(sequence, verbose=0).flatten()[0]
        rul = float(np.clip(raw_pred, 0, RUL_CAP))

        # alert level
        alert_level, alert_message = get_alert_level(rul)

        # SHAP explanation
        shap_values = []
        try:
            shap_values = compute_shap(sequence)
        except Exception as e:
            print(f"SHAP error: {str(e)}")
            # API still returns prediction even if SHAP fails

        return PredictResponse(
            engine_id=request.engine_id,
            predicted_rul=round(rul, 2),
            alert_level=alert_level,
            alert_message=alert_message,
            shap_values=shap_values,
            model_version='run_02_v1.0'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/fleet', response_model=FleetResponse)
def fleet():
    # serving precomputed predictions — instant response, no inference
    # sorted by RUL ascending — most critical engines first
    engines = []
    for i, rul in enumerate(fleet_preds):
        alert_level, alert_message = get_alert_level(float(rul))
        engines.append(EngineStatus(
            engine_id=i + 1,
            predicted_rul=round(float(rul), 2),
            alert_level=alert_level,
            alert_message=alert_message
        ))

    engines.sort(key=lambda x: x.predicted_rul)

    red   = sum(1 for e in engines if e.alert_level == 'RED')
    amber = sum(1 for e in engines if e.alert_level == 'AMBER')
    green = sum(1 for e in engines if e.alert_level == 'GREEN')

    return FleetResponse(
        total_engines=len(engines),
        red_count=red,
        amber_count=amber,
        green_count=green,
        engines=engines
    )


# ---- run locally ----
if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8001, reload=True)