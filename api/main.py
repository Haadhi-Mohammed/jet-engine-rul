# api/main.py
# FastAPI application for Jet Engine RUL Prediction

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from typing import List
import numpy as np
import pandas as pd
import pickle
import json
import shap
from pathlib import Path
import uvicorn

import tensorflow as tf

from rul.alerts import get_alert_level
from rul.config import RED_BELOW, AMBER_BELOW
from rul.layers import CUSTOM_OBJECTS

# ---- paths ----
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

# ---- load model config ----
# the API trusts the artifacts saved next to the model (not rul.config),
# because they describe how THIS model was trained. tests/test_rul.py
# checks the two never disagree.
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

# fail fast: refuse to start rather than silently feed sensors in the wrong order
if list(scaler.feature_names_in_) != feature_cols:
    raise RuntimeError(
        f"scaler.pkl was fitted on {list(scaler.feature_names_in_)}, "
        f"but feature_cols.pkl says {feature_cols}"
    )

# ---- model metadata — read from the artifacts, never hardcoded ----
_arch        = best_config['architecture']
MODEL_NAME   = f"{_arch['n_gru_layers']}GRU({_arch['gru_units']})+LSTM({_arch['lstm_units']})+Attention"
MODEL_VERSION = 'run_02_v1.0'
_perf        = best_config['performance']
PERFORMANCE  = f"RMSE {_perf['test_rmse']:.2f} | R² {_perf['test_r2']:.3f}"

# ---- load model ----
model = tf.keras.models.load_model(
    MODELS_DIR / 'best_model.keras',
    custom_objects=CUSTOM_OBJECTS
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

# last 30 cycles per fleet engine, already scaled — used by /engines/{id}/explain
fleet_sequences = np.load(MODELS_DIR / 'fleet_sequences.npy')   # (100, 30, 14)

# ---- FastAPI app ----
app = FastAPI(
    title='Jet Engine RUL Prediction API',
    description=f'Predictive maintenance API for NASA CMAPSS turbofan engines. '
                f'{PERFORMANCE} | {MODEL_NAME}',
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
# one cycle of raw sensor readings — one float field per model feature.
# built from feature_cols so the schema can't drift from the model.
SensorReading = create_model(
    'SensorReading',
    **{col: (float, ...) for col in feature_cols}
)


class PredictRequest(BaseModel):
    engine_id: int = Field(..., description="Engine unit number")
    readings: List[SensorReading] = Field(
        ...,
        min_length=SEQUENCE_LENGTH,
        max_length=SEQUENCE_LENGTH,
        description=f"Exactly {SEQUENCE_LENGTH} consecutive cycles of RAW sensor readings"
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


class EngineSensors(BaseModel):
    engine_id:       int
    feature_cols:    List[str]
    # SCALED values (RobustScaler: 0 = training median) — for charts, not for /predict
    scaled_readings: List[List[float]]


# ---- helpers ----
def preprocess_readings(readings: List[SensorReading]) -> np.ndarray:
    # pydantic objects → DataFrame in feature_cols order → scale → add batch dim
    raw = pd.DataFrame(
        [[getattr(r, col) for col in feature_cols] for r in readings],
        columns=feature_cols
    )                                          # (30, 14)
    scaled = scaler.transform(raw)             # (30, 14)
    return scaled[np.newaxis]                  # (1, 30, 14)


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


def predict_sequence(engine_id: int, sequence: np.ndarray) -> PredictResponse:
    # shared by /predict and /engines/{id}/explain
    # sequence must already be SCALED, shape (1, 30, 14)
    raw_pred = model.predict(sequence, verbose=0).flatten()[0]
    rul = float(np.clip(raw_pred, 0, RUL_CAP))

    alert_level, alert_message = get_alert_level(rul)

    shap_values = []
    try:
        shap_values = compute_shap(sequence)
    except Exception as e:
        print(f"SHAP error: {str(e)}")
        # API still returns prediction even if SHAP fails

    return PredictResponse(
        engine_id=engine_id,
        predicted_rul=round(rul, 2),
        alert_level=alert_level,
        alert_message=alert_message,
        shap_values=shap_values,
        model_version=MODEL_VERSION
    )


# ---- endpoints ----
@app.get('/')
def root():
    return {
        'name':        'Jet Engine RUL Prediction API',
        'version':     '1.0.0',
        'status':      'running',
        'model':       MODEL_NAME,
        'model_version': MODEL_VERSION,
        'performance': best_config['performance'],
        'dataset':     'NASA CMAPSS FD001',
        # clients (the dashboard) read these instead of hardcoding them
        'feature_cols':     feature_cols,
        'sequence_length':  SEQUENCE_LENGTH,
        'rul_cap':          RUL_CAP,
        'alert_thresholds': {'red_below': RED_BELOW, 'amber_below': AMBER_BELOW},
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
        # raw readings from the caller → scale here, then predict
        sequence = preprocess_readings(request.readings)
        return predict_sequence(request.engine_id, sequence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/engines/{engine_id}/explain', response_model=PredictResponse)
def explain_engine(engine_id: int):
    # prediction + SHAP for a fleet engine, using its stored (already scaled) sensor window.
    # the dashboard uses this, so it never has to handle scaling itself.
    if not 1 <= engine_id <= len(fleet_sequences):
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")

    sequence = fleet_sequences[engine_id - 1][np.newaxis]   # (1, 30, 14)
    return predict_sequence(engine_id, sequence)


@app.get('/engines/{engine_id}/sensors', response_model=EngineSensors)
def engine_sensors(engine_id: int):
    # the stored sensor window for a fleet engine — lets the dashboard draw
    # its charts without shipping its own copy of the data
    if not 1 <= engine_id <= len(fleet_sequences):
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")

    return EngineSensors(
        engine_id=engine_id,
        feature_cols=feature_cols,
        scaled_readings=fleet_sequences[engine_id - 1].round(4).tolist()
    )


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