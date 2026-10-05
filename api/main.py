# api/main.py
# FastAPI application for Jet Engine RUL Prediction

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model
from typing import List
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
import logging
import threading
import time
import warnings
import numpy as np
import pandas as pd
import pickle
import json
import shap
import uvicorn

import tensorflow as tf

from rul.alerts import get_alert_level
from rul.config import RED_BELOW, AMBER_BELOW
from rul.layers import CUSTOM_OBJECTS

# ---- logging ----
# logging (not print) gives timestamps + levels, and shows up properly in Render's logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('rul-api')

# SHAP passes inputs to the Keras functional model as a list — harmless, but Keras
# warns on every call and floods the logs
warnings.filterwarnings('ignore', message=r'The structure of `inputs` doesn.t match')

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

# ---- input sanity limits ----
# RobustScaler maps each sensor to (value - median) / IQR. training data spans roughly
# -3 … +11 in those units, so anything beyond ±20 IQRs from the median is not a real
# engine reading — it's a unit mistake, a typo or garbage, and the model's answer
# would be meaningless. reject it instead of returning a confident-looking number.
MAX_ABS_SCALED = 20
INPUT_RANGES = {
    col: (round(float(c - MAX_ABS_SCALED * s), 4), round(float(c + MAX_ABS_SCALED * s), 4))
    for col, c, s in zip(feature_cols, scaler.center_, scaler.scale_)
}

# ---- model metadata — read from the artifacts, never hardcoded ----
_arch        = best_config['architecture']
_gru, _lstm  = f"{_arch['n_gru_layers']}GRU({_arch['gru_units']})", f"LSTM({_arch['lstm_units']})"
MODEL_NAME   = (f"{_lstm}+{_gru}" if _arch.get('reverse_order') else f"{_gru}+{_lstm}") + "+Attention"
MODEL_VERSION = best_config['model_version']
# headline = mean ± std over the 5 training seeds (one model's score is partly luck)
_seeds       = best_config['performance_across_seeds']
PERFORMANCE  = (f"RMSE {_seeds['rmse']['mean']:.1f} ± {_seeds['rmse']['std']:.1f} | "
                f"R² {_seeds['r2']['mean']:.2f} (5 seeds)")

# ---- load model ----
model = tf.keras.models.load_model(
    MODELS_DIR / 'best_model.keras',
    custom_objects=CUSTOM_OBJECTS
)
log.info("model loaded: %s", MODEL_NAME)

# ---- load SHAP explainer ----
shap_background = np.load(MODELS_DIR / 'shap_background.npy')
explainer = shap.GradientExplainer(model, shap_background)
SHAP_SEED = 0   # GradientExplainer samples randomly — fixed seed = same answer every time
log.info("SHAP explainer ready")

# ---- load fleet data (the 100 CMAPSS FD001 test engines) ----
fleet_preds     = np.load(MODELS_DIR / 'y_pred_test.npy')       # (100,)  precomputed RUL
fleet_sequences = np.load(MODELS_DIR / 'fleet_sequences.npy')   # (100, 30, 14) scaled windows
fleet_true_rul  = np.load(MODELS_DIR / 'fleet_true_rul.npy')    # (100,)  actual RUL, uncapped
log.info("fleet loaded: %d engines", len(fleet_preds))

# ---- FastAPI app ----
API_VERSION = '2.0.0'
app = FastAPI(
    title='Jet Engine RUL Prediction API',
    description=f'Predictive maintenance API for NASA CMAPSS turbofan engines. '
                f'{PERFORMANCE} | {MODEL_NAME}',
    version=API_VERSION
)

# public, read-only API with no cookies or logins — so any origin may call it,
# but credentials are never allowed ('*' + credentials would let any site send
# a user's cookies along)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 echoes the offending input back. for NaN/Infinity that
    # can't be encoded as JSON, so the error response itself crashed (→ 500).
    # return where + what went wrong, without the input.
    errors = [{k: e[k] for k in ('loc', 'msg', 'type') if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content={'detail': errors})


# ---- rate limiting ----
class RateLimiter:
    # sliding window per client: at most max_calls in any per_seconds window.
    # in-memory, so it's per process — fine for one Render instance.
    def __init__(self, max_calls: int, per_seconds: float):
        self.max_calls   = max_calls
        self.per_seconds = per_seconds
        self.calls       = defaultdict(deque)
        self.lock        = threading.Lock()   # FastAPI runs sync endpoints in a thread pool
        self.last_sweep  = time.monotonic()

    def _sweep(self, now: float):
        # forget clients with no calls inside the window, so the table can't grow forever
        for key in [k for k, q in self.calls.items() if not q or now - q[-1] > self.per_seconds]:
            del self.calls[key]
        self.last_sweep = now

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if now - self.last_sweep > self.per_seconds:
                self._sweep(now)
            q = self.calls[key]
            while q and now - q[0] > self.per_seconds:
                q.popleft()
            if len(q) >= self.max_calls:
                return False
            q.append(now)
            return True


# /predict runs the model + SHAP (CPU-heavy) — cap it so one client can't hog the server
predict_limiter = RateLimiter(max_calls=30, per_seconds=60)


def limit_predict(request: Request):
    # behind Render's proxy the client IP is the first entry of X-Forwarded-For.
    # NOTE: only trustworthy because the platform's proxies (Cloudflare → Render) set
    # this header; run the API directly on the internet and clients could fake it.
    # compute_slot() below protects the server regardless of who the client claims to be.
    forwarded = request.headers.get('x-forwarded-for')
    client = forwarded.split(',')[0].strip() if forwarded else request.client.host
    if not predict_limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests — max 30 predictions per minute")


# ---- one heavy computation at a time ----
# the model + SHAP are CPU- and memory-heavy. in a load test, 8 at once on the 512 MB
# free instance produced 502/503s. so they run one at a time; a request that can't get
# its turn within BUSY_TIMEOUT gets a quick 503 + Retry-After instead of piling up.
# (one at a time also keeps SHAP deterministic: GradientExplainer's seed calls
#  np.random.seed(), which is global to the whole process.)
COMPUTE_LOCK = threading.Lock()
BUSY_TIMEOUT = 10   # seconds


@contextmanager
def compute_slot():
    if not COMPUTE_LOCK.acquire(timeout=BUSY_TIMEOUT):
        raise HTTPException(status_code=503, detail="Server busy — please retry shortly",
                            headers={'Retry-After': '5'})
    try:
        yield
    finally:
        COMPUTE_LOCK.release()


# ---- request / response schemas ----
# one cycle of raw sensor readings — one float field per model feature.
# built from feature_cols so the schema can't drift from the model.
# allow_inf_nan=False: pydantic accepts NaN/Infinity for floats by default
SensorReading = create_model(
    'SensorReading',
    **{col: (float, Field(..., allow_inf_nan=False)) for col in feature_cols}
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
    # known only because this is a benchmark test set — a real fleet wouldn't have it
    actual_rul:    float
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

    # reject readings far outside anything the model has seen.
    # same error shape as pydantic's 422s, so clients handle one format
    out_of_range = np.argwhere(np.abs(scaled) > MAX_ABS_SCALED)   # [(cycle, sensor), ...]
    if len(out_of_range):
        cycle, j = (int(v) for v in out_of_range[0])
        col = feature_cols[j]
        lo, hi = INPUT_RANGES[col]
        raise HTTPException(status_code=422, detail=[{
            'loc':  ['body', 'readings', cycle, col],
            'msg':  f"{col} = {raw.iloc[cycle, j]} is outside the plausible range [{lo}, {hi}]",
            'type': 'value_out_of_range',
        }])

    return scaled[np.newaxis]                  # (1, 30, 14)


def compute_shap(sequence: np.ndarray) -> List[ShapValue]:
    # returns per-sensor importance averaged across all 30 timesteps
    shap_vals = np.array(
        explainer.shap_values(sequence, rseed=SHAP_SEED)
    ).squeeze(-1)                              # (1, 30, 14)

    mean_importance = np.abs(shap_vals).mean(axis=1)[0]  # (14,)

    return sorted([
        ShapValue(sensor=feat, importance=round(float(imp), 4))
        for feat, imp in zip(feature_cols, mean_importance)
    ], key=lambda x: x.importance, reverse=True)


def predict_sequence(engine_id: int, sequence: np.ndarray) -> PredictResponse:
    # shared by /predict and /engines/{id}/explain
    # sequence must already be SCALED, shape (1, 30, 14)
    # calling the model directly is much faster than model.predict() for one sample
    raw_pred = float(model(sequence, training=False).numpy().flatten()[0])
    rul = float(np.clip(raw_pred, 0, RUL_CAP))

    alert_level, alert_message = get_alert_level(rul)

    shap_values = []
    try:
        shap_values = compute_shap(sequence)
    except Exception:
        # API still returns the prediction even if SHAP fails
        log.exception("SHAP failed for engine %s", engine_id)

    return PredictResponse(
        engine_id=engine_id,
        predicted_rul=round(rul, 2),
        alert_level=alert_level,
        alert_message=alert_message,
        shap_values=shap_values,
        model_version=MODEL_VERSION
    )


def check_engine_id(engine_id: int):
    if not 1 <= engine_id <= len(fleet_sequences):
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")


# fleet windows never change and SHAP is seeded, so each engine's answer is
# computed once and then served from memory
explain_cache: dict[int, PredictResponse] = {}


def explain_fleet_engine(engine_id: int) -> PredictResponse:
    if engine_id in explain_cache:
        return explain_cache[engine_id]

    sequence = fleet_sequences[engine_id - 1][np.newaxis]   # (1, 30, 14)
    with compute_slot():
        result = predict_sequence(engine_id, sequence)

    # only cache complete answers — if SHAP failed this time, try again next request
    # instead of serving an empty explanation until the server restarts
    if result.shap_values:
        explain_cache[engine_id] = result
    return result


# ---- endpoints ----
@app.get('/')
def root():
    return {
        'name':        'Jet Engine RUL Prediction API',
        'version':     API_VERSION,
        'status':      'running',
        'model':       MODEL_NAME,
        'model_version': MODEL_VERSION,
        'performance': best_config['performance'],                     # the deployed model
        'performance_across_seeds': best_config['performance_across_seeds'],
        'performance_summary': PERFORMANCE,
        'dataset':     'NASA CMAPSS FD001',
        # clients (the dashboard) read these instead of hardcoding them
        'feature_cols':     feature_cols,
        'sequence_length':  SEQUENCE_LENGTH,
        'rul_cap':          RUL_CAP,
        'alert_thresholds': {'red_below': RED_BELOW, 'amber_below': AMBER_BELOW},
        'input_ranges':     INPUT_RANGES,
    }


@app.get('/health')
def health():
    # Render pings this — must stay fast, no model inference here.
    # the model loads at import time, so if this answers, the model is loaded.
    return {'status': 'healthy', 'model_version': MODEL_VERSION}


@app.post('/predict', response_model=PredictResponse, dependencies=[Depends(limit_predict)])
def predict(request: PredictRequest):
    # raw readings from the caller → validate + scale here, then predict.
    # length, NaN/Infinity and field checks already happened in PredictRequest (→ 422)
    sequence = preprocess_readings(request.readings)
    with compute_slot():                    # outside the try: a 503 "busy" must not become a 500
        try:
            return predict_sequence(request.engine_id, sequence)
        except Exception:
            # full details go to the server log; the client gets a generic message,
            # never internal paths, library versions or stack traces
            log.exception("prediction failed for engine %s", request.engine_id)
            raise HTTPException(status_code=500, detail="Internal error while predicting")


@app.get('/engines/{engine_id}/explain', response_model=PredictResponse)
def explain_engine(engine_id: int):
    # prediction + SHAP for a fleet engine, using its stored (already scaled) sensor window.
    # the dashboard uses this, so it never has to handle scaling itself.
    check_engine_id(engine_id)
    return explain_fleet_engine(engine_id)


@app.get('/engines/{engine_id}/sensors', response_model=EngineSensors)
def engine_sensors(engine_id: int):
    # the stored sensor window for a fleet engine — lets the dashboard draw
    # its charts without shipping its own copy of the data
    check_engine_id(engine_id)
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
    for i, (rul, actual) in enumerate(zip(fleet_preds, fleet_true_rul)):
        alert_level, alert_message = get_alert_level(float(rul))
        engines.append(EngineStatus(
            engine_id=i + 1,
            predicted_rul=round(float(rul), 2),
            actual_rul=float(actual),
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
# from the project root:  python -m api.main   (or: uvicorn api.main:app --reload --port 8001)
if __name__ == '__main__':
    uvicorn.run('api.main:app', host='0.0.0.0', port=8001, reload=True)
