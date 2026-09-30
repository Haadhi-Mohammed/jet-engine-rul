# tests for the FastAPI service
# run from the project root:  python -m pytest -q

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api

# the API loads the model + SHAP explainer at import time (slow),
# so one client is shared by every test in this file
client = TestClient(api.app)

FLEET_SEQUENCES = api.fleet_sequences   # (100, 30, 14) — scaled
FLEET_PREDS     = api.fleet_preds       # (100,)        — precomputed RUL


def raw_request(engine_id: int) -> dict:
    # /predict expects RAW sensor readings, exactly like a real engine would send.
    # the stored sequences are scaled, so undo the scaling first.
    raw = api.scaler.inverse_transform(FLEET_SEQUENCES[engine_id - 1])
    readings = [dict(zip(api.feature_cols, map(float, row))) for row in raw]
    return {'engine_id': engine_id, 'readings': readings}


# ---- basic endpoints ----
def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'healthy'


def test_fleet_counts_add_up():
    body = client.get('/fleet').json()
    assert body['total_engines'] == len(FLEET_PREDS)
    assert body['red_count'] + body['amber_count'] + body['green_count'] == body['total_engines']


# ---- /predict ----
@pytest.mark.parametrize('engine_id', [1, 20, 81])
def test_predict_with_raw_readings_matches_precomputed(engine_id):
    # the whole pipeline (scale → model) must reproduce the notebook's prediction
    r = client.post('/predict', json=raw_request(engine_id))
    assert r.status_code == 200
    assert r.json()['predicted_rul'] == pytest.approx(FLEET_PREDS[engine_id - 1], abs=0.05)


def test_predict_rejects_wrong_sequence_length():
    body = raw_request(1)
    body['readings'] = body['readings'][:29]
    assert client.post('/predict', json=body).status_code == 422


# ---- /engines/{id}/explain — what the dashboard drill-down uses ----
@pytest.mark.parametrize('engine_id', [1, 20, 34, 81])
def test_explain_matches_fleet_prediction(engine_id):
    r = client.get(f'/engines/{engine_id}/explain')
    assert r.status_code == 200
    body = r.json()
    assert body['engine_id'] == engine_id
    assert body['predicted_rul'] == pytest.approx(FLEET_PREDS[engine_id - 1], abs=0.05)
    assert len(body['shap_values']) == len(api.feature_cols)


def test_explain_gives_different_answers_for_different_engines():
    # regression test: the old dashboard double-scaled its input,
    # and every engine came back as 91.08 / GREEN
    ruls = {client.get(f'/engines/{i}/explain').json()['predicted_rul'] for i in (1, 20, 34, 81)}
    assert len(ruls) == 4


@pytest.mark.parametrize('engine_id', [0, 101, -3])
def test_explain_unknown_engine_is_404(engine_id):
    assert client.get(f'/engines/{engine_id}/explain').status_code == 404
