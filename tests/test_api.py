# tests for the FastAPI service
# run from the project root:  python -m pytest -q

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api

# the API loads the model + SHAP explainer at import time (slow),
# so one client is shared by every test in this file
client = TestClient(api.app)

FLEET_SEQUENCES = api.fleet_sequences   # (100, 30, 14) — scaled
FLEET_PREDS     = api.fleet_preds       # (100,)        — precomputed RUL


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    # every test starts with an empty rate-limit history, so tests don't affect each other
    api.predict_limiter.calls.clear()


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


def test_root_exposes_what_clients_need():
    # the dashboard relies on these keys instead of hardcoding values
    body = client.get('/').json()
    assert body['feature_cols'] == api.feature_cols
    assert body['alert_thresholds'] == {'red_below': 30, 'amber_below': 60}
    assert body['rul_cap'] == api.RUL_CAP
    assert body['model'] == api.MODEL_NAME
    assert body['model_version'] == api.best_config['model_version']
    assert set(body['performance_across_seeds']) >= {'rmse', 'r2'}


def test_fleet_counts_add_up():
    body = client.get('/fleet').json()
    assert body['total_engines'] == len(FLEET_PREDS)
    assert body['red_count'] + body['amber_count'] + body['green_count'] == body['total_engines']


def test_fleet_includes_actual_rul():
    engines = {e['engine_id']: e for e in client.get('/fleet').json()['engines']}
    assert engines[34]['actual_rul'] == api.fleet_true_rul[33] == 7


def test_cors_allows_any_origin_but_never_credentials():
    r = client.get('/health', headers={'Origin': 'https://example.com'})
    assert r.headers['access-control-allow-origin'] == '*'
    assert 'access-control-allow-credentials' not in r.headers


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


@pytest.mark.parametrize('bad_value', ['NaN', 'Infinity', '-Infinity'])
def test_predict_rejects_nan_and_infinity(bad_value):
    # valid JSON can't hold NaN, but Python's json accepts it — so build the body by hand
    body = json.dumps(raw_request(1)).replace('"s_2": ', f'"s_2": {bad_value}, "_x": ', 1)
    r = client.post('/predict', content=body, headers={'Content-Type': 'application/json'})
    assert r.status_code == 422


def test_predict_rejects_implausible_readings():
    # a real s_2 reading is ~640; 1e30 used to be accepted and returned RUL 96.4
    body = raw_request(1)
    body['readings'][4]['s_2'] = 1e30
    r = client.post('/predict', json=body)
    assert r.status_code == 422
    error = r.json()['detail'][0]                   # same list shape as pydantic's 422s
    assert error['loc'] == ['body', 'readings', 4, 's_2']
    assert error['type'] == 'value_out_of_range'
    assert 'plausible range' in error['msg']


def test_predict_hides_internal_errors(monkeypatch):
    # if the model crashes, the client gets a generic message, not the stack trace
    def broken_model(*args, **kwargs):
        raise RuntimeError('secret internal detail: /opt/render/project/src/models')
    monkeypatch.setattr(api, 'model', broken_model)

    r = client.post('/predict', json=raw_request(1))
    assert r.status_code == 500
    assert 'secret' not in r.text
    assert r.json()['detail'] == 'Internal error while predicting'


def test_predict_is_rate_limited(monkeypatch):
    monkeypatch.setattr(api, 'predict_limiter', api.RateLimiter(max_calls=2, per_seconds=60))
    codes = [client.post('/predict', json=raw_request(1)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ---- /engines/{id}/explain — what the dashboard drill-down uses ----
@pytest.mark.parametrize('engine_id', [1, 20, 34, 81])
def test_explain_matches_fleet_prediction(engine_id):
    r = client.get(f'/engines/{engine_id}/explain')
    assert r.status_code == 200
    body = r.json()
    assert body['engine_id'] == engine_id
    assert body['predicted_rul'] == pytest.approx(FLEET_PREDS[engine_id - 1], abs=0.05)
    assert len(body['shap_values']) == len(api.feature_cols)


def test_explain_is_deterministic():
    # SHAP is seeded, so the same engine always gets the same explanation
    first  = client.get('/engines/20/explain').json()
    api.explain_cache.clear()                       # force a real recomputation
    second = client.get('/engines/20/explain').json()
    assert first == second


def test_explain_gives_different_answers_for_different_engines():
    # regression test: the old dashboard double-scaled its input,
    # and every engine came back as 91.08 / GREEN
    ruls = {client.get(f'/engines/{i}/explain').json()['predicted_rul'] for i in (1, 20, 34, 81)}
    assert len(ruls) == 4


@pytest.mark.parametrize('endpoint', ['explain', 'sensors'])
@pytest.mark.parametrize('engine_id', [0, 101, -3])
def test_unknown_engine_is_404(endpoint, engine_id):
    assert client.get(f'/engines/{engine_id}/{endpoint}').status_code == 404


# ---- /engines/{id}/sensors ----
def test_sensors_returns_the_stored_window():
    body = client.get('/engines/34/sensors').json()
    assert body['feature_cols'] == api.feature_cols
    np.testing.assert_allclose(body['scaled_readings'], FLEET_SEQUENCES[33], atol=1e-4)


# ---- behaviour under failure and load ----
def test_shap_failure_is_not_cached(monkeypatch):
    # a one-off SHAP error must not stick: the next request should get a full explanation
    api.explain_cache.clear()
    real_shap = api.compute_shap

    def failing_shap(sequence):
        raise RuntimeError('transient')
    monkeypatch.setattr(api, 'compute_shap', failing_shap)
    assert client.get('/engines/5/explain').json()['shap_values'] == []

    monkeypatch.setattr(api, 'compute_shap', real_shap)       # the problem goes away
    assert len(client.get('/engines/5/explain').json()['shap_values']) == len(api.feature_cols)


def test_busy_server_answers_503_quickly(monkeypatch):
    # while another computation holds the slot, a new one waits BUSY_TIMEOUT, then gets 503
    api.explain_cache.clear()
    monkeypatch.setattr(api, 'BUSY_TIMEOUT', 0.1)
    api.COMPUTE_LOCK.acquire()
    try:
        r = client.post('/predict', json=raw_request(1))
        assert r.status_code == 503
        assert r.headers['retry-after'] == '5'
        assert client.get('/engines/7/explain').status_code == 503
    finally:
        api.COMPUTE_LOCK.release()
    assert client.post('/predict', json=raw_request(1)).status_code == 200


def test_concurrent_requests_all_succeed_with_identical_answers():
    # several requests at once: none fail, and seeded SHAP gives every one the same answer
    from concurrent.futures import ThreadPoolExecutor

    def call(_):
        with TestClient(api.app) as c:
            return c.post('/predict', json=raw_request(20)).json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, range(4)))
    assert all(r == results[0] for r in results)
    assert len(results[0]['shap_values']) == len(api.feature_cols)


def test_rate_limiter_forgets_idle_clients():
    limiter = api.RateLimiter(max_calls=5, per_seconds=0.05)
    for i in range(100):
        limiter.allow(f'client-{i}')
    import time
    time.sleep(0.1)
    limiter.allow('new-client')                     # triggers the sweep
    assert list(limiter.calls) == ['new-client']
