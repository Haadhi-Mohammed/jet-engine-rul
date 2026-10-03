# tests for the shared rul package

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from rul import config
from rul.alerts import get_alert_level

ROOT      = Path(__file__).parent.parent
MODELS    = ROOT / 'models'
RAW       = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'


# ---- 1. unit tests: small pure functions, check the edges ----
@pytest.mark.parametrize('rul, level', [
    (0, 'RED'), (29.99, 'RED'),
    (30, 'AMBER'), (59.99, 'AMBER'),       # boundaries are where bugs hide
    (60, 'GREEN'), (125, 'GREEN'),
])
def test_alert_levels(rul, level):
    assert get_alert_level(rul)[0] == level


def test_nasa_score_penalises_late_predictions_more():
    from rul.metrics import nasa_score
    # hand-computed: 10 cycles late → e^(10/10) - 1 = 1.718; 10 early → e^(10/13) - 1 = 1.158
    assert nasa_score([50], [60]) == pytest.approx(np.e - 1)
    assert nasa_score([50], [40]) == pytest.approx(np.exp(10 / 13) - 1)
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert nasa_score([50, 80], [50, 80]) == 0


def test_evaluate_clips_to_cap_before_scoring():
    from rul.metrics import evaluate
    # true RUL 145 and prediction 125 count as perfect: both mean "healthy, ≥ cap"
    m = evaluate([145, 10], [125, 10], cap=125)
    assert m['rmse'] == 0 and m['late_predictions'] == 0


def test_split_engines_never_shares_an_engine():
    import pandas as pd
    from rul.data import split_engines
    df = pd.DataFrame({'unit_number': np.repeat(np.arange(1, 101), 5)})
    tr, val = split_engines(df, val_fraction=0.2, seed=42)
    assert set(tr['unit_number']).isdisjoint(val['unit_number'])
    assert val['unit_number'].nunique() == 20
    assert len(tr) + len(val) == len(df)
    # same seed → same split, every time
    assert set(split_engines(df, 0.2, 42)[1]['unit_number']) == set(val['unit_number'])


# ---- 2. consistency tests: code constants must match the saved artifacts ----
# the model was trained with a specific feature list, window length and cap.
# if someone edits rul/config.py without retraining, these fail.
def load_pickle(name):
    with open(MODELS / name, 'rb') as f:
        return pickle.load(f)


def test_config_matches_model_config_pkl():
    saved = load_pickle('model_config.pkl')
    assert saved['feature_cols']    == config.FEATURE_COLS
    assert saved['sequence_length'] == config.SEQUENCE_LENGTH
    assert saved['rul_cap']         == config.RUL_CAP
    assert saved['sensors_dropped'] == config.SENSORS_TO_DROP


def test_config_matches_other_artifacts():
    assert load_pickle('feature_cols.pkl') == config.FEATURE_COLS
    # the scaler remembers the column names it was fitted on
    assert list(load_pickle('scaler.pkl').feature_names_in_) == config.FEATURE_COLS
    best = json.loads((MODELS / 'best_model_config.json').read_text())
    assert best['data']['feature_cols'] == config.FEATURE_COLS


def test_fleet_arrays_line_up():
    seqs  = np.load(MODELS / 'fleet_sequences.npy')
    preds = np.load(MODELS / 'y_pred_test.npy')
    assert seqs.shape == (len(preds), config.SEQUENCE_LENGTH, len(config.FEATURE_COLS))


# ---- 3. reproduction test: the package rebuilds the notebook's arrays exactly ----
# data/ is gitignored, so this only runs where the raw data has been downloaded
needs_data = pytest.mark.skipif(
    not (RAW / 'train_FD001.txt').exists() or not (PROCESSED / 'X_train.npy').exists(),
    reason='CMAPSS raw/processed data not present'
)


@needs_data
def test_pipeline_matches_original_notebook():
    # the vectorized rul.data functions rebuild 02_preprocessing.ipynb's arrays exactly
    # (that notebook fitted its scaler on all 100 training engines)
    from rul import data

    train, test, true_rul = data.load_cmapss(RAW)
    train  = data.add_train_rul(train)
    scaler = data.fit_scaler(train)

    X_train, y_train = data.create_sequences(data.scale(train, scaler))
    X_test = data.last_windows(data.scale(test, scaler))

    np.testing.assert_array_equal(X_train, np.load(PROCESSED / 'X_train.npy'))
    np.testing.assert_array_equal(y_train, np.load(PROCESSED / 'y_train.npy'))
    np.testing.assert_array_equal(X_test,  np.load(PROCESSED / 'X_test.npy'))
    np.testing.assert_array_equal(true_rul, np.load(PROCESSED / 'y_test.npy'))


@needs_data
def test_saved_artifacts_match_training_pipeline():
    # the deployed scaler and fleet data are exactly what scripts/train.py produces:
    # scaler fitted on the training engines of the fixed split, test windows scaled with it
    from rul import data
    from scripts.train import SPLIT_SEED

    train, test, true_rul = data.load_cmapss(RAW)
    tr, _  = data.split_engines(data.add_train_rul(train), val_fraction=0.2, seed=SPLIT_SEED)
    scaler = data.fit_scaler(tr)

    saved_scaler = load_pickle('scaler.pkl')
    np.testing.assert_allclose(scaler.center_, saved_scaler.center_)
    np.testing.assert_allclose(scaler.scale_,  saved_scaler.scale_)

    X_test = data.last_windows(data.scale(test, scaler))
    np.testing.assert_array_equal(X_test,   np.load(MODELS / 'fleet_sequences.npy'))
    np.testing.assert_array_equal(true_rul, np.load(MODELS / 'fleet_true_rul.npy'))
