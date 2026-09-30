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
def test_pipeline_reproduces_saved_arrays():
    from rul import data

    train, test, true_rul = data.load_cmapss(RAW)
    train  = data.add_train_rul(train)
    scaler = data.fit_scaler(train)

    # the freshly fitted scaler must equal the saved one
    saved_scaler = load_pickle('scaler.pkl')
    np.testing.assert_allclose(scaler.center_, saved_scaler.center_)
    np.testing.assert_allclose(scaler.scale_,  saved_scaler.scale_)

    X_train, y_train = data.create_sequences(data.scale(train, scaler))
    X_test = data.last_windows(data.scale(test, scaler))

    np.testing.assert_array_equal(X_train, np.load(PROCESSED / 'X_train.npy'))
    np.testing.assert_array_equal(y_train, np.load(PROCESSED / 'y_train.npy'))
    np.testing.assert_array_equal(X_test,  np.load(PROCESSED / 'X_test.npy'))
    np.testing.assert_array_equal(true_rul, np.load(PROCESSED / 'y_test.npy'))
    # and the API's copy of the fleet windows is the same data
    np.testing.assert_array_equal(X_test, np.load(MODELS / 'fleet_sequences.npy'))
