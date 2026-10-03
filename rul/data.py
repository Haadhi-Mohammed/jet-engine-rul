# CMAPSS loading and preprocessing — the pipeline from 02_preprocessing.ipynb as functions

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler

from rul.config import (
    RAW_COLS, SETTING_COLS, SENSORS_TO_DROP, FEATURE_COLS, RUL_CAP, SEQUENCE_LENGTH
)


def load_cmapss(raw_dir: Path, subset: str = 'FD001'):
    # returns (train_df, test_df, true_rul) with constant sensors and settings dropped
    # true_rul[i] is the real RUL at the last test cycle of engine i+1
    raw_dir = Path(raw_dir)

    def read(name):
        df = pd.read_csv(raw_dir / name, sep=r'\s+', header=None, names=RAW_COLS)
        return df.drop(columns=SENSORS_TO_DROP + SETTING_COLS)

    train = read(f'train_{subset}.txt')
    test  = read(f'test_{subset}.txt')
    true_rul = pd.read_csv(raw_dir / f'RUL_{subset}.txt', header=None).iloc[:, 0].to_numpy()
    return train, test, true_rul


def add_train_rul(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    # training engines run to failure, so RUL at cycle t = last cycle - t,
    # capped: early-life cycles all look "healthy", so they share one label
    train = train.copy()
    last_cycle = train.groupby('unit_number')['time_cycles'].transform('max')
    train['RUL'] = (last_cycle - train['time_cycles']).clip(upper=cap)
    return train


def split_engines(train: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42):
    # hold out WHOLE engines for validation. splitting windows at random would put
    # overlapping windows of the same engine on both sides — the model would be
    # validated on data it has almost seen, and the score would look too good.
    units = train['unit_number'].unique()
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, size=round(len(units) * val_fraction), replace=False)
    is_val = train['unit_number'].isin(val_units)
    return train[~is_val].copy(), train[is_val].copy()


def fit_scaler(train: pd.DataFrame) -> RobustScaler:
    # fit on TRAINING data only — fitting on test data would leak information
    return RobustScaler().fit(train[FEATURE_COLS])


def scale(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df


def create_sequences(train: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH):
    # every window of seq_len consecutive cycles → one sample
    # target = RUL at the last cycle of the window
    X, y = [], []
    for _, engine in train.groupby('unit_number', sort=False):
        values = engine[FEATURE_COLS].to_numpy()
        if len(values) < seq_len:
            continue
        # (n_windows, n_features, seq_len) → (n_windows, seq_len, n_features)
        X.append(sliding_window_view(values, seq_len, axis=0).transpose(0, 2, 1))
        y.append(engine['RUL'].to_numpy()[seq_len - 1:])
    return np.concatenate(X), np.concatenate(y)


def last_windows(test: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    # test engines stop before failure — we predict from each engine's last seq_len cycles.
    # engines shorter than seq_len are front-padded with zeros. NOTE: after scaling,
    # 0 means "median reading", not "missing". never triggers on FD001 (min 31 cycles).
    windows = []
    for _, engine in test.groupby('unit_number', sort=False):
        values = engine[FEATURE_COLS].to_numpy()[-seq_len:]
        pad = seq_len - len(values)
        windows.append(np.pad(values, ((pad, 0), (0, 0))) if pad else values)
    return np.stack(windows)
