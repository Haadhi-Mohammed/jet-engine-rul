# evaluation metrics for RUL prediction

import numpy as np

from rul.config import RUL_CAP


def nasa_score(y_true, y_pred) -> float:
    """
    PHM08 / CMAPSS scoring function (Saxena et al., 2008). lower is better.
    d = predicted - actual. late predictions (d > 0: the model thinks the engine
    has MORE life than it does) are penalised harder than early ones, because
    they mean maintenance happens after it was needed.
        d < 0:  exp(-d / 13) - 1
        d >= 0: exp( d / 10) - 1
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13), np.exp(d / 10)) - 1))


def evaluate(y_true, y_pred, cap: int = RUL_CAP) -> dict:
    # standard CMAPSS practice: clip both to the cap before scoring, since the
    # model is trained to say "≥ cap" for any healthy engine
    y_true = np.clip(np.asarray(y_true, float), 0, cap)
    y_pred = np.clip(np.asarray(y_pred, float), 0, cap)
    err = y_pred - y_true

    return {
        'rmse':  float(np.sqrt(np.mean(err ** 2))),
        'mae':   float(np.mean(np.abs(err))),
        'r2':    float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        'nasa_score': nasa_score(y_true, y_pred),
        'mean_error': float(err.mean()),            # > 0 = optimistic on average
        'late_predictions':  int(np.sum(err > 0)),  # predicted more life than actual
        'early_predictions': int(np.sum(err < 0)),
    }
