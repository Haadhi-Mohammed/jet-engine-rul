"""
Train and evaluate the RUL model — reproducible replacement for 03_training.ipynb.

    python -m scripts.train experiment   # 9 configs × 3 seeds, pick the best on VALIDATION
    python -m scripts.train final        # winner × 5 seeds, test ONCE, save artifacts for the API

the test set is never used to make a decision: `experiment` doesn't load it at all,
and `final` only reports on it after every choice has been made.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from rul import data
from rul.config import FEATURE_COLS, SEQUENCE_LENGTH, RUL_CAP, SENSORS_TO_DROP
from rul.metrics import evaluate
from rul.model import build_model, set_seed

ROOT       = Path(__file__).parent.parent
RAW_DIR    = ROOT / 'data' / 'raw'
MODELS_DIR = ROOT / 'models'
REPORTS    = ROOT / 'reports'

SPLIT_SEED = 42      # which engines go to validation — fixed for every run
VAL_CUTS   = {'rul_range': (10, 150), 'cuts_per_engine': 20, 'seed': 0}   # test-like validation
TRAINING = {'learning_rate': 0.001, 'batch_size': 256, 'max_epochs': 50, 'patience': 10}
VERSION  = 'v3'      # v2 = validated on all windows; v3 = test-like validation cuts

# the same 9 configurations as the original notebook experiment
CONFIGS = {
    'run_01': dict(n_gru_layers=2, gru_units=32, lstm_units=32, dropout_rate=0.2),
    'run_02': dict(n_gru_layers=2, gru_units=64, lstm_units=32, dropout_rate=0.2),
    'run_03': dict(n_gru_layers=3, gru_units=64, lstm_units=32, dropout_rate=0.2),
    'run_04': dict(n_gru_layers=3, gru_units=64, lstm_units=64, dropout_rate=0.2),
    'run_05': dict(n_gru_layers=4, gru_units=64, lstm_units=64, dropout_rate=0.2),  # dissertation
    'run_06': dict(n_gru_layers=4, gru_units=64, lstm_units=64, dropout_rate=0.3),
    'run_07': dict(n_gru_layers=3, gru_units=64, lstm_units=64, dropout_rate=0.3),
    'run_08': dict(n_gru_layers=2, gru_units=64, lstm_units=64, dropout_rate=0.3),
    'run_09': dict(n_gru_layers=2, gru_units=64, lstm_units=64, dropout_rate=0.2, reverse_order=True),
}


def describe(cfg: dict) -> str:
    gru, lstm = f"{cfg['n_gru_layers']}GRU({cfg['gru_units']})", f"LSTM({cfg['lstm_units']})"
    order = f"{lstm}+{gru}" if cfg.get('reverse_order') else f"{gru}+{lstm}"
    return f"{order} d={cfg['dropout_rate']}"


# ---- data ----
def prepare(include_test: bool):
    train, test, true_rul = data.load_cmapss(RAW_DIR)
    train = data.add_train_rul(train)
    tr, val = data.split_engines(train, val_fraction=0.2, seed=SPLIT_SEED)

    scaler = data.fit_scaler(tr)        # fitted on the 80 training engines only
    X_tr,  y_tr  = data.create_sequences(data.scale(tr,  scaler))
    val_scaled   = data.scale(val, scaler)
    # validation mirrors the test design: each engine cut "some time prior to failure",
    # scored on its last window. used for early stopping AND model selection.
    X_val, y_val = data.cut_windows(val_scaled, **VAL_CUTS)
    # every window of the validation engines — the v2 protocol, kept for comparison only
    X_val_all, y_val_all = data.create_sequences(val_scaled)

    d = dict(X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
             X_val_all=X_val_all, y_val_all=y_val_all, scaler=scaler,
             n_train_engines=tr['unit_number'].nunique(), n_val_engines=val['unit_number'].nunique())
    if include_test:
        d['X_test'] = data.last_windows(data.scale(test, scaler))
        d['y_test'] = true_rul
    return d


# ---- one training run ----
class MlflowEpochLogger(tf.keras.callbacks.Callback):
    # sends loss / val_loss to MLflow after every epoch, so the curves update live in the UI
    def __init__(self, mlflow):
        super().__init__()
        self.mlflow = mlflow

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.mlflow.log_metrics({k: float(v) for k, v in logs.items()
                                 if k in ('loss', 'val_loss', 'mae', 'val_mae')}, step=epoch + 1)


def train_one(cfg: dict, seed: int, d: dict, extra_callbacks=()):
    set_seed(seed)
    model = build_model(SEQUENCE_LENGTH, len(FEATURE_COLS), **cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(TRAINING['learning_rate']),
                  loss='mse', metrics=['mae'])
    history = model.fit(
        d['X_tr'], d['y_tr'],
        validation_data=(d['X_val'], d['y_val']),
        epochs=TRAINING['max_epochs'], batch_size=TRAINING['batch_size'],
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=TRAINING['patience'], restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            *extra_callbacks,
        ],
        verbose=0,
    )
    val_loss = history.history['val_loss']
    info = {
        'best_epoch': int(np.argmin(val_loss)) + 1,   # the epoch whose weights were kept
        'epochs_run': len(val_loss),
    }
    y_val_pred = model.predict(d['X_val'], verbose=0).flatten()
    info['val'] = evaluate(d['y_val'], y_val_pred)
    info['val_all_windows_rmse'] = evaluate(
        d['y_val_all'], model.predict(d['X_val_all'], verbose=0).flatten())['rmse']
    return model, info


def mlflow_setup(experiment: str):
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    return mlflow


# ---- step 1: choose a configuration using validation only ----
def run_experiment(seeds, only, out_csv, use_mlflow):
    d = prepare(include_test=False)
    print(f"train: {d['n_train_engines']} engines / {len(d['X_tr'])} windows | "
          f"val: {d['n_val_engines']} engines / {len(d['X_val'])} test-like cuts")
    mlflow = mlflow_setup(f'jet_engine_rul_{VERSION}_selection') if use_mlflow else None

    rows = []
    for name, cfg in CONFIGS.items():
        if only and name not in only:
            continue
        for seed in seeds:
            t0 = time.time()
            _, info = train_one(cfg, seed, d)
            row = {'run': name, 'config': describe(cfg), 'seed': seed,
                   'val_rmse': info['val']['rmse'], 'val_mae': info['val']['mae'],
                   'val_nasa': info['val']['nasa_score'],
                   'val_all_windows_rmse': info['val_all_windows_rmse'],
                   'best_epoch': info['best_epoch'], 'epochs_run': info['epochs_run'],
                   'seconds': round(time.time() - t0)}
            rows.append(row)
            print(f"  {name} seed={seed}  val RMSE {row['val_rmse']:.3f}  "
                  f"best epoch {row['best_epoch']}/{row['epochs_run']}  ({row['seconds']}s)", flush=True)
            if mlflow:
                with mlflow.start_run(run_name=f'{name}_seed{seed}'):
                    mlflow.log_params({**cfg, **TRAINING, 'seed': seed, 'split_seed': SPLIT_SEED})
                    mlflow.log_metrics({f'val_{k}': v for k, v in info['val'].items()} |
                                       {'best_epoch': info['best_epoch']})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = (df.groupby(['run', 'config'])['val_rmse']
                 .agg(['mean', 'std', 'min', 'max']).sort_values('mean').reset_index())
    print('\nranked by MEAN validation RMSE across seeds:')
    print(summary.round(3).to_string(index=False))
    winner = summary.iloc[0]['run']
    print(f'\nselected: {winner} — {describe(CONFIGS[winner])}')
    return winner


# ---- step 2: train the winner, evaluate on test once, save artifacts ----
def run_final(run_name, seeds, use_mlflow):
    cfg = CONFIGS[run_name]
    d = prepare(include_test=True)
    mlflow = mlflow_setup(f'jet_engine_rul_{VERSION}_final') if use_mlflow else None

    results, best = [], None
    for seed in seeds:
        if mlflow:
            # one MLflow run per seed, opened BEFORE training so the curves stream in live
            with mlflow.start_run(run_name=f'{run_name}_seed{seed}'):
                mlflow.log_params({**cfg, **TRAINING, 'seed': seed, 'split_seed': SPLIT_SEED})
                model, info = train_one(cfg, seed, d, extra_callbacks=[MlflowEpochLogger(mlflow)])
                y_pred = np.clip(model.predict(d['X_test'], verbose=0).flatten(), 0, RUL_CAP)
                info['test'] = evaluate(d['y_test'], y_pred)
                mlflow.log_metrics({f'val_{k}': v for k, v in info['val'].items()} |
                                   {f'test_{k}': v for k, v in info['test'].items()} |
                                   {'best_epoch': info['best_epoch']})
        else:
            model, info = train_one(cfg, seed, d)
            y_pred = np.clip(model.predict(d['X_test'], verbose=0).flatten(), 0, RUL_CAP)
            info['test'] = evaluate(d['y_test'], y_pred)
        info['seed'] = seed
        results.append(info)
        print(f"  seed={seed}  val RMSE {info['val']['rmse']:.3f}  test RMSE {info['test']['rmse']:.3f}"
              f"  NASA {info['test']['nasa_score']:.0f}", flush=True)
        # the deployed model is chosen by VALIDATION, not by test
        if best is None or info['val']['rmse'] < best[1]['val']['rmse']:
            best = (model, info, y_pred)

    model, info, y_pred = best
    test_keys = ['rmse', 'mae', 'r2', 'nasa_score']
    spread = {k: {'mean': float(np.mean([r['test'][k] for r in results])),
                  'std':  float(np.std([r['test'][k] for r in results], ddof=1))}
              for k in test_keys}
    version = f"{VERSION}-{run_name}-seed{info['seed']}"

    save_artifacts(model, d, y_pred, cfg, info, spread, version, run_name)

    report = {'selected_run': run_name, 'config': cfg, 'deployed_seed': info['seed'],
              'model_version': version, 'test_across_seeds': spread, 'per_seed': results}
    (REPORTS / 'final_metrics.json').write_text(json.dumps(report, indent=2))

    if mlflow:
        # summary run: the deployed model + the across-seed spread, with the files attached
        with mlflow.start_run(run_name=f'DEPLOYED_{version}'):
            mlflow.log_params({**cfg, **TRAINING, 'seed': info['seed'], 'split_seed': SPLIT_SEED})
            mlflow.log_metrics({f'val_{k}': v for k, v in info['val'].items()} |
                               {f'test_{k}': v for k, v in info['test'].items()} |
                               {f'test_{k}_mean': s['mean'] for k, s in spread.items()} |
                               {f'test_{k}_std': s['std'] for k, s in spread.items()})
            mlflow.log_artifact(str(MODELS_DIR / 'best_model.keras'))
            mlflow.log_artifact(str(REPORTS / 'final_metrics.json'))

    print(f"\ndeployed: {version} (best validation RMSE of {len(seeds)} seeds)")
    print(f"  its test:  " + '  '.join(f"{k} {info['test'][k]:.3f}" for k in test_keys))
    print(f"  all seeds: " + '  '.join(f"{k} {s['mean']:.3f} ± {s['std']:.3f}" for k, s in spread.items()))


def save_artifacts(model, d, y_pred, cfg, info, spread, version, run_name):
    model.save(MODELS_DIR / 'best_model.keras')

    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(d['scaler'], f)
    with open(MODELS_DIR / 'feature_cols.pkl', 'wb') as f:
        pickle.dump(FEATURE_COLS, f)
    with open(MODELS_DIR / 'model_config.pkl', 'wb') as f:
        pickle.dump({'feature_cols': FEATURE_COLS, 'sequence_length': SEQUENCE_LENGTH,
                     'n_features': len(FEATURE_COLS), 'rul_cap': RUL_CAP,
                     'sensors_dropped': SENSORS_TO_DROP}, f)

    # fleet data for the API — scaled with the NEW scaler
    np.save(MODELS_DIR / 'fleet_sequences.npy', d['X_test'])
    np.save(MODELS_DIR / 'fleet_true_rul.npy',  d['y_test'])
    np.save(MODELS_DIR / 'y_pred_test.npy',     y_pred.astype(np.float32))

    # SHAP background: 100 random TRAINING windows
    rng = np.random.default_rng(42)
    np.save(MODELS_DIR / 'shap_background.npy',
            d['X_tr'][rng.choice(len(d['X_tr']), size=100, replace=False)])

    best_config = {
        'model_version': version,
        'architecture': {k: cfg.get(k, False) for k in
                         ['n_gru_layers', 'gru_units', 'lstm_units', 'dropout_rate', 'reverse_order']}
                        | {'l2_reg': 0.001},
        'training': TRAINING | {'seed': info['seed'], 'best_epoch': info['best_epoch'],
                                'epochs_run': info['epochs_run'], 'split_seed': SPLIT_SEED,
                                'train_engines': d['n_train_engines'], 'val_engines': d['n_val_engines']},
        # the deployed model's own numbers (keys kept for the API)
        'performance': {f'test_{k}': round(v, 4) for k, v in info['test'].items()
                        if k in ('rmse', 'mae', 'r2', 'nasa_score')},
        'performance_across_seeds': {k: {m: round(v, 4) for m, v in s.items()} for k, s in spread.items()},
        'validation': {k: round(v, 4) for k, v in info['val'].items() if k in ('rmse', 'mae', 'r2')},
        'evaluation_notes': 'model chosen on a 20-engine validation split, scored like the test set: '
                            f"{VAL_CUTS['cuts_per_engine']} cuts per engine with true RUL uniform in "
                            f"{list(VAL_CUTS['rul_range'])} (test design per Saxena et al. 2008), last "
                            f'window only; test set (100 engines, RUL clipped at {RUL_CAP}) used once, after selection',
        'validation_protocol': {k: list(v) if isinstance(v, tuple) else v for k, v in VAL_CUTS.items()},
        'data': {'feature_cols': FEATURE_COLS, 'sequence_length': SEQUENCE_LENGTH,
                 'n_features': len(FEATURE_COLS), 'rul_cap': RUL_CAP, 'sensors_dropped': SENSORS_TO_DROP},
    }
    (MODELS_DIR / 'best_model_config.json').write_text(json.dumps(best_config, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('experiment')
    e.add_argument('--seeds', type=int, default=3)
    e.add_argument('--only', nargs='*', help='run only these configs, e.g. run_02 run_08')
    e.add_argument('--out', default=str(REPORTS / f'experiments_{VERSION}.csv'))
    e.add_argument('--no-mlflow', action='store_true')
    f = sub.add_parser('final')
    f.add_argument('--run', required=True, help='config name chosen by `experiment`')
    f.add_argument('--seeds', type=int, default=5)
    f.add_argument('--no-mlflow', action='store_true')
    args = p.parse_args()

    if args.cmd == 'experiment':
        run_experiment(range(args.seeds), args.only, args.out, not args.no_mlflow)
    else:
        run_final(args.run, range(args.seeds), not args.no_mlflow)
