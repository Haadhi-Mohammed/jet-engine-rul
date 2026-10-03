"""
Regenerate the result figures in reports/ from the saved v2 results.

    python -m scripts.make_figures

reads reports/experiments_v2.csv (selection experiment) and the deployed model's
test predictions in models/ — no training, runs in seconds.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul.config import RUL_CAP
from rul.metrics import evaluate

ROOT    = Path(__file__).parent.parent
REPORTS = ROOT / 'reports'
MODELS  = ROOT / 'models'

# reference palette (light surface)
SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_2     = '#52514e'
GRID      = '#e4e3df'
HIGHLIGHT = '#2a78d6'   # categorical slot 1 — the selected config / early predictions
SECOND    = '#eb6834'   # categorical slot 2 — late predictions
NEUTRAL   = '#a3a29c'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'text.color': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def experiment_comparison():
    df = pd.read_csv(REPORTS / 'experiments_v2.csv')
    stats = (df.groupby(['run', 'config'])['val_rmse'].agg(['mean', 'std'])
               .sort_values('mean').reset_index())
    winner = stats.iloc[0]['run']

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for y, row in enumerate(stats.itertuples()):
        color = HIGHLIGHT if row.run == winner else NEUTRAL
        seeds = df.loc[df['run'] == row.run, 'val_rmse']
        # individual seeds (hollow) + mean ± std (filled)
        ax.scatter(seeds, [y] * len(seeds), s=36, facecolors='none', edgecolors=color, linewidths=1.2, zorder=2)
        ax.errorbar(row.mean, y, xerr=row.std, fmt='o', ms=8, color=color, ecolor=color,
                    elinewidth=2, capsize=0, zorder=3)
        ax.text(stats['mean'].max() + stats['std'].max() + 0.12, y, f"{row.mean:.2f} ± {row.std:.2f}",
                va='center', fontsize=9, color=INK if row.run == winner else INK_2)

    ax.set_yticks(range(len(stats)), [f"{r.run}  {r.config}" for r in stats.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel('validation RMSE (cycles) — lower is better')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title('Model selection on 20 held-out validation engines\n'
                 'hollow = each of 3 seeds · filled = mean ± std · blue = selected',
                 loc='left', fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(REPORTS / 'experiment_comparison.png', dpi=150)
    plt.close(fig)


def predicted_vs_actual():
    y_true = np.clip(np.load(MODELS / 'fleet_true_rul.npy'), 0, RUL_CAP)
    y_pred = np.load(MODELS / 'y_pred_test.npy')
    final  = json.loads((REPORTS / 'final_metrics.json').read_text())
    seeds  = final['test_across_seeds']
    m      = evaluate(y_true, y_pred)
    err    = y_pred - y_true
    late   = err > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # predicted vs actual
    ax1.plot([0, RUL_CAP], [0, RUL_CAP], color=INK_2, linewidth=1, linestyle='--', zorder=1)
    ax1.scatter(y_true[~late], y_pred[~late], s=40, color=HIGHLIGHT, edgecolors=SURFACE, linewidths=1.5,
                label=f'early / exact ({(~late).sum()})', zorder=2)
    ax1.scatter(y_true[late], y_pred[late], s=40, color=SECOND, edgecolors=SURFACE, linewidths=1.5,
                label=f'late — predicted more life than actual ({late.sum()})', zorder=2)
    ax1.set_xlim(-3, RUL_CAP + 5); ax1.set_ylim(-3, RUL_CAP + 5)
    ax1.set_xlabel(f'actual RUL (cycles, capped at {RUL_CAP})')
    ax1.set_ylabel('predicted RUL (cycles)')
    ax1.grid(color=GRID, linewidth=0.8); ax1.set_axisbelow(True)
    ax1.legend(loc='lower right', frameon=False, fontsize=9)   # empty corner — no data underneath
    ax1.set_title(f"Deployed model ({final['model_version']}) — 100 test engines\n"
                  f"RMSE {m['rmse']:.2f} · R² {m['r2']:.3f} · NASA score {m['nasa_score']:.0f}",
                  loc='left', fontsize=11)

    # error distribution
    bins = np.arange(np.floor(err.min() / 5) * 5, np.ceil(err.max() / 5) * 5 + 5, 5)
    ax2.hist(err[err <= 0], bins=bins, color=HIGHLIGHT, edgecolor=SURFACE, linewidth=2, label='early')
    ax2.hist(err[err > 0],  bins=bins, color=SECOND,    edgecolor=SURFACE, linewidth=2, label='late')
    ax2.axvline(0, color=INK_2, linewidth=1, linestyle='--')
    ax2.set_xlabel('prediction error = predicted − actual (cycles)')
    ax2.set_ylabel('engines')
    ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))   # whole engines only
    ax2.grid(axis='y', color=GRID, linewidth=0.8); ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title(f"Error distribution — mean {m['mean_error']:+.1f} cycles (optimistic)\n"
                  f"across 5 seeds: RMSE {seeds['rmse']['mean']:.2f} ± {seeds['rmse']['std']:.2f}",
                  loc='left', fontsize=11)

    fig.tight_layout()
    fig.savefig(REPORTS / 'predicted_vs_actual.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    experiment_comparison()
    predicted_vs_actual()
    print('saved reports/experiment_comparison.png and reports/predicted_vs_actual.png')
