# rul — shared code for the Jet Engine RUL project
# used by the notebooks (training), the API (serving) and the tests,
# so each fact below is defined exactly once.
#
#   rul.config  — feature list, sequence length, RUL cap, alert thresholds
#   rul.alerts  — RUL → RED / AMBER / GREEN
#   rul.layers  — custom Keras layers (needed to load best_model.keras)
#   rul.data    — CMAPSS loading, RUL labels, engine-wise split, sliding windows
#   rul.model   — the GRU-LSTM + attention architecture, seeding
#   rul.metrics — RMSE / MAE / R² / NASA score
#
# kept deliberately empty so `import rul.layers` doesn't drag in pandas etc.
