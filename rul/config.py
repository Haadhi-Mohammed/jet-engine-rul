# project-wide constants — decisions made in 01_exploration.ipynb

# CMAPSS file layout
INDEX_COLS   = ['unit_number', 'time_cycles']
SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
SENSOR_COLS  = [f's_{i}' for i in range(1, 22)]
RAW_COLS     = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# constant or near-constant in FD001 — no information about degradation
SENSORS_TO_DROP = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

# the 14 sensors the model uses, in the order the model expects them
FEATURE_COLS = [s for s in SENSOR_COLS if s not in SENSORS_TO_DROP]

# degradation is only visible in roughly the last 125 cycles
RUL_CAP = 125

# how many past cycles the model looks at
SEQUENCE_LENGTH = 30

# alert levels — RUL below RED_BELOW is RED, below AMBER_BELOW is AMBER
RED_BELOW   = 30
AMBER_BELOW = 60
