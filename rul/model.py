# GRU-LSTM + attention model — the architecture from 03_training.ipynb

import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from rul.layers import ScaledDotProductAttention


def build_model(sequence_length, n_features,
                n_gru_layers=2, gru_units=64, lstm_units=32,
                dropout_rate=0.2, l2_reg=0.001, reverse_order=False):
    # reverse_order=True puts the LSTM before the GRUs (tests whether order matters)
    inputs = Input(shape=(sequence_length, n_features))
    x = inputs

    def gru_stack(x):
        for i in range(n_gru_layers):
            x = GRU(gru_units, return_sequences=True,
                    kernel_regularizer=l2(l2_reg), name=f'gru_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_gru_{i+1}')(x)
        return x

    def lstm_block(x):
        x = LSTM(lstm_units, return_sequences=True,
                 kernel_regularizer=l2(l2_reg), name='lstm_1')(x)
        return Dropout(dropout_rate, name='dropout_lstm')(x)

    x = lstm_block(gru_stack(x)) if not reverse_order else gru_stack(lstm_block(x))

    # attention over all timesteps, then keep the last one
    x = ScaledDotProductAttention(name='attention')(x)
    x = x[:, -1, :]

    # output head — relu keeps RUL non-negative
    x = Dense(32, activation='swish', name='dense_1')(x)
    x = Dropout(dropout_rate, name='dropout_output')(x)
    outputs = Dense(1, activation='relu', name='rul_output')(x)

    return Model(inputs=inputs, outputs=outputs)


def set_seed(seed: int):
    # same seed → same initial weights, same dropout masks, same batch order
    tf.keras.utils.set_random_seed(seed)          # python, numpy and tensorflow
    tf.config.experimental.enable_op_determinism()
