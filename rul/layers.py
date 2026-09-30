import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization


class ScaledDotProductAttention(Layer):
    """
    scaled dot-product self-attention (Q = K = V = input), with a
    residual connection and layer norm.
    learns which timesteps to focus on when predicting RUL.

    best_model.keras stores this layer by name only — the code has to exist
    at load time, so pass it via custom_objects:
        tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def build(self, input_shape):
        # sublayers must be built explicitly so their weights exist before
        # Keras restores them from a saved model
        self.layer_norm.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, units)
        d_k     = tf.cast(tf.shape(x)[-1], tf.float32)
        scale   = tf.math.sqrt(d_k)
        scores  = tf.matmul(x, x, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, x)
        return self.layer_norm(x + context)


CUSTOM_OBJECTS = {'ScaledDotProductAttention': ScaledDotProductAttention}
