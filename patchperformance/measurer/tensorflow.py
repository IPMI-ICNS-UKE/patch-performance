import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses

from patchperformance.measurer.base import BaseMeasurer


class TensorflowMeasurer(BaseMeasurer):
    @staticmethod
    def binary_cross_entropy(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        performance = K.binary_crossentropy(
            target=target,
            output=prediction
        )

        return tf.math.reduce_sum(performance, axis=0)

    @staticmethod
    def l2(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        performance = tf.sqrt((target - prediction) ** 2)

        return tf.math.reduce_sum(performance, axis=0)
