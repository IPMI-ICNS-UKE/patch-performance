import tensorflow as tf
from tensorflow.keras import losses

from patchperformance.measurer.base import BaseMeasurer


class TensorflowMeasurer(BaseMeasurer):
    @staticmethod
    def binary_cross_entropy(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        performance = losses.binary_crossentropy(
            y_true=target,
            y_pred=prediction
        )

        return tf.math.reduce_sum(performance, axis=0)

    @staticmethod
    def l2(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        performance = losses.mean_squared_error(
            y_true=target,
            y_pred=prediction
        )

        return tf.math.reduce_sum(performance, axis=0)
