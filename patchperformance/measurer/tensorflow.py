from functools import wraps

import tensorflow as tf
import tensorflow.keras.backend as K

from patchperformance.measurer.base import BaseMeasurer


def sum_batch(func):
    """
    A wrapper function/decorator for summing up along the batch dimension.

    Parameters
    ----------
    func : callable
        Tensorflow measure function

    Returns
    -------
    summing_func : callable
        Tensorflow measure function that sums up along the batch dimension
    """

    @wraps(func)
    def with_batch_sum(*args, **kwargs):
        tensor = func(*args, **kwargs)
        return tf.math.reduce_sum(tensor, axis=0)

    return with_batch_sum


class TensorflowMeasurer(BaseMeasurer):
    @staticmethod
    @sum_batch
    def binary_cross_entropy(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """
        Calculates the binary cross-entropy for given prediction and target tensor.
        The resulting tensor is not reduced to a scalar.

        Parameters
        ----------
        prediction : tensor
            N-d prediction tensor with batch size as first dimension
        target : tensor
            N-d target tensor with batch size as first dimension

        Returns
        -------
        performance : tensor
            Corresponding performance tensor
        """
        performance = K.binary_crossentropy(target=target, output=prediction)

        return performance

    @staticmethod
    @sum_batch
    def l2(prediction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """
        Calculates the L2 norm (Euclidean norm) for given prediction and target tensor.
        The resulting tensor is not reduced to a scalar.

        Parameters
        ----------
        prediction : tensor
            N-d prediction tensor with batch size as first dimension
        target : tensor
            N-d target tensor with batch size as first dimension

        Returns
        -------
        performance : tensor
            Corresponding performance tensor
        """
        performance = tf.sqrt((target - prediction) ** 2)

        return performance
