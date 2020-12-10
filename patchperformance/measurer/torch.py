from functools import wraps

import torch
import torch.nn.functional as F

from patchperformance.measurer.base import BaseMeasurer


def sum_batch(func):
    """
    A wrapper function/decorator for summing up along the batch dimension.

    Parameters
    ----------
    func : callable
        Torch measure function

    Returns
    -------
    summing_func : callable
        Torch measure function that sums up along the batch dimension
    """

    @wraps(func)
    def with_batch_sum(*args, **kwargs):
        tensor = func(*args, **kwargs)
        return tensor.sum(0)

    return with_batch_sum


class TorchMeasurer(BaseMeasurer):
    @staticmethod
    @sum_batch
    def binary_cross_entropy(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
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
        with torch.no_grad():
            performance = F.binary_cross_entropy(
                input=prediction, target=target, reduction="none"
            )

            return performance

    @staticmethod
    @sum_batch
    def binary_cross_entropy_with_logits(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the binary cross-entropy for given prediction and target tensor.
        Here, the prediction tensor is fed into a Sigmoid layer beforehand.
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
        with torch.no_grad():
            performance = F.binary_cross_entropy_with_logits(
                input=prediction, target=target, reduction="none"
            )

            return performance

    @staticmethod
    @sum_batch
    def l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        with torch.no_grad():
            performance = torch.sqrt((target - prediction) ** 2)

            return performance

    @staticmethod
    def l2_with_logits(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the L2 norm (Euclidean norm) for given prediction and target tensor.
        Here, the prediction tensor is fed into a Sigmoid layer beforehand.
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
        with torch.no_grad():
            prediction = torch.sigmoid(prediction)
            return TorchMeasurer.l2(prediction=prediction, target=target)
