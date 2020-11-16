import torch
import torch.nn.functional as F

from patchperformance.measurer.base import BaseMeasurer


class TorchMeasurer(BaseMeasurer):
    @staticmethod
    def binary_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        performance = F.binary_cross_entropy(
            input=prediction,
            target=target,
            reduction='none'
        )

        return performance.sum(0)

    @staticmethod
    def l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        performance = F.mse_loss(
            input=prediction,
            target=target,
            reduction='none'
        )

        return performance.sum(0)
