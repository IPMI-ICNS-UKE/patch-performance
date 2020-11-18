import torch
import torch.nn.functional as F

from patchperformance.measurer.base import BaseMeasurer


class TorchMeasurer(BaseMeasurer):
    @staticmethod
    def binary_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            performance = F.binary_cross_entropy(
                input=prediction,
                target=target,
                reduction='none'
            )

            return performance.sum(0)

    @staticmethod
    def binary_cross_entropy_with_logits(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            performance = F.binary_cross_entropy_with_logits(
                input=prediction,
                target=target,
                reduction='none'
            )

            return performance.sum(0)

    @staticmethod
    def l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            performance = torch.sqrt((target - prediction) ** 2)

            return performance.sum(0)

    @staticmethod
    def l2_with_logits(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction = torch.sigmoid(prediction)
            return TorchMeasurer.l2(prediction=prediction, target=target)
