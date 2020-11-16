from torch.nn.modules import Module

from patchperformance.measurer.torch import TorchMeasurer
from patchperformance.performance.base import BasePatchPerformance


class TorchPatchPerformance(BasePatchPerformance):
    _MEASURER = TorchMeasurer

    def __init__(self, loss: Module, measure: str = 'binary_cross_entropy'):
        BasePatchPerformance.__init__(
            self,
            loss=loss,
            measure=measure
        )

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def _normalize_input(self, *args, **kwargs) -> dict:
        prediction = kwargs.get('input', None)
        target = kwargs.get('target', None)

        if prediction is None:
            prediction = args[0]
        if target is None:
            target = args[1]

        return {
            'prediction': prediction,
            'target': target
        }
