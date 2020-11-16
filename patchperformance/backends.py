from torch.nn.modules import Module

from patchperformance.base import _BasePatchPerformance


class TorchPatchPerformance(_BasePatchPerformance):
    def __init__(self, loss: Module, measure: str = 'cross_entropy'):
        _BasePatchPerformance.__init__(
            self,
            loss=loss,
            measure=measure
        )

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def _normalize_input(self, *args, **kwargs) -> dict:
        prediction = kwargs.get('input', None)
        target = kwargs.get('target', None)

        if not prediction:
            prediction = args[0]
        if not target:
            target = args[1]

        return {
            'prediction': prediction,
            'target': target
        }


class TensorflowPatchPerformance(_BasePatchPerformance):
    def __init__(self, loss, measure: str = 'cross_entropy'):
        _BasePatchPerformance.__init__(
            self,
            loss=loss,
            measure=measure
        )

    def _normalize_input(self, *args, **kwargs) -> dict:
        prediction = kwargs.get('y_pred', None)
        target = kwargs.get('y_true', None)

        if not prediction:
            prediction = args[1]
        if not target:
            target = args[0]

        return {
            'prediction': prediction,
            'target': target
        }


if __name__ == '__main__':
    import torch
    import numpy as np
    import torch.nn as nn

    predictions = np.random.random((32, 1, 128, 128)).astype(np.float32)
    predictions[:, :, 48:80, 48:80] = 0.8

    targets = np.ones_like(predictions)

    predictions = torch.from_numpy(predictions)
    targets = torch.from_numpy(targets)

    loss = nn.BCELoss()
    pp = TorchPatchPerformance.track(loss, measure='cross_entropy')

    forward_func = pp.forward
