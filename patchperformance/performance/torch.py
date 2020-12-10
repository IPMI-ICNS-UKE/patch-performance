from torch.nn.modules import Module

from patchperformance.measurer.torch import TorchMeasurer
from patchperformance.performance.base import BasePatchPerformance


class TorchPatchPerformance(BasePatchPerformance):
    _MEASURER = TorchMeasurer

    def __init__(self, loss: Module, measure: str = "binary_cross_entropy"):
        """
        Main class used for tracking the patch performance.

        Parameters
        ----------
        loss : callable
            A Torch loss function that implements `__call__(*args, **kwargs)`
        measure : str
            A string specifying the measure function that should be used. Valid
            measure functions are defined in `TorchMeasurer`.
        """
        BasePatchPerformance.__init__(self, loss=loss, measure=measure)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def _normalize_input(self, *args, **kwargs) -> dict:
        prediction = kwargs.get("input", None)
        target = kwargs.get("target", None)

        if prediction is None:
            try:
                prediction = args[0]
            except IndexError as e:
                raise ValueError(
                    f"No args given: args=[{args}], kwarg " f"keys={{{kwargs}}}"
                )
        if target is None:
            try:
                target = args[1]
            except IndexError as e:
                raise ValueError(
                    f"Insufficient number of args given: args=[{args}], "
                    f"kwarg keys={{{kwargs}}}"
                )

        return {"prediction": prediction, "target": target}
