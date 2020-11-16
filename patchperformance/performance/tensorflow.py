from patchperformance.measurer.tensorflow import TensorflowMeasurer
from patchperformance.performance.base import BasePatchPerformance


class TensorflowPatchPerformance(BasePatchPerformance):
    _MEASURER = TensorflowMeasurer

    def __init__(self, loss, measure: str = 'binary_cross_entropy'):
        BasePatchPerformance.__init__(
            self,
            loss=loss,
            measure=measure
        )

    def _normalize_input(self, *args, **kwargs) -> dict:
        prediction = kwargs.get('y_pred', None)
        target = kwargs.get('y_true', None)

        if prediction is None:
            prediction = args[1]
        if target is None:
            target = args[0]

        return {
            'prediction': prediction,
            'target': target
        }
