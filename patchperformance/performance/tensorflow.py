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

        if target is None:
            try:
                target = args[0]
            except IndexError as e:
                raise ValueError(f'No args given: args=[{args}], kwarg keys={{{kwargs}}}')
        if prediction is None:
            try:
                prediction = args[1]
            except IndexError as e:
                raise ValueError(f'Insufficient number of args given: args=[{args}], kwarg keys={{{kwargs}}}')

        return {
            'prediction': prediction,
            'target': target
        }
