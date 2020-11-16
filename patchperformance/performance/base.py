from abc import ABC, abstractmethod

from patchperformance.exceptions import NoPatchesError
from patchperformance.measurer.base import BaseMeasurer


class BasePatchPerformance(ABC):
    _MEASURER = BaseMeasurer

    def __init__(self, loss, measure: str = 'binary_cross_entropy'):
        self.loss = loss
        self.measure = measure

        self.n_patches_seen = 0
        self._measure_sum = None

    @property
    def measure(self):
        return self.__measure

    @measure.setter
    def measure(self, value):
        if value == 'binary_cross_entropy':
            measure_function = self._MEASURER.binary_cross_entropy
        elif value == 'binary_cross_entropy_with_logits':
            measure_function = self._MEASURER.binary_cross_entropy_with_logits
        elif value == 'l2':
            measure_function = self._MEASURER.l2
        elif value == 'l2_with_logits':
            measure_function = self._MEASURER.l2_with_logits
        else:
            raise ValueError(f'{value} is not a valid measure')

        self._measure_function = measure_function
        self.__measure = value

    def _update(self, prediction, target):
        if not self.n_patches_seen:
            self._measure_sum = self._measure_function(
                prediction=prediction,
                target=target
            )
        else:
            self._measure_sum += self._measure_function(
                prediction=prediction,
                target=target
            )

        self.n_patches_seen += target.shape[0]

    @abstractmethod
    def _normalize_input(self, *args, **kwargs) -> dict:
        pass

    def __call__(self, *args, **kwargs):
        self._update(
            **self._normalize_input(*args, **kwargs)
        )

        return self.loss(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.loss, item)

    def calculate_performance(self):
        if self.n_patches_seen:
            return self._measure_sum / self.n_patches_seen
        else:
            raise NoPatchesError(patch_performance=self)

    def reset(self):
        self.n_patches_seen = 0
        self._measure_sum = None

    @classmethod
    def track(cls, loss, measure: str = 'cross_entropy'):
        return cls(loss=loss, measure=measure)
