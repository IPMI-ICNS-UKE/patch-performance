from abc import ABC, abstractmethod

import torch.nn.functional as f
from torch.nn import Module


class _BasePatchPerformance(ABC):
    def __init__(self, loss, measure: str = 'cross_entropy'):
        self.loss = loss
        self.measure = measure

        self._n_samples = 0
        self._measure_sum = None

    @property
    def measure(self):
        return self.__measure

    @measure.setter
    def measure(self, value):
        if value == 'cross_entropy':
            def _measure_function(prediction, target):
                return f.binary_cross_entropy(input=prediction, target=target, reduction='none').sum(0)

        elif value == 'l2':
            def _measure_function(prediction, target):
                return f.mse_loss(input=prediction, target=target, reduction='none').sum(0)
        else:
            raise ValueError(f'{value} is not a valid measure')

        self._measure_function = _measure_function
        self.__measure = value

    def _update(self, prediction, target):
        if not self._n_samples:
            self._measure_sum = self._measure_function(prediction=prediction, target=target)
        else:
            self._measure_sum += self._measure_function(prediction=prediction, target=target)

        self._n_samples += target.shape[0]

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

    @classmethod
    def track(cls, loss: Module, measure: str = 'cross_entropy'):
        return cls(loss=loss, measure=measure)
