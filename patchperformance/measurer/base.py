from abc import ABC, abstractmethod


class BaseMeasurer(ABC):
    @staticmethod
    @abstractmethod
    def sum_batch(batch_tensor):
        pass

    @staticmethod
    @abstractmethod
    def binary_cross_entropy(prediction, target):
        pass

    @staticmethod
    @abstractmethod
    def l2(prediction, target):
        pass
