from abc import ABC, abstractmethod


class BaseMeasurer(ABC):
    @staticmethod
    @abstractmethod
    def sum_batch(batch_tensor):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def binary_cross_entropy(prediction, target):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def binary_cross_entropy_with_logits(prediction, target):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def l2(prediction, target):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def l2_with_logits(prediction, target):
        raise NotImplementedError
