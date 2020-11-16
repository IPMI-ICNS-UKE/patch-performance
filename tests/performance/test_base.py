import pytest

from tests.setups import *
from patchperformance.performance.base import BasePatchPerformance


class VoidBasePatchPerformance(BasePatchPerformance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _normalize_input(*args, **kwargs):
        prediction = kwargs.get('prediction', None)
        target = kwargs.get('target', None)

        if prediction is None:
            prediction = args[0]
        if target is None:
            target = args[1]

        return {
            'prediction': prediction,
            'target': target
        }


class DummyBCE:
    def __init__(self):
        pass

    @staticmethod
    def forward(prediction, target, eps=1e-15):
        prediction = prediction.flatten()
        target = target.flatten()
        prediction = np.clip(prediction, eps, 1 - eps)
        n = prediction.shape[0]
        return -1 / n * np.sum(
            (target * np.log(prediction)) + ((1 - target) * np.log(1 - prediction))
        )

    @staticmethod
    def __call__(prediction, target, eps=1e-15):
        return DummyBCE.forward(prediction=prediction, target=target, eps=eps)


class TestBasePatchPerformance:
    dce = DummyBCE()

    @pytest.mark.parametrize("valid_measure", valid_measures)
    def test_measure_setter_valid(self, valid_measure):
        assert VoidBasePatchPerformance(
            loss=self.dce,
            measure=valid_measure
        ) is not None

    def test_measure_setter_invalid(self):
        with pytest.raises(ValueError):
            VoidBasePatchPerformance(
                loss=self.dce,
                measure=invalid_string_input
            )

    @pytest.mark.parametrize("valid_measure", valid_measures)
    def test_track_valid(self, valid_measure):
        assert VoidBasePatchPerformance.track(
            loss=self.dce,
            measure=valid_measure
        ) is not None

    def test_track_invalid(self):
        with pytest.raises(ValueError):
            VoidBasePatchPerformance.track(
                loss=self.dce,
                measure=invalid_string_input
            )

    @pytest.mark.parametrize("valid_measure", valid_measures)
    def test_call(self, valid_measure, dummy_predictions, dummy_targets):
        bpp = VoidBasePatchPerformance.track(
            loss=self.dce,
            measure=valid_measure
        )
        decorated_loss = bpp(prediction=dummy_predictions, target=dummy_targets)
        direct_loss = DummyBCE.forward(prediction=dummy_predictions, target=dummy_targets)
        assert decorated_loss == direct_loss


if __name__ == '__main__':
    dce = DummyBCE()
    bpp = VoidBasePatchPerformance.track(
        loss=dce,
        measure='binary_cross_entropy'
    )
    dummy_predictions = np.random.random((32, 1, 128, 128)).astype(np.float32)
    dummy_predictions[:, :, 48:80, 48:80] = 0.8
    dummy_targets = np.ones_like(dummy_predictions)

    decorated_loss = bpp(prediction=dummy_predictions, target=dummy_targets)
    direct_loss = DummyBCE.forward(prediction=dummy_predictions, target=dummy_targets)
