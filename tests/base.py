import pytest

from tests.setups import *
from patchperformance.base import _BasePatchPerformance


class VoidBasePatchPerformance(_BasePatchPerformance):
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
        prediction = np.clip(prediction, eps, 1-eps)
        n = prediction.shape[0]
        return -1 / n * np.sum(
            (target * np.log(prediction)) + ((1 - target) * np.log(1 - prediction))
        )

    @staticmethod
    def __call__(prediction, target, eps=1e-15):
        return DummyBCE.forward(prediction=prediction, target=target, eps=eps)


class TestBasePatchPerformance:
    @pytest.mark.parametrize("valid_measure", valid_measures)
    def test_measure_setter_valid(self, valid_measure):
        assert VoidBasePatchPerformance(
            loss=DummyBCE,
            measure=valid_measure
        ) is not None

    def test_measure_setter_invalid(self):
        with pytest.raises(ValueError):
            VoidBasePatchPerformance(
                loss=DummyBCE,
                measure=invalid_string_input
            )

    @pytest.mark.parametrize("valid_measure", valid_measures)
    def test_track_valid(self, valid_measure):
        assert VoidBasePatchPerformance.track(
            loss=DummyBCE,
            measure=valid_measure
        ) is not None

    def test_track_invalid(self):
        with pytest.raises(ValueError):
            VoidBasePatchPerformance.track(
                loss=DummyBCE,
                measure=invalid_string_input
            )

    @pytest.mark.parametrize("valid_measure", valid_measures)
    def test_call(self, valid_measure):
        bpp = VoidBasePatchPerformance.track(
            loss=DummyBCE,
            measure=valid_measure
        )
        decorated_loss = bpp(prediction=dummy_predictions, target=dummy_targets)
        direct_loss = DummyBCE.forward(prediction=dummy_predictions, target=dummy_targets)
        assert decorated_loss == direct_loss
        assert bpp._n_samples == 1
        assert bpp._measure_sum.shape == dummy_predictions.shape[1:]
        assert bpp._measure_sum.sum() / bpp._n_samples == direct_loss
