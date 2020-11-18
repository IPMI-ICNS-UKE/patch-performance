import pytest

tf = pytest.importorskip('tensorflow')

from patchperformance import TensorflowPatchPerformance
from tests.conftest import valid_measures, invalid_measure

tensorflow_losses = [
    tf.keras.losses.BinaryCrossentropy(),
    tf.keras.losses.MeanSquaredError()
]


class TestTensorflowPatchPerformance:

    @pytest.mark.parametrize('measure', valid_measures)
    @pytest.mark.parametrize('loss', tensorflow_losses)
    def test_measure_setter_valid(self, measure, loss):
        patch_performance = TensorflowPatchPerformance(
            loss=loss,
            measure=measure
        )
        assert patch_performance.measure == measure
        assert patch_performance.loss == loss

    @pytest.mark.parametrize('loss', tensorflow_losses)
    def test_measure_setter_invalid(self, loss):
        with pytest.raises(ValueError):
            TensorflowPatchPerformance(
                loss=loss,
                measure=invalid_measure
            )

    @pytest.mark.parametrize('measure', valid_measures)
    @pytest.mark.parametrize('loss', tensorflow_losses)
    def test_track_valid(self, measure, loss):
        patch_performance = TensorflowPatchPerformance.track(
            loss=loss,
            measure=measure
        )
        assert patch_performance.measure == measure
        assert patch_performance.loss == loss

    @pytest.mark.parametrize('loss', tensorflow_losses)
    def test_track_invalid(self, loss):
        with pytest.raises(ValueError):
            TensorflowPatchPerformance.track(
                loss=loss,
                measure=invalid_measure
            )

    @pytest.mark.parametrize('measure', valid_measures)
    @pytest.mark.parametrize('loss', tensorflow_losses)
    def test_call(
            self,
            measure,
            loss,
            dummy_tensorflow_predictions,
            dummy_tensorflow_targets):
        patch_performance = TensorflowPatchPerformance.track(
            loss=loss,
            measure=measure
        )
        dummy_predictions = tf.convert_to_tensor(dummy_tensorflow_predictions)
        dummy_targets = tf.convert_to_tensor(dummy_tensorflow_targets)

        forwarded_loss_value = patch_performance(dummy_predictions, dummy_targets)
        direct_loss_value = loss(dummy_predictions, dummy_targets)

        tf.debugging.assert_near(forwarded_loss_value, direct_loss_value)
        assert patch_performance.n_patches_seen == dummy_predictions.shape[0]
        assert patch_performance._measure_sum.shape == dummy_predictions.shape[1:]
