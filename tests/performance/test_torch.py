import pytest
import torch.nn as nn
import torch

from tests.setups import *
from patchperformance import TorchPatchPerformance

torch_losses = [
    nn.BCELoss(),
    nn.BCEWithLogitsLoss()
]

class TestBasePatchPerformance:

    @pytest.mark.parametrize("valid_measure", valid_measures)
    @pytest.mark.parametrize("loss", torch_losses)
    def test_measure_setter_valid(self, valid_measure, loss):
        assert TorchPatchPerformance(
            loss=loss,
            measure=valid_measure
        ) is not None

    @pytest.mark.parametrize("loss", torch_losses)
    def test_measure_setter_invalid(self, loss):
        with pytest.raises(ValueError):
            TorchPatchPerformance(
                loss=loss,
                measure=invalid_string_input
            )

    @pytest.mark.parametrize("valid_measure", valid_measures)
    @pytest.mark.parametrize("loss", torch_losses)
    def test_track_valid(self, valid_measure, loss):
        assert TorchPatchPerformance.track(
            loss=loss,
            measure=valid_measure
        ) is not None

    @pytest.mark.parametrize("loss", torch_losses)
    def test_track_invalid(self, loss):
        with pytest.raises(ValueError):
            TorchPatchPerformance.track(
                loss=loss,
                measure=invalid_string_input
            )

    @pytest.mark.parametrize("valid_measure", valid_measures)
    @pytest.mark.parametrize("loss", torch_losses)
    def test_call(self, valid_measure, loss, dummy_predictions, dummy_targets):
        tpp = TorchPatchPerformance.track(
            loss=loss,
            measure=valid_measure
        )
        dummy_predictions = torch.from_numpy(dummy_predictions)
        dummy_targets = torch.from_numpy(dummy_targets)
        decorated_loss = tpp(dummy_predictions, dummy_targets)
        direct_loss = loss.forward(dummy_predictions, dummy_targets)
        print(decorated_loss)
        print(direct_loss)
        assert decorated_loss == direct_loss
        assert tpp.n_patches_seen == dummy_predictions.shape[0]
        assert tpp._measure_sum.shape == dummy_predictions.shape[1:]
        # assert tpp._measure_sum.sum() / (tpp.n_patches_seen * np.prod(dummy_predictions.shape[1:])) == direct_loss


if __name__ == '__main__':
    dce = nn.BCELoss()
    tpp = TorchPatchPerformance.track(
        loss=dce,
        measure='binary_cross_entropy'
    )
    dummy_predictions = np.random.random((32, 1, 128, 128)).astype(np.float32)
    dummy_predictions[:, :, 48:80, 48:80] = 0.8
    dummy_targets = np.ones_like(dummy_predictions)
    dummy_predictions = torch.from_numpy(dummy_predictions)
    dummy_targets = torch.from_numpy(dummy_targets)

    decorated_loss = tpp(dummy_predictions, dummy_targets)
    direct_loss = dce.forward(dummy_predictions, dummy_targets)
