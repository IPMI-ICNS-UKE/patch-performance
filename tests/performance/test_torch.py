import pytest

torch = pytest.importorskip('torch')

import torch.nn as nn

from patchperformance import TorchPatchPerformance
from tests.conftest import valid_measures, invalid_measure

torch_losses = [
    nn.BCELoss(),
    nn.BCEWithLogitsLoss()
]


class TestTorchPatchPerformance:

    @pytest.mark.parametrize('measure', valid_measures)
    @pytest.mark.parametrize('loss', torch_losses)
    def test_measure_setter_valid(self, measure, loss):
        patch_performance = TorchPatchPerformance(
            loss=loss,
            measure=measure
        )
        assert patch_performance.measure == measure
        assert patch_performance.loss == loss

    @pytest.mark.parametrize('loss', torch_losses)
    def test_measure_setter_invalid(self, loss):
        with pytest.raises(ValueError):
            TorchPatchPerformance(
                loss=loss,
                measure=invalid_measure
            )

    @pytest.mark.parametrize('measure', valid_measures)
    @pytest.mark.parametrize('loss', torch_losses)
    def test_track_valid(self, measure, loss):
        patch_performance = TorchPatchPerformance.track(
            loss=loss,
            measure=measure
        )
        assert patch_performance.measure == measure
        assert patch_performance.loss == loss

    @pytest.mark.parametrize('loss', torch_losses)
    def test_track_invalid(self, loss):
        with pytest.raises(ValueError):
            TorchPatchPerformance.track(
                loss=loss,
                measure=invalid_measure
            )

    @pytest.mark.parametrize('measure', valid_measures)
    @pytest.mark.parametrize('loss', torch_losses)
    def test_call(
            self,
            measure,
            loss,
            dummy_torch_predictions,
            dummy_torch_targets):
        patch_performance = TorchPatchPerformance.track(
            loss=loss,
            measure=measure
        )
        dummy_predictions = torch.from_numpy(dummy_torch_predictions)
        dummy_targets = torch.from_numpy(dummy_torch_targets)

        forwarded_loss_value = patch_performance(dummy_predictions, dummy_targets)
        direct_loss_value = loss.forward(dummy_predictions, dummy_targets)

        assert torch.allclose(forwarded_loss_value, direct_loss_value)
        assert patch_performance.n_patches_seen == dummy_predictions.shape[0]
        assert patch_performance._measure_sum.shape == dummy_predictions.shape[1:]

    @pytest.mark.parametrize('measure', valid_measures)
    @pytest.mark.parametrize('loss', torch_losses)
    def test_simple_training(
            self,
            measure,
            loss,
            dummy_torch_targets
    ):
        dummy_torch_targets = torch.from_numpy(dummy_torch_targets)
        model = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3), padding=1),
            nn.Sigmoid()
        )

        patch_performance = TorchPatchPerformance(
            loss=loss,
            measure=measure
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        model.train(True)
        for epoch in range(10):
            optimizer.zero_grad()
            dummy_output = model(dummy_torch_targets)
            loss_value = patch_performance(dummy_output, dummy_torch_targets)
            loss_value.backward()

            patch_performance.calculate_performance().cpu().numpy()
            patch_performance.reset()

            assert patch_performance.n_patches_seen == 0
            assert patch_performance._measure_sum is None
