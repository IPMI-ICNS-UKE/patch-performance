import pytest
import numpy as np

valid_measures = [
    'binary_cross_entropy',
    'l2'
]

invalid_string_input = 'rghrjgrdgdgsacvxcvysavbbzfbfd'


@pytest.fixture
def dummy_predictions():
    dummy_predictions = np.random.random((32, 1, 128, 128)).astype(np.float32)
    dummy_predictions[:, :, 48:80, 48:80] = 0.8
    return dummy_predictions


@pytest.fixture
def dummy_targets(dummy_predictions):
    dummy_targets = np.ones_like(dummy_predictions)
    return dummy_targets
