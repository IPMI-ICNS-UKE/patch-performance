import numpy as np
import pytest

valid_measures = [
    'binary_cross_entropy',
    'l2'
]

invalid_measure = 'invalid_measure'

_dummy_batch = np.random.random((32, 128, 128)).astype(np.float32)


@pytest.fixture(scope='session')
def dummy_torch_predictions():
    dummy_predictions = _dummy_batch[:, np.newaxis]
    dummy_predictions[:, :, 48:80, 48:80] = 0.8
    return dummy_predictions


@pytest.fixture(scope='session')
def dummy_torch_targets(dummy_torch_predictions):
    dummy_targets = np.ones_like(dummy_torch_predictions)
    return dummy_targets


@pytest.fixture(scope='session')
def dummy_tensorflow_predictions():
    dummy_predictions = _dummy_batch[..., np.newaxis]
    dummy_predictions[:, 48:80, 48:80, :] = 0.8
    return dummy_predictions


@pytest.fixture(scope='session')
def dummy_tensorflow_targets(dummy_tensorflow_predictions):
    dummy_targets = np.ones_like(dummy_tensorflow_predictions)
    return dummy_targets
