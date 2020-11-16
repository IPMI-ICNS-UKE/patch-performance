
import numpy as np

valid_measures = [
    'cross_entropy',
    'l2'
]

invalid_string_input = 'rghrjgrdgdgsacvxcvysavbbzfbfd'

dummy_predictions = np.random.random((32, 1, 128, 128)).astype(np.float32)
dummy_predictions[:, :, 48:80, 48:80] = 0.8

dummy_targets = np.ones_like(dummy_predictions)


