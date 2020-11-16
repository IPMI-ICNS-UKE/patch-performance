
import numpy as np
import torch
import torch.nn as nn
from patchperformance import TorchPatchPerformance

# pred = np.random.random((32, 1, 128, 128)).astype(np.float32)
#
# predictions = pred.copy()
# predictions[:, :, 48:80, 48:80] = 0.8
predictions = np.ones((32, 1, 128, 128)) * 0.5
targets = np.ones_like(predictions)
predictions = torch.from_numpy(predictions)
targets = torch.from_numpy(targets)
loss = nn.BCELoss()
pp_torch_loss = TorchPatchPerformance.track(loss, measure='binary_cross_entropy')
for i in range(10):
    value = pp_torch_loss(predictions, targets)
    print(value)

pp_th = pp_torch_loss.calculate_performance()