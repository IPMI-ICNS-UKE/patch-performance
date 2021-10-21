from itertools import product
from math import ceil

import numpy as np


def find_optimal_margin(performance: np.ndarray, threshold: float = 0.05):
    margins = []
    performances = []
    for margin in product(*tuple(range(ceil(s / 2 - 1)) for s in performance.shape)):
        slicing = tuple(slice(m, s - m) for m, s in zip(margin, performance.shape))
        mean_performance = performance[slicing].mean()
        margins.append(margin)
        performances.append(mean_performance)

    performances = np.array(performances)
    margins = np.array(margins)

    best_performance = performances.min()
    threshold_performance = best_performance * (1 + threshold)

    mask = performances <= threshold_performance
    acceptable_performances = performances[mask]
    acceptable_margins = margins[mask]

    # TODO: find best margin
    margin_sizes = acceptable_margins.prod(axis=1)
    smallest_margin_idx = np.argmin(margin_sizes)

    return acceptable_margins[smallest_margin_idx], acceptable_performances[smallest_margin_idx]


# def find_optimal_stride(patch_performance: np.ndarray, threshold: float = 0.01):
#     optimal_margins = []
#     for i_axis in range(patch_performance.ndim):
#         reduce_axes = tuple(i for i in range(patch_performance.ndim) if i != i_axis)
#         axis_performance = performance.mean(axis=reduce_axes)
#         margins, performances = find_optimal_margin(
#             axis_performance, threshold=threshold
#         )
#         optimal_margins.append(margins[0])
#
#     return optimal_margins


if __name__ == '__main__':
    # performance = np.load("tests/data/steady_patch_performance.npy")
    # performance = performance.squeeze()
    # optimal_margins = find_optimal_stride(performance, threshold=0.01)
    # print(optimal_margins)

    performance = np.load("tests/data/variable_patch_performance.npy")
    performance = performance.squeeze()
    m, p = find_optimal_margin(performance, threshold=0.20)
