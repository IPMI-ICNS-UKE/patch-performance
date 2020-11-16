try:
    from patchperformance.performance.tensorflow import TensorflowPatchPerformance
except ImportError:
    pass

try:
    from patchperformance.performance.torch import TorchPatchPerformance
except ImportError:
    pass
