# Patch Performance
[![Build Status](https://img.shields.io/travis/IPMI-ICNS-UKE/patch-performance/master?style=flat-square)](https://travis-ci.org/IPMI-ICNS-UKE/patch-performance)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=flat-square&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org)

Don't waste your time and money on inefficient patch-based inference!

## Introduction
This is the official implementation of the patch performace technique first published in the following MICCAI 2020 paper (cf. Sec. 2.6 & Fig. 1b):
[Widening the Focus: Biomedical Image Segmentation Challenges and the Underestimated Role of Patch Sampling and Inference Strategies](https://rdcu.be/cbiyL)

## What excatly is patch performance and how to use it?
When dealing with large images (e.g. 2D/3D/4D medical images) your GPU memory is often too small to perform training and inference in an one-step fashion, i.e. feeding the whole image into the network in order to receive the complete segmentation. The most common solution to this problem is dividing the image at hand into smaller and GPU memory-friendly patches extracted in an ordered way. After inference the patches are stitcher back together to obtain the final prediction output.  

In doing so, the following question arises: How to determine the most efficient patch overlap used for extraction? Here, we define efficiency in terms of prediction performance (e.g. overall Dice score in the case of image segmentation) divided by the required inference wall-clock time. In other words: Shifting the prediction patch location just by 1 pixel in each dimension will most likely yield top-performing results but is certainly not efficient in any way.

By calculating the patch performance the optimal patch overlap can be estimated from the resulting spatially resolved map. For some datasets, the mean pixel-wise performance is not dependent on the relative patch coordinate at all, for others there is a strong correlation.
In the former case, running the inference with a larger patch overlap will increase your elapsed wall-clock time whereas the prediciton performance will most likely not profit from it. Consequently, this approach will decrease your overall inference efficiency.

## Installation
To install this package from PyPI please run
```shell
pip install patchperformance
```

## Usage
Tracking the patch performance is easy. Just wrap your loss function with `TorchPatchPerformance`or `TensorflowPatchPerformance`.
Here's an example using PyTorch:
```python
import torch
import torch.nn as nn
from patchperformance import TorchPatchPerformance

# We use binary cross entropy loss here
# but any nn.Module-based loss is fine
loss = nn.BCELoss()

# Wrap your loss. Here, the measure parameter defines
# the function used for evaluating the patch performance.
# Your loss remains unaffected.
loss = TorchPatchPerformance.track(loss, measure='binary_cross_entropy')

# Use your loss as you would normally do
predictions: torch.Tensor = ...
targets: torch.Tensor = ...
loss_value = loss(predictions, targets)

# calcualte the accumulated patch performance, e.g., at
# the end of each epoch
patch_performance: torch.Tensor = loss.calculate_performance()

# Usually the patch performance maps of the first few
# epochs/iterations are meaningless, especially if the network
# is trained from scratch. Thus, the accumulated patch
# performance can be reset by calling
loss.reset()
# We recommend to use patch performance maps of the very last
# epoch/iterations (i.e. after model convergence)
```

## Citation
If you find the patch performance technique and/or this repository useful, please consider citing our corresponding MICCAI 2020 paper:
```
@InProceedings{MadestaSchmitz2020,
author="Madesta, Frederic and Schmitz, R{\"u}diger and R{\"o}sch, Thomas and Werner, Ren{\'e}",
editor="Martel, Anne L. and Abolmaesumi, Purang and Stoyanov, Danail and Mateus, Diana and Zuluaga, Maria A. and Zhou, S. Kevin and Racoceanu, Daniel and Joskowicz, Leo",
title="Widening the Focus: Biomedical Image Segmentation Challenges and the Underestimated Role of Patch Sampling and Inference Strategies",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="289--298",
isbn="978-3-030-59719-1"
}
```
