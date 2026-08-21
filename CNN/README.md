# Convolutional Neural Network Projects

This section contains two CIFAR-10 image-classification notebooks with saved training curves, diagnostic plots, and test results.

- [CIFAR-10 Baseline CNN](01_CIFAR10_Baseline_CNN.ipynb) — test accuracy 0.7292 and macro F1 0.7304, reproduced with a fresh 25-epoch run.
- [CIFAR-10 Improved CNN](02_CIFAR10_Improved_CNN.ipynb) — test accuracy 0.9008, top-3 accuracy 0.9824, and macro F1 0.9006, reproduced from the validation-selected checkpoint.

To rerun them, place the excluded CIFAR-10 files in `datasets/cifar-10/`, A complete improved-model run permits up to 70 CPU-intensive epochs.
