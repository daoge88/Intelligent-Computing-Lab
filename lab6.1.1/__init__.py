"""lab6.1.1: 基于三层全连接神经网络实现 MNIST 手写数字分类。"""

from .layers_1 import (
    FCLayer,
    FullyConnectedLayer,
    ReLULayer,
    SoftmaxCrossEntropyLoss,
    SoftmaxLossLayer,
)
from .mnist_mlp_cpu import MLP, MNIST_MLP, build_mnist_mlp, load_mnist, train

__all__ = [
    "FCLayer",
    "FullyConnectedLayer",
    "ReLULayer",
    "SoftmaxCrossEntropyLoss",
    "SoftmaxLossLayer",
    "MLP",
    "MNIST_MLP",
    "build_mnist_mlp",
    "load_mnist",
    "train",
]
