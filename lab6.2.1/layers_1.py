# file: layers_1.py
import numpy as np


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):
        self.weight = np.random.normal(
            loc=0.0,
            scale=std,
            size=(self.num_input, self.num_output),
        ).astype(np.float32)
        self.bias = np.zeros((1, self.num_output), dtype=np.float32)

    def forward(self, input_tensor):
        self.input = input_tensor
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        if not hasattr(self, "vw"):
            self.vw = np.zeros_like(self.weight)
            self.vb = np.zeros_like(self.bias)
        momentum = 0.9
        self.vw = momentum * self.vw - lr * self.d_weight
        self.vb = momentum * self.vb - lr * self.d_bias
        self.weight = self.weight + self.vw
        self.bias = self.bias + self.vb

    def load_param(self, weight, bias):
        assert weight.shape == (self.num_input, self.num_output)
        assert bias.shape in {(self.num_output,), (1, self.num_output)}
        self.weight = weight.astype(np.float32, copy=False)
        self.bias = bias.reshape(1, -1).astype(np.float32, copy=False)

    def save_param(self):
        return self.weight, self.bias


class ReLULayer(object):
    def forward(self, input_tensor):
        self.input = input_tensor
        return np.maximum(0.0, input_tensor)

    def backward(self, top_diff):
        return top_diff * (self.input > 0)


class SoftmaxLossLayer(object):
    def forward(self, input_tensor):
        input_max = np.max(input_tensor, axis=1, keepdims=True)
        input_exp = np.exp(input_tensor - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label.astype(int)] = 1.0
        loss = -np.sum(np.log(self.prob + 1e-12) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        return (self.prob - self.label_onehot) / self.batch_size
