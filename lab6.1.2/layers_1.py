# file: layers_1.py
import numpy as np


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
        # 动量缓存 (用于 SGD with Momentum)
        self.vw = np.zeros_like(self.weight)
        self.vb = np.zeros_like(self.bias)

    def forward(self, input):  # 前向传播的计算
        self.input = input
        # TODO: 全连接层的前向传播，计算输出结果
        self.output = np.zeros((input.shape[0], self.num_output))
        for n in range(input.shape[0]):
            self.output[n] = np.matmul(self.input[n], self.weight) + self.bias[0]
        return self.output

    def backward(self, top_diff):  # 反向传播的计算
        # TODO: 全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):  # 参数更新
        # 使用 SGD with Momentum 进行参数更新
        momentum = 0.9
        if not hasattr(self, 'vw'):
            self.vw = np.zeros_like(self.weight)
            self.vb = np.zeros_like(self.bias)
        self.vw = momentum * self.vw - lr * self.d_weight
        self.vb = momentum * self.vb - lr * self.d_bias
        self.weight = self.weight + self.vw
        self.bias = self.bias + self.vb

    def load_param(self, weight, bias):  # 参数加载
        self.weight = weight
        self.bias = bias

    def save_param(self):  # 参数保存
        return self.weight, self.bias


class ReLULayer(object):
    def forward(self, input):  # 前向传播的计算
        self.input = input
        # TODO: ReLU层的前向传播，计算输出结果
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):  # 反向传播的计算
        # TODO: ReLU层的反向传播，计算本层损失
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff


class SoftmaxLossLayer(object):
    def forward(self, input):  # 前向传播的计算
        # TODO: Softmax损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):  # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label.astype(int)] = 1.0
        loss = -np.sum(np.log(self.prob + 1e-12) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):  # 反向传播的计算
        # TODO: Softmax损失层的反向传播，计算本层损失
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
