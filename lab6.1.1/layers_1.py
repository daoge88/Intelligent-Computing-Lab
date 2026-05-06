import numpy as np


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros((1, self.num_output))
        self.vw = np.zeros_like(self.weight)
        self.vb = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        momentum = 0.9
        if not hasattr(self, 'vw'):
            self.vw = np.zeros_like(self.weight)
            self.vb = np.zeros_like(self.bias)
        self.vw = momentum * self.vw - lr * self.d_weight
        self.vb = momentum * self.vb - lr * self.d_bias
        self.weight = self.weight + self.vw
        self.bias = self.bias + self.vb

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer(object):
    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff


class SoftmaxLossLayer(object):
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label.astype(int)] = 1.0
        loss = -np.sum(np.log(self.prob + 1e-12) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff


class FCLayer(FullyConnectedLayer):
    def __init__(self, input_dim, output_dim, weight_scale=None):
        super().__init__(input_dim, output_dim)
        std = np.sqrt(2.0 / input_dim) if weight_scale is None else weight_scale
        self.init_param(std=std)

    @property
    def W(self):
        return self.weight

    @W.setter
    def W(self, value):
        self.weight = value

    @property
    def b(self):
        return self.bias

    @b.setter
    def b(self, value):
        self.bias = value

    @property
    def dW(self):
        return self.d_weight

    @property
    def db(self):
        return self.d_bias

    def step(self, lr, momentum=0.9, weight_decay=0.0):
        if weight_decay != 0.0:
            self.d_weight = self.d_weight + weight_decay * self.weight
        self.update_param(lr)


class SoftmaxCrossEntropyLoss(SoftmaxLossLayer):
    def forward(self, logits, labels=None):
        prob = super().forward(logits)
        if labels is None:
            return prob
        return self.get_loss(labels)
