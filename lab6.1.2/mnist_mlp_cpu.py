# file: mnist_mlp_cpu.py
import numpy as np
import os
import struct
import time

try:
    from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
except ModuleNotFoundError:
    from .layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer


def _find_mnist_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(base, 'mnist_data'),
        os.path.join(base, '..', 'mnist_data'),
        os.path.join(base, '..', '..', 'mnist_data'),
    ]:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.join(base, 'mnist_data')


MNIST_DIR = _find_mnist_dir()
TRAIN_DATA = 'train-images-idx3-ubyte'
TRAIN_LABEL = 'train-labels-idx1-ubyte'
TEST_DATA = 't10k-images-idx3-ubyte'
TEST_LABEL = 't10k-labels-idx1-ubyte'


class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16,
                 out_classes=10, lr=0.01, max_epoch=2, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def load_mnist(self, file_dir, is_images='True'):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data

    def load_data(self):
        print('Loading MNIST data from %s ...' % MNIST_DIR)
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
        np.random.shuffle(self.train_data)

    def build_model(self):
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        prob = self.softmax.forward(h3)
        return prob

    def backward(self):
        dloss = self.softmax.backward()
        dh3 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        num_samples = self.test_data.shape[0]
        for idx in range((num_samples + self.batch_size - 1) // self.batch_size):
            start = idx * self.batch_size
            end = min(start + self.batch_size, num_samples)
            batch_images = self.test_data[start:end, :-1]
            start_time = time.time()
            prob = self.forward(batch_images)
            end_time = time.time()
            pred_labels = np.argmax(prob, axis=1)
            pred_results[start:end] = pred_labels
            print('CPU inferencing time: %f' % (end_time - start_time))
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)
        return accuracy


def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, e = 256, 128, 10
    mlp = MNIST_MLP(batch_size=10000, hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()

    # 数据预处理: 归一化 + 标准化（与训练时一致）
    mlp.train_data = mlp.train_data.astype(np.float64)
    mlp.test_data = mlp.test_data.astype(np.float64)
    mlp.train_data[:, :-1] /= 255.0
    mlp.test_data[:, :-1] /= 255.0

    params = np.load(param_dir, allow_pickle=True).item()
    if 'mean' in params and 'std' in params:
        mlp.train_data[:, :-1] = (mlp.train_data[:, :-1] - params['mean']) / params['std']
        mlp.test_data[:, :-1] = (mlp.test_data[:, :-1] - params['mean']) / params['std']

    # 不训练，直接加载预训练权重
    # mlp.train()
    # mlp.save_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    mlp.load_model(param_dir)

    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
