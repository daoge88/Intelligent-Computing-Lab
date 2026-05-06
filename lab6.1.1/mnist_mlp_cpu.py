import gzip
import os
import struct

import numpy as np

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
    def __init__(self, batch_size=100, input_size=784, hidden1=768, hidden2=384, hidden3=192,
                 out_classes=10, lr=0.05, max_epoch=14, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.train_data = np.zeros((0, input_size + 1), dtype=np.float64)
        self.test_data = np.zeros((0, input_size + 1), dtype=np.float64)

    def _open_mnist_file(self, file_dir):
        if os.path.exists(file_dir):
            return open(file_dir, 'rb')
        gz_path = file_dir + '.gz'
        if os.path.exists(gz_path):
            return gzip.open(gz_path, 'rb')
        raise FileNotFoundError(file_dir)

    def load_mnist(self, file_dir, is_images=True):
        with self._open_mnist_file(file_dir) as bin_file:
            bin_data = bin_file.read()
        if is_images:
            fmt_header = '>iiii'
            _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            _, num_images = struct.unpack_from(fmt_header, bin_data, 0)
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
        self.train_data = np.append(train_images, train_labels, axis=1).astype(np.float64)
        self.test_data = np.append(test_images, test_labels, axis=1).astype(np.float64)
        self.train_label = self.train_data[:, -1]
        self.test_label = self.test_data[:, -1]
        self.train_labels = self.train_label
        self.test_labels = self.test_label

    def shuffle_data(self):
        np.random.shuffle(self.train_data)

    def build_model(self):
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.hidden3)
        self.relu3 = ReLULayer()
        self.fc4 = FullyConnectedLayer(self.hidden3, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.loss_fn = self.softmax
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]
        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3, self.relu3, self.fc4]
        self.trainable_layers = self.update_layer_list

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        h3 = self.relu3.forward(h3)
        h4 = self.fc4.forward(h3)
        prob = self.softmax.forward(h4)
        return prob

    def backward(self):
        dloss = self.softmax.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.relu3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def step(self, lr, momentum=0.9, weight_decay=0.0):
        if weight_decay != 0.0:
            self.fc1.d_weight += weight_decay * self.fc1.weight
            self.fc2.d_weight += weight_decay * self.fc2.weight
            self.fc3.d_weight += weight_decay * self.fc3.weight
            self.fc4.d_weight += weight_decay * self.fc4.weight
        self.update(lr)

    def predict(self, x, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        pred_results = np.zeros([x.shape[0]])
        for idx in range((x.shape[0] + batch_size - 1) // batch_size):
            start = idx * batch_size
            end = min(start + batch_size, x.shape[0])
            prob = self.forward(x[start:end])
            pred_results[start:end] = np.argmax(prob, axis=1)
        return pred_results.astype(np.int64)

    def evaluate(self, x=None, y=None):
        if x is None or y is None:
            pred_results = self.predict(self.test_data[:, :-1])
            accuracy = np.mean(pred_results == self.test_data[:, -1])
        else:
            pred_results = self.predict(x)
            accuracy = np.mean(pred_results == y)
        print('Accuracy in test set: %f' % accuracy)
        return accuracy

    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        params['w4'], params['b4'] = self.fc4.save_param()
        np.save(param_dir, params)

    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        self.fc4.load_param(params['w4'], params['b4'])

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))


class MLP(MNIST_MLP):
    pass


def random_shift(imgs, max_shift=1):
    batch = imgs.shape[0]
    imgs_2d = imgs.reshape(batch, 28, 28)
    dx = np.random.randint(-max_shift, max_shift + 1, size=batch)
    dy = np.random.randint(-max_shift, max_shift + 1, size=batch)
    shifted = np.empty_like(imgs_2d)
    for i in range(batch):
        shifted[i] = np.roll(np.roll(imgs_2d[i], dx[i], axis=1), dy[i], axis=0)
    return shifted.reshape(batch, 784)


def load_mnist(data_dir=MNIST_DIR):
    mlp = MNIST_MLP()
    base_backup = globals()['MNIST_DIR']
    try:
        globals()['MNIST_DIR'] = data_dir
        mlp.load_data()
    finally:
        globals()['MNIST_DIR'] = base_backup
    return (
        mlp.train_data[:, :-1].astype(np.float32),
        mlp.train_data[:, -1].astype(np.uint8),
        mlp.test_data[:, :-1].astype(np.float32),
        mlp.test_data[:, -1].astype(np.uint8),
    )


def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, h3, e = 768, 384, 192, 14
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, hidden3=h3, max_epoch=e, batch_size=100, lr=0.05)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()

    mlp.fc1.weight = np.random.normal(0, np.sqrt(2.0 / 784), (784, h1))
    mlp.fc2.weight = np.random.normal(0, np.sqrt(2.0 / h1), (h1, h2))
    mlp.fc3.weight = np.random.normal(0, np.sqrt(2.0 / h2), (h2, h3))
    mlp.fc4.weight = np.random.normal(0, np.sqrt(2.0 / h3), (h3, 10))

    train_images = mlp.train_data[:, :-1].astype(np.float64) / 255.0
    train_labels = mlp.train_data[:, -1].astype(np.float64)
    test_images = mlp.test_data[:, :-1].astype(np.float64) / 255.0
    test_labels = mlp.test_data[:, -1].astype(np.float64)

    train_mean = np.mean(train_images, axis=0)
    train_std = np.std(train_images, axis=0) + 1e-8
    train_images = (train_images - train_mean) / train_std
    test_images = (test_images - train_mean) / train_std

    mlp.train_data = np.column_stack([train_images, train_labels])
    mlp.test_data = np.column_stack([test_images, test_labels])
    mlp.train_label = mlp.train_data[:, -1]
    mlp.test_label = mlp.test_data[:, -1]
    mlp.train_labels = mlp.train_label
    mlp.test_labels = mlp.test_label

    max_batch = mlp.train_data.shape[0] // mlp.batch_size
    current_lr = 0.05
    for idx_epoch in range(mlp.max_epoch):
        mlp.shuffle_data()
        epoch_loss = 0.0
        for idx_batch in range(max_batch):
            start = idx_batch * mlp.batch_size
            end = start + mlp.batch_size
            batch_images = mlp.train_data[start:end, :-1]
            batch_labels = mlp.train_data[start:end, -1]
            batch_images = random_shift(batch_images, max_shift=1)
            mlp.forward(batch_images)
            loss = mlp.softmax.get_loss(batch_labels)
            mlp.backward()
            mlp.update(current_lr)
            epoch_loss += loss
        avg_loss = epoch_loss / max_batch
        print('Epoch %d/%d  loss: %.4f  lr: %.5f' % (idx_epoch + 1, e, avg_loss, current_lr))
        current_lr *= 0.97
    return mlp


def train(*args, **kwargs):
    mlp = build_mnist_mlp()
    return mlp, mlp.evaluate()


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
