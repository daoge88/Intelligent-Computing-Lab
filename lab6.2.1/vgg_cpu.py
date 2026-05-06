# file: vgg_cpu.py
import numpy as np
import struct
import os
import scipy.io
import time
import sys

import imageio
from PIL import Image

try:
    from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
    from layers_2 import ConvolutionalLayer, FlattenLayer, MaxPoolingLayer
except ImportError:
    from .layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
    from .layers_2 import ConvolutionalLayer, FlattenLayer, MaxPoolingLayer


class VGG19(object):
    def __init__(self, param_path="../../imagenet-vgg-verydeep-19.mat"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        sys.path.append(self.base_dir)
        self.param_path = param_path
        self.param_layer_name = (
            "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
            "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
            "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3",
            "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
            "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5",
            "flatten", "fc6", "relu6", "fc7", "relu7", "fc8", "softmax",
        )

    def build_model(self):
        print("Building VGG19 model ...")
        self.layers = {}

        self.layers["conv1_1"] = ConvolutionalLayer(3, 3, 64, 1, 1)
        self.layers["relu1_1"] = ReLULayer()
        self.layers["conv1_2"] = ConvolutionalLayer(3, 64, 64, 1, 1)
        self.layers["relu1_2"] = ReLULayer()
        self.layers["pool1"] = MaxPoolingLayer(2, 2)

        self.layers["conv2_1"] = ConvolutionalLayer(3, 64, 128, 1, 1)
        self.layers["relu2_1"] = ReLULayer()
        self.layers["conv2_2"] = ConvolutionalLayer(3, 128, 128, 1, 1)
        self.layers["relu2_2"] = ReLULayer()
        self.layers["pool2"] = MaxPoolingLayer(2, 2)

        self.layers["conv3_1"] = ConvolutionalLayer(3, 128, 256, 1, 1)
        self.layers["relu3_1"] = ReLULayer()
        self.layers["conv3_2"] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers["relu3_2"] = ReLULayer()
        self.layers["conv3_3"] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers["relu3_3"] = ReLULayer()
        self.layers["conv3_4"] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers["relu3_4"] = ReLULayer()
        self.layers["pool3"] = MaxPoolingLayer(2, 2)

        self.layers["conv4_1"] = ConvolutionalLayer(3, 256, 512, 1, 1)
        self.layers["relu4_1"] = ReLULayer()
        self.layers["conv4_2"] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers["relu4_2"] = ReLULayer()
        self.layers["conv4_3"] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers["relu4_3"] = ReLULayer()
        self.layers["conv4_4"] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers["relu4_4"] = ReLULayer()
        self.layers["pool4"] = MaxPoolingLayer(2, 2)

        self.layers["conv5_1"] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers["relu5_1"] = ReLULayer()
        self.layers["conv5_2"] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers["relu5_2"] = ReLULayer()
        self.layers["conv5_3"] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers["relu5_3"] = ReLULayer()
        self.layers["conv5_4"] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers["relu5_4"] = ReLULayer()
        self.layers["pool5"] = MaxPoolingLayer(2, 2)

        self.layers["flatten"] = FlattenLayer((512, 7, 7), (25088,), order="matconvnet")
        self.layers["fc6"] = FullyConnectedLayer(25088, 4096)
        self.layers["relu6"] = ReLULayer()
        self.layers["fc7"] = FullyConnectedLayer(4096, 4096)
        self.layers["relu7"] = ReLULayer()
        self.layers["fc8"] = FullyConnectedLayer(4096, 1000)
        self.layers["softmax"] = SoftmaxLossLayer()

    def init_model(self):
        for layer_name in self.param_layer_name:
            layer = self.layers[layer_name]
            if hasattr(layer, "init_param"):
                layer.init_param()

    def load_model(self):
        print("Loading parameters from %s" % self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = np.mean(params["normalization"][0, 0][0], axis=(0, 1)).astype(np.float32)
        print("ImageNet mean:", self.image_mean)

        classes_struct = params["classes"][0, 0]["description"]
        self.labels = [classes_struct[0, idx][0] for idx in range(classes_struct.shape[1])]

        for idx in range(params["layers"].shape[1]):
            layer = params["layers"][0, idx]
            layer_name = layer["name"][0, 0][0]
            layer_type = layer["type"][0, 0][0]
            if layer_name not in self.layers:
                continue
            if "weights" not in layer.dtype.names:
                continue

            weight, bias = layer["weights"][0, 0][0, 0], layer["weights"][0, 0][0, 1]
            if layer_name.startswith("conv"):
                weight = np.transpose(weight, (2, 0, 1, 3))
                bias = bias.reshape(-1)
                self.layers[layer_name].load_param(weight, bias)
            elif layer_name.startswith("fc"):
                weight = weight.reshape(-1, weight.shape[3])
                bias = bias.reshape(1, -1)
                self.layers[layer_name].load_param(weight, bias)
            else:
                raise ValueError("Unsupported weighted layer: %s (%s)" % (layer_name, layer_type))

        del params

    def load_image(self, image_path):
        print("Loading and preprocessing image from " + image_path)
        self.input_image = imageio.imread(image_path)
        pil_img = Image.fromarray(self.input_image)
        pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        self.input_image = np.array(pil_img, dtype=np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(self.input_image, [1] + list(self.input_image.shape))
        self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])

    def forward(self):
        print("Inferencing ...")
        start_time = time.time()
        current = self.input_image
        for layer_name in self.param_layer_name:
            current = self.layers[layer_name].forward(current)
        print("Inference time: %f" % (time.time() - start_time))
        return current

    def evaluate(self):
        prob = self.forward()
        top1 = int(np.argmax(prob[0]))
        print("Classification result: id = %d, prob = %f" % (top1, prob[0, top1]))
        return prob


if __name__ == "__main__":
    vgg = VGG19()
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    vgg.load_image("../../cat1.jpg")
    vgg.evaluate()
