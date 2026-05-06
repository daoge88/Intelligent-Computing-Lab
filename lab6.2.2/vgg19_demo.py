# -*- coding: UTF-8 -*-
import pycnnl
import time
import numpy as np
import os
import scipy.io
from PIL import Image
import imageio


class VGG19(object):
    def __init__(self):
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []
        self.filter_quant_params = []

    def _make_vector(self, values):
        vector = pycnnl.IntVector(len(values))
        for idx, value in enumerate(values):
            vector[idx] = int(value)
        return vector

    def build_model(self, param_path="../../imagenet-vgg-verydeep-19.mat"):
        self.param_path = param_path

        self.net.setInputShape(1, 3, 224, 224)

        self.net.createConvLayer("conv1_1", self._make_vector([1, 3, 224, 224]), 64, 3, 1, 1, 1)
        self.net.createReLuLayer("relu1_1")
        self.net.createConvLayer("conv1_2", self._make_vector([1, 64, 224, 224]), 64, 3, 1, 1, 1)
        self.net.createReLuLayer("relu1_2")
        self.net.createPoolingLayer("pool1", self._make_vector([1, 64, 224, 224]), 2, 2)

        self.net.createConvLayer("conv2_1", self._make_vector([1, 64, 112, 112]), 128, 3, 1, 1, 1)
        self.net.createReLuLayer("relu2_1")
        self.net.createConvLayer("conv2_2", self._make_vector([1, 128, 112, 112]), 128, 3, 1, 1, 1)
        self.net.createReLuLayer("relu2_2")
        self.net.createPoolingLayer("pool2", self._make_vector([1, 128, 112, 112]), 2, 2)

        self.net.createConvLayer("conv3_1", self._make_vector([1, 128, 56, 56]), 256, 3, 1, 1, 1)
        self.net.createReLuLayer("relu3_1")
        self.net.createConvLayer("conv3_2", self._make_vector([1, 256, 56, 56]), 256, 3, 1, 1, 1)
        self.net.createReLuLayer("relu3_2")
        self.net.createConvLayer("conv3_3", self._make_vector([1, 256, 56, 56]), 256, 3, 1, 1, 1)
        self.net.createReLuLayer("relu3_3")
        self.net.createConvLayer("conv3_4", self._make_vector([1, 256, 56, 56]), 256, 3, 1, 1, 1)
        self.net.createReLuLayer("relu3_4")
        self.net.createPoolingLayer("pool3", self._make_vector([1, 256, 56, 56]), 2, 2)

        self.net.createConvLayer("conv4_1", self._make_vector([1, 256, 28, 28]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu4_1")
        self.net.createConvLayer("conv4_2", self._make_vector([1, 512, 28, 28]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu4_2")
        self.net.createConvLayer("conv4_3", self._make_vector([1, 512, 28, 28]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu4_3")
        self.net.createConvLayer("conv4_4", self._make_vector([1, 512, 28, 28]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu4_4")
        self.net.createPoolingLayer("pool4", self._make_vector([1, 512, 28, 28]), 2, 2)

        self.net.createConvLayer("conv5_1", self._make_vector([1, 512, 14, 14]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu5_1")
        self.net.createConvLayer("conv5_2", self._make_vector([1, 512, 14, 14]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu5_2")
        self.net.createConvLayer("conv5_3", self._make_vector([1, 512, 14, 14]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu5_3")
        self.net.createConvLayer("conv5_4", self._make_vector([1, 512, 14, 14]), 512, 3, 1, 1, 1)
        self.net.createReLuLayer("relu5_4")
        self.net.createPoolingLayer("pool5", self._make_vector([1, 512, 14, 14]), 2, 2)

        self.net.createMlpLayer("fc6", self._make_vector([1, 1, 1, 25088]), self._make_vector([1, 1, 25088, 4096]), self._make_vector([1, 1, 1, 4096]))
        self.net.createReLuLayer("relu6")

        self.net.createMlpLayer("fc7", self._make_vector([1, 1, 1, 4096]), self._make_vector([1, 1, 4096, 4096]), self._make_vector([1, 1, 1, 4096]))
        self.net.createReLuLayer("relu7")

        self.net.createMlpLayer("fc8", self._make_vector([1, 1, 1, 4096]), self._make_vector([1, 1, 4096, 1000]), self._make_vector([1, 1, 1, 1000]))

        self.net.createSoftmaxLayer("softmax", self._make_vector([1, 1, 1000]), 1)

    def load_model(self):
        print("Loading parameters from file " + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params["normalization"][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))

        for idx in range(self.net.size()):
            if "conv" in self.net.getLayerName(idx):
                weight, bias = params["layers"][0][idx][0][0][0][0]
                weight = np.transpose(weight, [3, 0, 1, 2]).flatten().astype(np.float64)
                bias = bias.reshape(-1).astype(np.float64)
                self.net.loadParams(idx, weight, bias)
            if "fc" in self.net.getLayerName(idx):
                weight, bias = params["layers"][0][idx][0][0][0][0]
                weight = weight.reshape(
                    [
                        weight.shape[0] * weight.shape[1] * weight.shape[2],
                        weight.shape[3],
                    ]
                )
                weight = weight.flatten().astype(np.float64)
                bias = bias.reshape(-1).astype(np.float64)
                self.net.loadParams(idx, weight, bias)

    def load_image(self, image_dir):
        self.image = image_dir
        image_mean = np.array([123.68, 116.779, 103.939])
        print("Loading and preprocessing image from " + image_dir)
        input_image = imageio.imread(image_dir)
        pli_img = Image.fromarray(input_image)
        pli_img = pli_img.resize((224, 224), Image.Resampling.LANCZOS)
        input_image = np.array(pli_img, dtype=np.float64)
        input_image -= image_mean
        input_image = np.reshape(input_image, [1] + list(input_image.shape))
        input_data = input_image.flatten().astype(np.float64)
        self.net.setInputData(input_data)

    def forward(self):
        return self.net.forward()

    def get_top5(self, label):
        start = time.time()
        self.forward()
        end = time.time()

        result = self.net.getOutputData()

        labels = []
        with open("../synset_words.txt", "r") as f:
            labels = f.readlines()

        top1 = False
        top5 = False
        print("------ Top 5 of " + self.image + " ------")
        prob = sorted(list(result), reverse=True)[:6]
        if result.index(prob[0]) == label:
            top1 = True
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if idx == label:
                top5 = True
            print("%f - " % top + labels[idx].strip())

        print("inference time: %f" % (end - start))
        return top1, top5

    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        start = time.time()
        with open(file_list, "r") as f:
            file_list = f.readlines()
            total_num = len(file_list)
            for line in file_list:
                image = line.split()[0].strip()
                label = int(line.split()[1].strip())
                self.load_image(image)
                top1, top5 = self.get_top5(label)
                if top1:
                    top1_num += 1
                if top5:
                    top5_num += 1
        end = time.time()

        print("Global accuracy : ")
        print("accuracy1: %f (%d/%d) " % (float(top1_num) / float(total_num), top1_num, total_num))
        print("accuracy5: %f (%d/%d) " % (float(top5_num) / float(total_num), top5_num, total_num))
        print("Total execution time: %f" % (end - start))


if __name__ == "__main__":
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    vgg.evaluate("../file_list")
