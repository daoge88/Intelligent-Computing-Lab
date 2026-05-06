# file: layers_2.py
import numpy as np


class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride

    def init_param(self, std=0.01):
        self.weight = np.random.normal(
            loc=0.0,
            scale=std,
            size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out),
        ).astype(np.float32)
        self.bias = np.zeros((self.channel_out,), dtype=np.float32)

    def forward(self, input_tensor):
        self.input = input_tensor.astype(np.float32, copy=False)
        if self.padding > 0:
            self.input_pad = np.pad(
                self.input,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )
        else:
            self.input_pad = self.input

        height_out = (self.input_pad.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input_pad.shape[3] - self.kernel_size) // self.stride + 1

        output = np.zeros(
            (self.input_pad.shape[0], self.channel_out, height_out, width_out),
            dtype=np.float32,
        )
        for kernel_h in range(self.kernel_size):
            h_slice = slice(kernel_h, kernel_h + self.stride * height_out, self.stride)
            for kernel_w in range(self.kernel_size):
                w_slice = slice(kernel_w, kernel_w + self.stride * width_out, self.stride)
                patch = self.input_pad[:, :, h_slice, w_slice]
                patch_out = np.tensordot(
                    patch,
                    self.weight[:, kernel_h, kernel_w, :],
                    axes=([1], [0]),
                )
                output += np.transpose(patch_out, (0, 3, 1, 2))

        output += self.bias.reshape(1, -1, 1, 1)
        self.output = output.astype(np.float32, copy=False)
        return self.output

    def load_param(self, weight, bias):
        assert weight.shape == (
            self.channel_in,
            self.kernel_size,
            self.kernel_size,
            self.channel_out,
        )
        assert bias.shape in {(self.channel_out,), (1, self.channel_out)}
        self.weight = weight.astype(np.float32, copy=False)
        self.bias = bias.reshape(-1).astype(np.float32, copy=False)


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input_tensor):
        self.input = input_tensor.astype(np.float32, copy=False)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        output = None
        for kernel_h in range(self.kernel_size):
            h_slice = slice(kernel_h, kernel_h + self.stride * height_out, self.stride)
            for kernel_w in range(self.kernel_size):
                w_slice = slice(kernel_w, kernel_w + self.stride * width_out, self.stride)
                patch = self.input[:, :, h_slice, w_slice]
                if output is None:
                    output = patch.copy()
                else:
                    output = np.maximum(output, patch)

        self.output = output.astype(np.float32, copy=False)
        assert self.output.shape[2] == height_out
        assert self.output.shape[3] == width_out
        return self.output


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape, order="matconvnet"):
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.order = order
        assert np.prod(self.input_shape) == np.prod(self.output_shape)

    def forward(self, input_tensor):
        assert tuple(input_tensor.shape[1:]) == self.input_shape
        if self.order == "matconvnet":
            self.input = np.transpose(input_tensor, (0, 2, 3, 1))
        elif self.order == "nchw":
            self.input = input_tensor
        else:
            raise ValueError("Unsupported flatten order: %s" % self.order)
        self.output = self.input.reshape((self.input.shape[0],) + self.output_shape)
        return self.output.astype(np.float32, copy=False)
