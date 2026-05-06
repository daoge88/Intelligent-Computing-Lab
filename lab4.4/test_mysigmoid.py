import numpy as np
import torch
import torchvision
import numpy as np
# 导入自定义连接库
from op_mysigmoid import op_exp


def mysigmoid_cpu(rand):
    rand = rand.contiguous()
    # 调用 mysigmoid 函数对 rand 进行处理得到输出结果 output
    output = op_exp.mysigmoid_cpu(rand)
    return output.contiguous()


def test_mysigmoid():
    torch.manual_seed(12345)
    rand = (torch.randn(3, 512, 512, dtype=torch.float32).abs() + 1)
    # 调用 mysigmoid_cpu 函数对 rand 进行处理得到输出结果 output_cpu
    output_cpu = mysigmoid_cpu(rand)
    print("------------------mysigmoid test completed----------------------")
    print("input: ", rand)
    print("input_size:", rand.size())
    print("output: ", output_cpu)
    print("output_size:", output_cpu.size())

    print("TEST mysigmoid PASS!\n")


test_mysigmoid()
