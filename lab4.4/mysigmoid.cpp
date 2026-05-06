// file: mysigmoid.cpp
// Pytorch 扩展头文件的引用
#include <torch/extension.h>
#include <cmath>
#include <vector>
using namespace std;

// mysigmoid_cpu 函数的具体实现
torch::Tensor mysigmoid_cpu(const torch::Tensor & dets) {
  // 将输入的 tensor 转化为浮点类型的 vector
  auto dets_contig = dets.contiguous();
  std::vector<float> input_data(dets_contig.data_ptr<float>(),
                                dets_contig.data_ptr<float>() + dets_contig.numel());
  int input_size = input_data.size();
  // 创建一个浮点类型的 output_data，大小与输入相同
  std::vector<float> output_data(input_size);
  // 对输入向量的每个元素计算 mysigmoid: 1 / (1 + exp(-x))
  for (int i = 0; i < input_size; ++i) {
    output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
  }
  // Create tensor options with dtype float32
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  // Create a tensor from the output vector
  auto foo = torch::from_blob(output_data.data(), {int64_t(output_data.size())}, opts).clone();
  // 将得到的 tensor 重塑为 [1, 3, 512, 512]
  auto output = foo.reshape({1, 3, 512, 512});
  return output;
}

// 算子绑定为 Pytorch 的模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mysigmoid_cpu", &mysigmoid_cpu, "mysigmoid forward (CPU)");
}
