# Intelligent-Computing-Lab

智能计算系统课程实验代码（《大模型智能计算系统实验教程 第 2 版》）。每个 `labX.Y/` 目录是一份测试通过的实验提交，文件平铺，方便云平台拷贝。

## 目录

| 目录 | 实验 | 文件 | 备注 |
|---|---|---|---|
| `lab4.1/` | VGG19 推理（PyTorch） | `evaluate_cpu.py`、`evaluate_cnnl_mfus.py`、`generate_pth.py` | |
| `lab4.2/` | 实时风格迁移推理 | `evaluate_cpu.py`、`evaluate_cnnl_mfus.py` | |
| `lab4.3/` | 实时风格迁移训练 | `train.py`、`train-mlu.py` | |
| `lab4.4/` | 自定义 mysigmoid 算子（C++ 扩展） | `mysigmoid.cpp`、`setup.py`、`test_mysigmoid.py`、`evaluate_cpu.py`、`README.md` | 含编译/运行步骤 |
| `lab4.5/` | Transformer 翻译（IWSLT2014） | `modules.py`、`AttModel.py`、`eval.py`、`README.md` | BLEU ≈ 17.10 |
| `lab6.1.1/` | MNIST MLP（CPU 手工实现） | `layers_1.py`、`mnist_mlp_cpu.py` | |
| `lab6.1.2/` | MNIST MLP demo（带预训练权重） | `layers_1.py`、`mnist_mlp_cpu.py`、`mnist_mlp_demo.py`、`weight.npy` | `weight.npy` ~1.8 MB |
| `lab6.2.1/` | VGG19（CPU 手工实现） | `layers_1.py`、`layers_2.py`、`vgg_cpu.py` | |
| `lab6.2.2/` | VGG19（DLP/MLU） | `vgg19_demo.py` | |

## 云平台对应路径

通常拷贝到 `/opt/code_chap*/exp_X_Y_Z/` 下覆盖 `stu_upload/` 或工作目录中的同名空白文件即可。每个 lab 的具体归位方式参考其内 `README.md`（如有），或直接对照框架代码的目录结构。

## 备注

- `lab4.4/`、`lab4.5/` 由本仓库 owner 重新写过；其余 `labX.Y/` 来自 `submitted/` 目录中的历史 zip（早期已通过测试）。
- `submitted/` 不在仓库里 —— 只是本地 zip 归档。
