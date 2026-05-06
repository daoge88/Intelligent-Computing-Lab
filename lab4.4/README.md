# Lab 4.4 — 自定义 mysigmoid 算子（PyTorch C++ 扩展）

## 文件归位

云平台原始目录：`/opt/code_chap_4/exp_4_4_mysigmoid/`

把本目录下的 4 个源文件 cp 到对应位置：

| 本目录文件 | 拷到云上目录 |
|---|---|
| `mysigmoid.cpp`     | `stu_upload/op_mysigmoid/mysigmoid.cpp` |
| `setup.py`          | `stu_upload/op_mysigmoid/setup.py` |
| `test_mysigmoid.py` | `stu_upload/test_mysigmoid.py` |
| `evaluate_cpu.py`   | `stu_upload/evaluate_cpu.py` |

一键命令（在云端 lab 仓库 clone 后执行）：

```bash
LAB=lab4.4
DST=/opt/code_chap_4/exp_4_4_mysigmoid/stu_upload
cp $LAB/mysigmoid.cpp     $DST/op_mysigmoid/
cp $LAB/setup.py          $DST/op_mysigmoid/
cp $LAB/test_mysigmoid.py $DST/
cp $LAB/evaluate_cpu.py   $DST/
```

## 运行步骤

```bash
cd /opt/code_chap_4/exp_4_4_mysigmoid/stu_upload/op_mysigmoid
# 1) 编译 C++ 算子，生成 op_exp.cpython-310-x86_64-linux-gnu.so
python setup.py build_ext --inplace

cd ..
# 2) 算子单元测试 — 输入 (3,512,512)，输出 (1,3,512,512)
python test_mysigmoid.py

# 3) 风格迁移推理 — 输出在 ../out/cpu/0.jpg
python evaluate_cpu.py
```

## 实现要点

- **`mysigmoid.cpp`**：`tensor → vector<float>` 后逐元素算 `1/(1+exp(-x))`，再用 `from_blob().clone()` 还原成 tensor，最后按指导书要求 reshape 成 `[1,3,512,512]`。
- **`setup.py`**：`name='op_exp'`，与现存预编译产物 `op_exp.cpython-310-x86_64-linux-gnu.so` 命名一致；编译命令 `python setup.py build_ext --inplace`。
- **`test_mysigmoid.py`**：`from op_mysigmoid import op_exp`，调用 `op_exp.mysigmoid_cpu(rand)`。注意输入 `(3,512,512)` 经过算子后会被 reshape 成 `(1,3,512,512)` —— 这是 cpp 端的硬编码 reshape 决定的，符合指导书 4.4.5.1 节描述。
- **`evaluate_cpu.py`**：基于 4.2 通过版的 TransNet（`bias=False` Conv + 5×ResBlock + 2× Upsample），**末尾去掉 `nn.Sigmoid()`**，在 `forward` 里改用 `op_exp.mysigmoid_cpu(x)` 完成激活。

## 与 4.2 的差异

```diff
 class TransNet(nn.Module):
     def __init__(self):
         self.layer = nn.Sequential(
             ...
             nn.Conv2d(32, 3, kernel_size=9, padding=4),
-            nn.Sigmoid(),
         )

     def forward(self, x):
-        return torch.nn.functional.relu(self.layer(x))
+        x = self.layer(x)
+        out = op_exp.mysigmoid_cpu(x)
+        return out
```

权重文件 `fst.pth` 与 4.2 共用，结构必须保持一致（除最末 `nn.Sigmoid` 外）。
