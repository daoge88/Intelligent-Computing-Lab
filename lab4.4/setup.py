# file: setup.py
from setuptools import setup
from torch.utils import cpp_extension

setup(
    # 编译后的链接库名称
    name='op_exp',
    ext_modules=[
        cpp_extension.CppExtension(
            name='op_exp',
            sources=['mysigmoid.cpp'],
        )
    ],
    # 执行编译命令设置
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)

print("generate .so PASS!\n")
