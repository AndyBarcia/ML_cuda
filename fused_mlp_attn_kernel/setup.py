from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_mlp',
    ext_modules=[
        CUDAExtension('attention_mlp', [
            'attention_forward.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)