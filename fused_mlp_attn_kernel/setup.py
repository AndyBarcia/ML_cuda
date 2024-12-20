from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_mlp',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('attention_mlp_cuda', [
            'attention_forward.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)