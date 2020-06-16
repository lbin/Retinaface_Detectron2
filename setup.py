#!/usr/bin/env python

import glob
import os
from distutils.core import Extension, setup

import numpy
import torch
from Cython.Build import cythonize
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

requirements = ["torch", "torchvision"]

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

bbox_oevelaps_extensions = [
    Extension(
        "mmdet.evaluation.bbox",
        ["mmdet/evaluation/box_overlaps.pyx"],
        include_dirs=[numpy.get_include()],
    )
]


def get_dcnv2_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "mmdet/layers/DCNv2/src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    os.environ["CC"] = "g++"
    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        # raise NotImplementedError('Cuda is not available')
        pass

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "_ext",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="mmdet",
    version="0.1.0",
    author="lbin",
    url="https://github.com/lbin/Retinaface_Mobilenet_Pytorch",
    description="mmdet",
    # packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "termcolor>=1.1",
        "Pillow",  # you can also use pillow-simd for better performance
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore",
        "future",  # used by caffe2
        "pydot",  # used to save caffe2 SVGs
    ],
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": ["flake8", "isort", "black==19.3b0", "flake8-bugbear", "flake8-comprehensions"],
    },
    ext_modules=[cythonize(bbox_oevelaps_extensions)[0], get_dcnv2_extensions()[0]],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
