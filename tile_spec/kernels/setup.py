"""
TileSpec CUDA Kernels Setup

Build:
    cd tile_spec && pip install .

Or for development:
    cd tile_spec && pip install -e .
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tile_spec_kernels",
    version="0.1.0",
    description="CUDA kernels for TileSpec speculative decoding",
    ext_modules=[
        CUDAExtension(
            name="tile_spec_kernels",
            sources=["csrc/ragged_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
