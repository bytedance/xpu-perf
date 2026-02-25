import sys
import warnings
import os
import re
import ast
import shutil
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)


ck_dir = os.environ.get("CK_DIR", "/mnt/raid0/shengnxu/composable_kernel")
this_dir = os.path.dirname(os.path.abspath(__file__))
bd_dir = f"{this_dir}/build"
PACKAGE_NAME = 'rocmKernels'
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True

FORCE_CXX11_ABI = False


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found."
    )


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def rename_cpp_to_cu(pths):
    ret = []
    dst = bd_dir
    for pth in pths:
        if not os.path.exists(pth):
            continue
        for entry in os.listdir(pth):
            if os.path.isdir(f'{pth}/{entry}'):
                continue
            newName = entry
            if entry.endswith(".cpp") or entry.endswith(".cu"):
                newName = entry.replace(".cpp", ".cu")
                ret.append(f'{dst}/{newName}')
            shutil.copy(f'{pth}/{entry}', f'{dst}/{newName}')
    return ret


def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a",
                     "gfx940", "gfx941", "gfx942", "gfx1100"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"


cmdclass = {}
ext_modules = []

if IS_ROCM:
    # use codegen get code dispatch
    if not os.path.exists(bd_dir):
        os.makedirs(bd_dir)

    print(f"\n\ntorch.__version__  = {torch.__version__}\n\n")
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag.append("-DOLD_GENERATOR_PATH")
    if os.path.exists(ck_dir):
        generator_flag.append("-DFIND_CK")

    cc_flag = []

    archs = os.getenv("GPU_ARCHS", "native").split(";")
    validate_and_update_archs(archs)

    cc_flag = [f"--offload-arch={arch}" for arch in archs]

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    renamed_sources = rename_cpp_to_cu([f"{this_dir}/csrc"])
    renamed_ck_srcs = rename_cpp_to_cu(
        [f"{ck_dir}/example/ck_tile/02_layernorm2d/instances",
         f"{this_dir}/csrc/impl/",
         # f'for other kernels'
         ])

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"] + generator_flag,
        "nvcc":
            [
                "-O3", "-std=c++17",
                "-mllvm", "-enable-post-misched=0",
                "-DUSE_PROF_API=1",
                "-D__HIP_PLATFORM_HCC__=1",
                "-D__HIP_PLATFORM_AMD__=1",
                # "-DLEGACY_HIPBLAS_DIRECT",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF_OPERATORS__",
        ]
            + generator_flag
            + cc_flag,
    }

    include_dirs = [
        f"{this_dir}/build",
        f"{ck_dir}/include",
        f"{ck_dir}/library/include",
        f"{ck_dir}/example/ck_tile/02_layernorm2d",
    ]

    ext_modules.append(
        CUDAExtension(
            name=PACKAGE_NAME,
            sources=renamed_sources+renamed_ck_srcs,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
        )
    )
else:
    raise NotImplementedError("Only ROCM is supported")


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / \
                (1024 ** 3)  # free memory in GB
            # each JOB peak memory cost is ~8-9GB when threads = 4
            max_num_jobs_memory = int(free_memory_gb / 9)

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
if os.path.exists(bd_dir):
    shutil.rmtree(bd_dir)
    shutil.rmtree(f"./.eggs")
    shutil.rmtree(f"./{PACKAGE_NAME}.egg-info")

if os.path.exists('./build'):
    shutil.rmtree(f"./build")
