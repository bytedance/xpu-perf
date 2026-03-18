# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.utils import is_torch_min_version

def fake_torch_compile(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# GPU & MLU torch compile is disabled, for precision testing.
#jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2
#if is_torch_min_version("2.2.0a0"):
#    jit_fuser = torch.compile
jit_fuser = fake_torch_compile

