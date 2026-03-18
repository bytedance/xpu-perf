# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

def replace_type_str():
    from functools import wraps

    def wrapper_type(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            output = fn(*args, **kwargs)
            if isinstance(output, str):
                if output == 'torch.mlu.FloatTensor':
                    output = 'torch.cuda.FloatTensor'
                elif output == 'torch.mlu.BFloat16Tensor':
                    output = 'torch.cuda.BFloat16Tensor'
                elif output == 'torch.mlu.HalfTensor':
                    output = 'torch.cuda.HalfTensor'
            return output

        return decorated
    torch.Tensor.type = wrapper_type(torch.Tensor.type)

device_type=None

try:
    import torch_mlu
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    from torch_mlu.utils.gpu_migration import migration
    replace_type_str()
    device_type='mlu'
except ImportError:
    device_type='gpu'


from .global_vars import get_args
from .global_vars import get_signal_handler
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_wandb_writer
from .global_vars import get_one_logger
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .initialize  import initialize_megatron
from .training import pretrain, get_model, get_train_valid_test_num_samples

from .utils import (print_rank_0,
                    is_last_rank,
                    print_rank_last)
