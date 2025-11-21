import itertools
import math
import os
import sys
import pathlib
from functools import partial
from time import time
from typing import Any, List, Optional
import torch
import types

import habana_frameworks.torch as ht

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}

try:

    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    os.environ["PT_HPU_SDPA_Q_SLICE_SIZE"]="1024"
    os.environ["PT_HPU_QKV_SLICE_SEQ_LEN_THLD"]="1024"

    class FusedSDPAOP(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            # currently not support c8
            if self.dtype != self.cache_dtype:
                raise NotImplementedError("not support q_dtype != cache_dtype")

            if self.mode == "prefill" or self.mode == "decode":

                self.input_tensor_info.update(
                    {
                        "q": OpTensorInfo(
                            shape=[
                                self.batch_size,
                                self.q_head_num,
                                self.num_tokens // self.batch_size,
                                self.head_dim,
                            ],
                            dtype=self.torch_dtype,
                            device=self.backend.get_torch_device_name(),
                        ),
                        "k_cache": OpTensorInfo(
                            shape=[
                                self.batch_size,
                                self.kv_head_num,
                                self.max_kv_len,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                        ),
                        "v_cache": OpTensorInfo(
                            shape=[
                                self.batch_size,
                                self.kv_head_num,
                                self.max_kv_len,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                        ),
                    }
                )
            else:
                raise NotImplementedError("not support")

        def flash_attention_run(self, tensor_mapping):
            # get pre-allocated tensors
            q = tensor_mapping["q"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            # print(f"{q.shape=},{k_cache.shape=},{v_cache.shape=}")

            out = FusedSDPA.apply(q, k_cache, v_cache, None, 0.0, True, None, 'fast')

            return out
    OP_MAPPING["fused_sdpa"] = FusedSDPAOP
except:
    pass
