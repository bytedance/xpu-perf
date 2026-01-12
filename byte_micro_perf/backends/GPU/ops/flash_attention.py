import sys
import pathlib
from functools import partial
import torch

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    class FA2Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.attn_mode == "prefill":
                self.prefill_init()
            elif self.attn_mode == "decode":
                self.decode_init()

        def prefill_init(self):
            if not (
                self.dtype == "bfloat16" and 
                self.compute_dtype == "bfloat16" and 
                self.cache_dtype == "bfloat16"    
            ):
                raise ValueError(
                    "FlashAttentionOp only support bfloat16 dtype and compute_dtype and cache_dtype"
                )

            if not(
                self.cache_type == "linear"
            ):
                raise ValueError(
                    "FlashAttentionOp only support linear cache_type"
                )

            if not (
                self.batch_size == 1 and 
                self.cache_lens[0] == 0
            ):
                raise ValueError(
                    "FlashAttentionOp only support prefill with batch_size == 1 and cache_len == 0"
                )
            
            self._run_func = self.prefill_run


        def prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"].view(self.batch_size, self.num_tokens, self.q_head_num, self.head_dim)
            k_cache = tensor_mapping["k_cache"].view(self.batch_size, self.num_tokens, self.kv_head_num, self.head_dim)
            v_cache = tensor_mapping["v_cache"].view(self.batch_size, self.num_tokens, self.kv_head_num, self.head_dim)
            
            out = flash_attn_func(
                q, 
                k_cache, v_cache, 
                causal=self.is_causal
            )
            return out
            


        def decode_init(self):
            if not (
                self.dtype == "bfloat16" and 
                self.compute_dtype == "bfloat16" and 
                self.cache_dtype == "bfloat16"
            ):
                raise ValueError(
                    "FlashAttentionOp only support bfloat16 dtype and compute_dtype and cache_dtype"
                )
            
            if not(
                self.cache_type == "linear"
            ):
                raise ValueError(
                    "FlashAttentionOp only support linear cache_type"
                )
            
            self._run_func = self.decode_run


        def decode_run(self, tensor_mapping):
            q = tensor_mapping["q"].view(self.batch_size, self.max_q_len, self.q_head_num, self.head_dim)
            slot_mapping = tensor_mapping["slot_mapping"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            kv_lens = tensor_mapping["kv_lens"]

            k_cache = tensor_mapping["k_cache"].view(self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim)
            v_cache = tensor_mapping["v_cache"].view(self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim)

            out = flash_attn_with_kvcache(
                q, 
                k_cache, v_cache, 
                cache_seqlens=kv_lens, 
                cache_batch_idx=slot_mapping, 
                causal=self.is_causal
            )
            
            return out


    OP_MAPPING["flash_attn_v2"] = FA2Op
except:
    pass



try:
    from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache

    class FA3Op(FA2Op):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

    OP_MAPPING["flash_attn_v3"] = FA3Op
except:
    pass