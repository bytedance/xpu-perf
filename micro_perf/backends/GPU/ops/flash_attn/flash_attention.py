import sys
import pathlib
from functools import partial
import torch

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[4])
)


from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size



try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    @ProviderRegistry.register_vendor_impl("flash_attention", "fa2")
    class FA2Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.attn_mode == "prefill":
                if self.dtype == "bfloat16" \
                    and self.dst_dtype == "bfloat16" \
                    and self.cache_dtype == "bfloat16" \
                    and self.qk_compute_dtype == "bfloat16" \
                    and self.pv_compute_dtype == "bfloat16" \
                    and self.cache_type == "linear":
                    pass
                else:
                    raise ValueError(
                        f"{type(self).__name__} prefill not support this combination."
                    )
                
                if self.batch_size != 1 :
                    raise ValueError(
                        f"{type(self).__name__} prefill only support batch_size == 1."
                    )
                
                if self.cache_lens[0] != 0:
                    raise ValueError(
                        f"{type(self).__name__} prefill only support cache_lens[0] == 0."
                    )


            elif self.attn_mode == "decode":
                if self.dtype == "bfloat16" \
                    and self.dst_dtype == "bfloat16" \
                    and self.cache_dtype == "bfloat16" \
                    and self.qk_compute_dtype == "bfloat16" \
                    and self.pv_compute_dtype == "bfloat16" \
                    and self.cache_type == "paged":
                    pass
                else:
                    raise ValueError(
                        f"{type(self).__name__} decode not support this combination."
                    )
                
                q_lens_set = set(self.q_lens)
                if len(q_lens_set) != 1:
                    raise ValueError(
                        f"{type(self).__name__} decode only support q_lens == q_lens[0]."
                    )

            else:
                raise ValueError(
                    f"{type(self).__name__} not support this attn_mode: {self.attn_mode}."
                )
        
        def vendor_impl(self):
            super().vendor_impl()
            if self.attn_mode == "prefill":
                self._run_func = self.prefill_run
            elif self.attn_mode == "decode":
                self._run_func = self.decode_run



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
            

        def decode_run(self, tensor_mapping):
            q = tensor_mapping["q"].view(self.batch_size, self.max_q_len, self.q_head_num, self.head_dim)
            k_cache = tensor_mapping["k_cache"].view(self.total_cache_blocks, self.block_size, self.kv_head_num, self.head_dim)
            v_cache = tensor_mapping["v_cache"].view(self.total_cache_blocks, self.block_size, self.kv_head_num, self.head_dim)

            block_table = tensor_mapping["block_table"]
            kv_lens = tensor_mapping["kv_lens"]

            out = flash_attn_with_kvcache(
                q, 
                k_cache, 
                v_cache, 
                cache_seqlens=kv_lens, 
                cache_batch_idx=None,
                block_table=block_table,  
                causal=self.is_causal
            )
            
            return out

except:
    pass



try:
    from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache

    @ProviderRegistry.register_vendor_impl("flash_attention", "fa3")
    class FA3Op(FA2Op):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

except:
    pass