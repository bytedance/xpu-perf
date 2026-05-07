from functools import partial
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.flash_attention import FlashAttentionOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size



try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    @ProviderRegistry.register_vendor_impl("flash_attention", "fa2")
    class FA2Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            _allowed_dtypes = ("bfloat16", "float16")
            _allowed_block_sizes = (0, 64)
            block_size = self.args_dict.get("block_size", 0)
            if block_size not in _allowed_block_sizes:
                raise ValueError(
                    f"{type(self).__name__} only supports block_size in {_allowed_block_sizes}, got {block_size}."
                )

            if self.attn_mode == "prefill":
                if all(d in _allowed_dtypes for d in (
                    self.dtype, self.dst_dtype, self.cache_dtype,
                    self.qk_compute_dtype, self.pv_compute_dtype
                )):
                    pass
                else:
                    raise ValueError(
                        f"{type(self).__name__} prefill not support this combination."
                    )

            elif self.attn_mode == "decode":
                if all(d in _allowed_dtypes for d in (
                    self.dtype, self.dst_dtype, self.cache_dtype,
                    self.qk_compute_dtype, self.pv_compute_dtype
                )):
                    pass
                else:
                    raise ValueError(
                        f"{type(self).__name__} decode not support this combination."
                    )
                if self.cache_type == "linear":
                    kv_lens_set = set(self.kv_lens)
                    if len(kv_lens_set) != 1:
                        raise ValueError(
                            f"{type(self).__name__} decode linear cache requires all kv_lens equal, got {self.kv_lens}."
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
            self._run_func = self.vendor_impl_run


        def vendor_impl_run(self, tensor_mapping):
            if self.attn_mode == "prefill":
                return self.prefill_run(tensor_mapping)
            if self.attn_mode == "decode":
                return self.decode_run(tensor_mapping)
            raise ValueError(
                f"{type(self).__name__} not support this attn_mode: {self.attn_mode}."
            )



        def prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"].view(self.batch_size, self.num_tokens, self.q_head_num, self.head_dim)
            kv_len = self.kv_lens[0] if self.batch_size == 1 else max(self.kv_lens)
            k_raw = tensor_mapping["k_cache"]
            v_raw = tensor_mapping["v_cache"]

            if self.cache_type == "linear":
                k_cache = k_raw[:, :, :kv_len, :].permute(0, 2, 1, 3)
                v_cache = v_raw[:, :, :kv_len, :].permute(0, 2, 1, 3)
            else:
                # paged: k_raw [total_blocks, kv_head_num, block_size, head_dim], reassemble by block_table into
                # [1, kv_len, kv_head_num, head_dim]
                block_table = tensor_mapping["block_table"]
                num_blocks = (kv_len + self.block_size - 1) // self.block_size
                k_parts, v_parts = [], []
                for b in range(num_blocks):
                    phys = block_table[0, b].item()
                    start = b * self.block_size
                    length = min(self.block_size, kv_len - start)
                    k_parts.append(k_raw[phys, :, :length, :])
                    v_parts.append(v_raw[phys, :, :length, :])
                k_cat = torch.cat(k_parts, dim=1)
                v_cat = torch.cat(v_parts, dim=1)
                k_cache = k_cat.permute(1, 0, 2).unsqueeze(0)
                v_cache = v_cat.permute(1, 0, 2).unsqueeze(0)

            out = flash_attn_func(
                q, k_cache, v_cache,
                causal=self.is_causal
            )
            return out
            

        def decode_run(self, tensor_mapping):
            q = tensor_mapping["q"].view(self.batch_size, self.max_q_len, self.q_head_num, self.head_dim)
            k_raw = tensor_mapping["k_cache"]
            v_raw = tensor_mapping["v_cache"]
            kv_lens = tensor_mapping["kv_lens"]

            if self.cache_type == "linear":
                # linear: k_cache [batch_size, kv_head_num, max_kv_len, head_dim] -> flash_attn_func expects [B, kv_len, H, D]
                kv_len = int(kv_lens[0].item())
                k_cache = k_raw[:, :, :kv_len, :].permute(0, 2, 1, 3)
                v_cache = v_raw[:, :, :kv_len, :].permute(0, 2, 1, 3)
                out = flash_attn_func(q, k_cache, v_cache, causal=self.is_causal)
            else:
                # paged: k_cache [total_blocks, kv_head_num, block_size, head_dim] -> [total_blocks, block_size, kv_head_num, head_dim]
                k_cache = k_raw.permute(0, 2, 1, 3)
                v_cache = v_raw.permute(0, 2, 1, 3)
                block_table = tensor_mapping["block_table"]
                out = flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    cache_seqlens=kv_lens, cache_batch_idx=None,
                    block_table=block_table, causal=self.is_causal
                )
            return out

except:
    pass

# Only enable FA2; keep FA3 disabled (FA2 is sufficient for current use).
# try:
#     from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
#     @ProviderRegistry.register_vendor_impl("flash_attention", "fa3")
#     class FA3Op(FA2Op):
#         ...
# except:
#     pass
