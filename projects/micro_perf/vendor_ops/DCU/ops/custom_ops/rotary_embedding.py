from itertools import chain
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.rotary_embedding import RotaryEmbeddingOp

try:
    # from vllm import _custom_ops as ops
    from custom_ops import rotary_embedding

    @ProviderRegistry.register_vendor_impl("rotary_embedding", "custom_ops")
    class LightopRotaryEmbeddingOp(RotaryEmbeddingOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["custom_ops"]
            self.require_profiling = True

        def vendor_impl_run(self, tensor_mapping):
            packed_qkv = tensor_mapping["packed_qkv"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cos = tensor_mapping["cos"]
            sin = tensor_mapping["sin"]

            cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()

            dim_start = self.rope_offset
            dim_end = self.rope_offset + self.rope_dim

            positions = getattr(self, "positions", None)
            if positions is None or positions.numel() != self.num_tokens:
                # Use self.cache_lens / self.q_lens to avoid indexing GPU cache_lens (can trigger sync).
                positions_list = [
                    self.cache_lens[b] + j
                    for b in range(self.batch_size)
                    for j in range(self.q_lens[b])
                ]
                positions = torch.tensor(positions_list, dtype=torch.int64, device=packed_qkv.device)

            q,k = rotary_embedding(
                positions=positions,
                query=packed_qkv[:, :self.q_head_num, dim_start:dim_end].view(packed_qkv.size(0), -1),
                key=packed_qkv[:, self.q_head_num:self.q_head_num + self.kv_head_num, dim_start:dim_end].view(packed_qkv.size(0), -1),
                cos_sin_cache=cos_sin_cache,
                head_size=self.rope_dim,
                q_head_num=self.q_head_num,
                kv_head_num=self.kv_head_num,
                is_neox=1,
            )
            return packed_qkv
except Exception:
    pass
