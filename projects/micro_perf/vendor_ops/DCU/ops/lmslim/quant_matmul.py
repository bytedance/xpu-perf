import torch
from typing import Optional, List

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.quant_matmul import QuantMatmulOp

try:
    from lmslim.quantize.quant_ops import triton_scaled_mm
    from lmslim import quant_ops

    @ProviderRegistry.register_vendor_impl("quant_matmul", "lmslim")
    class LmslimQuantMatmulOp(QuantMatmulOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["lmslim"]
            
        def blaslt_scaled_mm(
            self, 
            a_int8: torch.Tensor,           # int8 [M, K]
            b_int8: torch.Tensor,           # int8 [K, N]
            scale_a: torch.Tensor,          # fp32 scale for A
            scale_b: torch.Tensor,          # fp32 scale for B
            out_dtype: torch.dtype,
            bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            m = a_int8.shape[0]
            k = a_int8.shape[1]
            n = b_int8.shape[0]
            _, out = quant_ops.hipblaslt_w8a8_gemm(a_int8, b_int8, scale_a, scale_b, m, n, k, "NT", out_dtype,bias)
            return out

        def vendor_impl_run(self, tensor_mapping):
            # get pre-allocated input tensors, require hidden_states contiguous, expert_weight not
            hidden_states = tensor_mapping["hidden_states"]
            per_token_scale = tensor_mapping["per_token_scale"]
            expert_weight = tensor_mapping["expert_weight"]#.transpose(0,1)
            expert_scale = tensor_mapping["expert_scale"]

            out = self.blaslt_scaled_mm(hidden_states, expert_weight, per_token_scale, expert_scale, out_dtype=self.dst_torch_dtype, bias=None)
            
            return out
except Exception as e:
    import traceback
    traceback.print_exc()
