"""
Lightop implementation of MoeSwigluDynamicQuant, calling lightop.moe_swiglu_dynamic_quant
(6 tensors + 1 float; outputs are written into pre-allocated quant_tokens / per_token_scale).
"""
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype, get_torch_dtype_size
from functools import partial
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.moe_swiglu_dynamic_quant import MoeSwigluDynamicQuantOp

try:
    from lightop import op

    @ProviderRegistry.register_vendor_impl("moe_swiglu_dynamic_quant", "lightop")
    class LightopMoeSwigluDynamicQuantOp(MoeSwigluDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["lightop"]
            self.require_profiling = True

        def vendor_impl(self):
            self.torch_dtype = get_torch_dtype(self.dtype)
            self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

            # input/output tensors
            self.input_tensor_info = {
                "scatter_tokens": OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size * 2], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                "experts_smooth_scale": OpTensorInfo(
                    shape=[self.num_experts_per_rank, self.hidden_size], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ), 
                "experts_token_count": OpTensorInfo(
                    shape=[self.num_experts_per_rank], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(
                        self.expert_dispatch_token_count, dtype=dtype, device=device)
                ), 
                "experts_token_offset": OpTensorInfo(
                    shape=[self.num_experts_per_rank], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(
                        self.expert_dispatch_token_offset, dtype=dtype, device=device)
                )
            }
            self.output_tensor_info = {
                "quant_tokens": OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size], 
                    dtype=self.dst_torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                "per_token_scale": OpTensorInfo(
                    shape=[self.dispatch_tokens], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                ),
            }

            # calculator
            self.input_tensor_size = sum([
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            ])
            self.output_tensor_size = sum([
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            ])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors, 
                create_inputs=True, 
                create_outputs=True
            )

            # run func
            self._run_func = self.vendor_impl_run


        def vendor_impl_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            experts_smooth_scale = tensor_mapping["experts_smooth_scale"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_offset = tensor_mapping["experts_token_offset"]
            # Pre-allocated by framework; kernel writes in-place and returns None.
            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            # Signature: (scatter, smooth_scale, token_count, token_offset,
            #             quant_out, scale_out, float_param) -> None
            op.moe_swiglu_dynamic_quant(
                scatter_tokens,
                experts_smooth_scale,
                experts_token_count,
                experts_token_offset,
                quant_tokens,
                per_token_scale,
                1.0,
            )
            return quant_tokens, per_token_scale

except Exception:
    # lightop is optional; if unavailable we just don't register this provider.
    pass
