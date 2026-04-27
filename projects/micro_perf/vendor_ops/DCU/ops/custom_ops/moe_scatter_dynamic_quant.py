from functools import partial
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.moe_scatter_dynamic_quant import MoeScatterDynamicQuantOp
from xpu_perf.micro_perf.core.utils import static_quant

try:
    from custom_ops import moe_scatter_dynamic_quant

    @ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "custom_ops")
    class CustomOpsMoeScatterDynamicQuantOp(MoeScatterDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["custom_ops"]

        def vendor_impl(self):
            # Keep base semantic tensor definitions, only swap run function.
            super().vendor_impl()
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            # get pre-allocated input tensors
            hidden_states = tensor_mapping["hidden_states"]
            experts_smooth_scale = tensor_mapping["experts_smooth_scale"]
            selected_experts = tensor_mapping["selected_experts"]
            moe_weights = tensor_mapping["moe_weights"]

            # get pre-allocated output tensors
            scatter_tokens = tensor_mapping["scatter_tokens"]
            scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]

            # For ease of reference in code demonstration,
            # all the following tensors are precomputed.
            # Vendors are required to implement the corresponding computation logic during integration.
            scatter_token_id = tensor_mapping["scatter_token_id"]
            scatter_token_weight = tensor_mapping["scatter_token_weight"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_offset = tensor_mapping["experts_token_offset"]

            #import traceback
            #traceback.print_stack()

            if experts_smooth_scale.shape[0] == self.num_experts:
                experts_smooth_scale_per_rank = experts_smooth_scale[self.experts_start_idx:self.experts_end_idx]
            else:
                experts_smooth_scale_per_rank = experts_smooth_scale

            result = moe_scatter_dynamic_quant(
                hidden_states=hidden_states,
                experts_smooth_scale=experts_smooth_scale_per_rank,
                selected_experts=selected_experts,
                moe_weights=moe_weights,
                scatter_tokens=scatter_tokens,
                scatter_per_token_scale=scatter_per_token_scale,
                scatter_token_id=scatter_token_id,
                scatter_token_weight=scatter_token_weight,
                experts_token_count=experts_token_count,
                experts_token_offset=experts_token_offset,
                topk=self.topk,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                dst_dtype=self.dst_torch_dtype,
                balanced=True,
            )

            if isinstance(result, tuple) and len(result) == 6:
                scatter_tokens, scatter_per_token_scale, \
                scatter_token_id, scatter_token_weight, \
                experts_token_count, experts_token_offset = result

            return scatter_tokens, scatter_per_token_scale, \
                scatter_token_id, scatter_token_weight, \
                experts_token_count, experts_token_offset

except:
    pass
