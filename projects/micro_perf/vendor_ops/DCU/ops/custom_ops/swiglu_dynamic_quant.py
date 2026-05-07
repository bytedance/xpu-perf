from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.swiglu_dynamic_quant import SwigluDynamicQuantOp

try:
    from custom_ops import swiglu_dynamic_quant

    @ProviderRegistry.register_vendor_impl("swiglu_dynamic_quant", "custom_ops")
    class CustomOpsSwigluDynamicQuantOp(SwigluDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["custom_ops"]

        def vendor_impl(self):
            # custom_ops kernel writes into preallocated output tensors, so we must
            # create outputs in the tensor mapping (base impl uses create_outputs=False).
            super().vendor_impl()
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            # get pre-allocated input tensors
            hidden_states = tensor_mapping["hidden_states"]
            smooth_scale = tensor_mapping["smooth_scale"]
            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            quant_tokens, per_token_scale = swiglu_dynamic_quant(
                hidden_states,
                smooth_scale,
                quant_tokens,
                per_token_scale,
                self.num_tokens,
                self.hidden_size)
            return quant_tokens, per_token_scale
     
except:
    pass
