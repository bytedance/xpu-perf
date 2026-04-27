from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.add_rms_norm_dynamic_quant import AddRmsNormDynamicQuantOp

try:
    from lightop import op
    @ProviderRegistry.register_vendor_impl("add_rms_norm_dynamic_quant", "lightop")
    class LightopAddRMSNormDynamicQuantOp(AddRmsNormDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["lightop"]

        def vendor_impl(self):
            # lightop kernel writes into preallocated output tensors, so we must
            # create outputs in the tensor mapping (base impl uses create_outputs=False).
            super().vendor_impl()
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):

            src = tensor_mapping["hidden_states"]
            weight = tensor_mapping["norm_weight"]
            smoothScale = tensor_mapping["smooth_scale"]
            residual = tensor_mapping["residual"]
            per_token_scale = tensor_mapping["per_token_scale"]
            dst = tensor_mapping["quant_tokens"]
            
            # after_res, dst, per_token_scale = op.miopen_add_rms_norm_dynamic_quant(src, residual, weight, smoothScale, self.eps)
            if self.output_mode == "none":
                op.rms_norm_smooth_per_token_dynamic_quant(dst, src, weight, smoothScale, per_token_scale,self.eps, residual, None,None,False,False)
                return dst, per_token_scale

            elif self.output_mode == "res":
                after_res = tensor_mapping["after_res"]
                op.rms_norm_smooth_per_token_dynamic_quant(dst, src, weight, smoothScale, per_token_scale,self.eps, residual, None,after_res,False,False)
                return dst, per_token_scale, after_res
            elif self.output_mode == "norm":
                after_norm = tensor_mapping["after_norm"]
                op.rms_norm_smooth_per_token_dynamic_quant(dst, src, weight, smoothScale, per_token_scale,self.eps, residual, after_norm,None,False,False)
                return dst, per_token_scale, after_norm

except:
    pass
