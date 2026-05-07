from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.add_rms_norm_dynamic_quant import AddRmsNormDynamicQuantOp

try:
    from custom_ops import addrmsnormdynamicquant
    @ProviderRegistry.register_vendor_impl("add_rms_norm_dynamic_quant", "custom_ops")
    class CustomopsAddRMSNormDynamicQuantOp(AddRmsNormDynamicQuantOp):
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

            src = tensor_mapping["hidden_states"]
            weight = tensor_mapping["norm_weight"]
            smoothScale = tensor_mapping["smooth_scale"]
            residual = tensor_mapping["residual"]
            per_token_scale = tensor_mapping["per_token_scale"]
            dst = tensor_mapping["quant_tokens"]
            
            if self.output_mode == "none":
                addrmsnormdynamicquant(src,weight,smoothScale,residual,dst,per_token_scale,None,None,0,self.eps)
                return dst, per_token_scale
            elif self.output_mode == "res":
                after_res = tensor_mapping["after_res"]
                addrmsnormdynamicquant(src,weight,smoothScale,residual,dst,per_token_scale,after_res,None,1,self.eps)
                return dst, per_token_scale, after_res
            elif self.output_mode == "norm":
                after_norm = tensor_mapping["after_norm"]
                addrmsnormdynamicquant(src,weight,smoothScale,residual,dst,per_token_scale,None,after_norm,2,self.eps)
                return dst, per_token_scale, after_norm

except:
    pass
