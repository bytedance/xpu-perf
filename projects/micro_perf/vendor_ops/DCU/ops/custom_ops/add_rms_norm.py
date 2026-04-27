from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.add_rms_norm import AddRmsNormOp
from xpu_perf.micro_perf.core.utils import calc_tensor_size

try:
    from custom_ops import addrmsnorm
    @ProviderRegistry.register_vendor_impl("add_rms_norm", "custom_ops")
    class CustomopsAddRMSNormop(AddRmsNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["custom_ops"]

        def vendor_impl(self):
            # Keep base semantic tensor definitions, only swap run function.
            super().vendor_impl()
            if "output" in self.output_tensor_info:
                self.write_bytes = calc_tensor_size(self.output_tensor_info["output"])
                self.io_bytes = self.read_bytes + self.write_bytes
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            src = tensor_mapping["hidden_states"]
            weight = tensor_mapping["norm_weight"]
            residual = tensor_mapping["residual"]

            dst = addrmsnorm(src, residual, weight, self.eps)

            return dst

except:
    pass
