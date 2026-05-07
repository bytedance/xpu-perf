from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.qk_rms_norm import QKRMSNormOp


from lightop import op
@ProviderRegistry.register_vendor_impl("qk_rms_norm", "lightop")

class LightopQKRMSNormOp(QKRMSNormOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        self.extra_providers = ["lightop"]

    def vendor_impl_run(self, tensor_mapping):
        token_data = tensor_mapping["token_data"]
        q_norm_weight = tensor_mapping["q_norm_weight"]
        k_norm_weight = tensor_mapping["k_norm_weight"]

        op.fuse_qkv_head_rms_norm(token_data, q_norm_weight, k_norm_weight, self.q_head_num, self.kv_head_num, self.qk_head_dim, self.eps)

        return token_data



