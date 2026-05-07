import torch

from xpu_perf_provider_base_ops.llm_ops.moe_softmax_topk import MoeSoftmaxTopkOp
from xpu_perf.micro_perf.core.op import ProviderRegistry

from lightop import op

@ProviderRegistry.register_vendor_impl("moe_softmax_topk", "lightop")        
class LightopMoeSoftmaxTopkOp(MoeSoftmaxTopkOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        self.extra_providers = ["lightop"]

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        gating_output = tensor_mapping["gating_output"]

        if self.compute_mode == "pre-softmax":


            M, _ = gating_output.shape

            topk_weights = torch.empty(M,
                                        self.topk,
                                        dtype=torch.float32,
                                        device=gating_output.device)
            topk_ids = torch.empty(M,
                                    self.topk,
                                    dtype=torch.int32,
                                    device=gating_output.device)
            token_expert_indicies = torch.empty(M,
                                                self.topk,
                                                dtype=torch.int32,
                                                device=gating_output.device)


            op.topk_softmax(
            topk_weights, topk_ids,
            token_expert_indicies, gating_output,True)

            del token_expert_indicies  # Not used. Will be used in the future.

            
            return topk_weights, topk_ids
        else:
            raise NotImplementedError
