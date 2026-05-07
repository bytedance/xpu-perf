"""LLM op: moe_quant_group_gemm_up (base definition).

This exists for compatibility with legacy split-op vendor implementations.
The semantic definition is intentionally identical to `moe_quant_group_gemm`;
vendor providers may use different kernel/packing for up/down.
"""

from ._common import *


@ProviderRegistry.register_base_impl("moe_quant_group_gemm_up", "ComputeEngine")
class MoeQuantGroupGemmUpOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm"]:
            raise ValueError(
                f"{type(self).__name__} only supports llm arg_type, but got {self.arg_type}"
            )

        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]

        # moe info
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        # parallel info
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.ep_rank = self.args_dict.get("ep_rank", 0)

        # get moe token dispatch info
        (
            self.num_scatter_tokens,
            self.num_scatter_tokens_per_rank,
            self.num_experts_per_rank,
            self.experts_start_idx,
            self.experts_end_idx,
            self.all_select_experts,
            self.all_select_weights,
            self.dispatch_tokens,
            self.used_src_tokens,
            self.expert_dispatch_tokens,
            self.expert_dispatch_weights,
            self.scatter_token_id,
            self.scatter_token_weight,
            self.expert_dispatch_token_count,
            self.expert_dispatch_token_offset,
        ) = get_moe_tokens_info(
            self.num_tokens,
            self.num_experts,
            self.topk,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
        )

        # dtype tuple
        self.dtype = self.args_dict.get("dtype", "int8")
        self.w_dtype = self.args_dict.get("w_dtype", "int8")
        self.compute_dtype = self.args_dict.get("compute_dtype", "int8")
        self.dst_dtype = self.args_dict.get("dst_dtype", "bfloat16")


