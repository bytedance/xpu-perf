"""LLM op: moe_quant_group_gemm (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("moe_quant_group_gemm", "ComputeEngine")
class MoeQuantGroupGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(
                f"MoeQuantGroupGemmOp only support llm arg_type, but got {self.arg_type}"
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

        # get moe token disptch info
        self.num_scatter_tokens, \
        self.num_scatter_tokens_per_rank, \
        self.num_experts_per_rank, \
        self.experts_start_idx, \
        self.experts_end_idx, \
        self.all_select_experts, \
        self.all_select_weights, \
        self.dispatch_tokens, \
        self.used_src_tokens, \
        self.expert_dispatch_tokens, \
        self.expert_dispatch_weights, \
        self.scatter_token_id, \
        self.scatter_token_weight, \
        self.expert_dispatch_token_count, \
        self.expert_dispatch_token_offset = get_moe_tokens_info(
            self.num_tokens, self.num_experts, self.topk, 
            ep_size=self.ep_size, ep_rank=self.ep_rank
        )

        # 以下参数决定当前的 moe_quant_group_gemm 的具体数据类型
        self.dtype = self.args_dict.get("dtype", "int8")
        self.w_dtype = self.args_dict.get("w_dtype", "int8")
        self.compute_dtype = self.args_dict.get("compute_dtype", "int8")
        self.dst_dtype = self.args_dict.get("dst_dtype", "bfloat16")

    def vendor_parser(self):
        if self.dtype == "int8" \
            and self.w_dtype == "int8" \
            and self.compute_dtype == "int8" \
            and self.dst_dtype == "bfloat16":
            pass
        else:
            raise ValueError(
                f"MoeQuantGroupGemmOp base impl not support: "
                f"dtype={self.dtype}, w_dtype={self.w_dtype}, "
                f"compute_dtype={self.compute_dtype}, dst_dtype={self.dst_dtype}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.w_torch_dtype = get_torch_dtype(self.w_dtype)
        self.compute_torch_dtype = get_torch_dtype(self.compute_dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.input_tensor_info = {}
        self.output_tensor_info = {}

        # hidden_states
        self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
            shape=[self.dispatch_tokens, self.hidden_size], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros
        )
        self.input_tensor_info["per_token_scale"] = OpTensorInfo(
            shape=[self.dispatch_tokens], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )
        
        # experts_weight
        self.input_tensor_info["experts_weight"] = OpTensorInfo(
            shape=[self.num_experts_per_rank, self.new_hidden_size, self.hidden_size], 
            dtype=self.w_torch_dtype, 
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros
        )
        self.input_tensor_info["experts_scale"] = OpTensorInfo(
            shape=[self.num_experts_per_rank, self.new_hidden_size], 
            dtype=torch.float32, 
            device=self.backend.get_torch_device_name(),
            creator=torch.ones
        )
        
        # expert distribution info
        self.input_tensor_info["experts_token_count"] = OpTensorInfo(
            shape=[self.num_experts_per_rank], 
            dtype=torch.int32, 
            device=self.backend.get_torch_device_name(),
            creator=lambda size, dtype, device: torch.tensor(
                self.expert_dispatch_token_count, dtype=dtype, device=device)
        )
        self.input_tensor_info["experts_token_offset"] = OpTensorInfo(
            shape=[self.num_experts_per_rank], 
            dtype=torch.int32, 
            device=self.backend.get_torch_device_name(),
            creator=lambda size, dtype, device: torch.tensor(
                self.expert_dispatch_token_offset, dtype=dtype, device=device)
        )

        self.output_tensor_info["y"] = OpTensorInfo(
            shape=[self.dispatch_tokens, self.new_hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=self.backend.get_torch_device_name()
        )

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        # Subtract weight/scale bytes for inactive experts (no tokens dispatched)
        inactive = self.num_experts_per_rank - sum(1 for c in self.expert_dispatch_token_count if c > 0)
        inactive_bytes = inactive * self.new_hidden_size * (
            self.hidden_size * self.input_tensor_info["experts_weight"].dtype.itemsize
            + self.input_tensor_info["experts_scale"].dtype.itemsize)
        self.read_bytes = self.input_tensor_size - inactive_bytes
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 2 * self.dispatch_tokens * self.hidden_size * self.new_hidden_size


        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]
        experts_weight = tensor_mapping["experts_weight"]
        experts_scale = tensor_mapping["experts_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]

        # get pre-allocated output tensor
        y = tensor_mapping["y"]


        # use loop gemm and fp32 to simulate int8 group_gemm
        for i in range(self.num_experts_per_rank):
            cur_token_start = experts_token_offset[i]
            cur_token_end = cur_token_start + experts_token_count[i]

            cur_tokens = scatter_tokens[cur_token_start:cur_token_end]
            cur_tokens_scale = per_token_scale[cur_token_start:cur_token_end]

            cur_weight = experts_weight[i]
            cur_weight_scale = experts_scale[i]

            y[cur_token_start:cur_token_end] = fake_quant_gemm(
                cur_tokens, cur_tokens_scale, 
                cur_weight, cur_weight_scale, 
                dst_torch_dtype=self.dst_torch_dtype, 
                trans_w=True
            )
        return y



