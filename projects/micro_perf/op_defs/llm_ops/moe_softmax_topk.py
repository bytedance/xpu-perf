"""LLM op: moe_softmax_topk (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("moe_softmax_topk", "ComputeEngine")
class MoeSoftmaxTopkOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(
                f"MoeSoftmaxTopkOp only support llm arg_type, but got {self.arg_type}"
            )

        # pre-defined attrs
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]
        self.compute_mode = self.args_dict["compute_mode"]
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size


        # 以下参数决定当前 moe_softmax_topk 的具体数据类型
        self.dtype = self.args_dict["dtype"]


    def vendor_parser(self):
        if self.dtype in ["float32"]:
            pass
        else:
            raise ValueError(
                f"MoeSoftmaxTopkOp base impl only support float32 dtype, but got {self.dtype}"
            )

        if self.compute_mode not in ["pre-softmax", "post-softmax"]:
            raise ValueError(
                f"MoeSoftmaxTopkOp base impl only support pre-softmax and post-softmax compute mode, but got {self.compute_mode}"
            )

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)

        # input/output tensors
        self.input_tensor_info = {
            "gating_output": OpTensorInfo(
                shape=[self.num_tokens, self.num_experts], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "selected_experts": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
            ), 
            "moe_weights": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
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
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        gating_output = tensor_mapping["gating_output"]

        # softmax --> topk --> normlize
        if self.compute_mode == "pre-softmax":
            softmax_output = torch.softmax(gating_output, dim=-1)
            moe_weights, selected_experts = torch.topk(softmax_output, self.topk, dim=-1)
            moe_weights = moe_weights / moe_weights.sum(dim=-1, keepdim=True)
            return selected_experts, moe_weights
        # topk --> softmax
        elif self.compute_mode == "post-softmax":
            topk_output, selected_experts = torch.topk(gating_output, self.topk, dim=-1)
            softmax_output = torch.softmax(topk_output, dim=-1)
            return selected_experts, softmax_output




