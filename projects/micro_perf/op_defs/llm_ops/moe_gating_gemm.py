"""LLM op: moe_gating_gemm (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("moe_gating_gemm", "ComputeEngine")
class MoeGatingGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(f"MoeGatingGemmOp only support llm arg_type, but got {self.arg_type}")

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.num_experts = self.args_dict["num_experts"]

        # 以下参数决定当前 moe_gating_gemm 的具体数据类型
        self.dtype = self.args_dict.get("dtype", "float32")
        self.compute_dtype = self.args_dict.get("compute_dtype", "float32")
        self.dst_dtype = self.args_dict.get("dst_dtype", self.dtype)

    def vendor_parser(self):
        if self.dtype == "float32" and self.compute_dtype == "float32" and self.dst_dtype == "float32":
            pass
        else:
            raise ValueError(f"MoeGatingGemmOp only support float32-->float32, but got {self.dtype}--> {self.dst_dtype}")
    
    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.compute_torch_dtype = get_torch_dtype(self.compute_dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)
        
        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "gating_weight": OpTensorInfo(
                shape=[self.num_experts, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "gating_output": OpTensorInfo(
                shape=[self.num_tokens, self.num_experts], 
                dtype=self.dst_torch_dtype, 
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

        self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.num_experts

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        gating_output = torch.mm(
            tensor_mapping["hidden_states"], 
            tensor_mapping["gating_weight"].t()
        ).type(self.dst_torch_dtype)
        return gating_output



