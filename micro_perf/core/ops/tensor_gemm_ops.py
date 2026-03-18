import sys
import pathlib
from functools import partial

import torch

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[2])
)

from core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype
from core.op import BasicOp, register_base_impl


@register_base_impl
class GemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise ValueError(f"arg_type {self.arg_type} is not supported")
    
        # pre-defined attrs
        if self.arg_type == "default":
            self.M = self.args_dict["M"]
            self.K = self.args_dict["K"]
            self.N = self.args_dict["N"]
        elif self.arg_type == "llm":
            self.sp_size = self.args_dict.get("sp_size", 1)
            self.M = self.args_dict["num_tokens"] // self.sp_size
            self.K = self.args_dict["hidden_size"]
            self.N = self.args_dict["new_hidden_size"]
        else:
            raise ValueError(f"arg_type {self.arg_type} is not supported")

        # 以下参数决定当前 gemm 的具体数据类型
        self.dtype = self.args_dict["dtype"]

        self.vendor_parser()
        self.vendor_impl()
    

    def vendor_parser(self):
        if self.dtype in ["float32", "tfloat32", "float16", "bfloat16"]:
            pass
        else:
            raise ValueError(f"dtype {self.dtype} is not supported")

    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.dst_dtype = self.args_dict.get("dst_dtype", self.dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)
    
        self.input_tensor_info = {
            "a": OpTensorInfo(
                shape=[self.M, self.K],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "b": OpTensorInfo(
                shape=[self.K, self.N],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }
        self.output_tensor_info = {
            "c": OpTensorInfo(
                shape=[self.M, self.N],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

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

        self.calc_flops = self.M * self.N * self.K * 2

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )

        self._run_func = self.vendor_impl_run


    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = torch.matmul(a, b)
        return c
    