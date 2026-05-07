"""
Torch implementation of MoeGatingGemm: during create_tensors, convert gating_weight to a
[hidden_size, num_experts] contiguous layout to avoid repeated transpose on the matmul hot path.
"""
import torch
from functools import partial
from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.moe_gating_gemm import MoeGatingGemmOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size


@ProviderRegistry.register_vendor_impl("moe_gating_gemm", "torch")
class TorchMoeGatingGemmOp(MoeGatingGemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["torch"]
        self.require_profiling = True

    def vendor_parser(self):
        if self.dtype in ("float32", "bfloat16") and self.compute_dtype in ("float32", "bfloat16") and self.dst_dtype == "float32":
            pass
        else:
            raise ValueError(f"MoeGatingGemmOp only support float32-->float32, but got {self.dtype}--> {self.dst_dtype}")

    def vendor_impl(self):
        # Reuse base setup and then switch gating_weight layout to [hidden_size, num_experts].
        # This avoids per-run transpose in the hot mm path.
        super().vendor_impl()

        self.input_tensor_info["gating_weight"] = OpTensorInfo(
            shape=[self.hidden_size, self.num_experts],
            dtype=self.torch_dtype,
            device=self.backend.get_torch_device_name(),
        )

        # Recompute io stats to match updated tensor shape bookkeeping.
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )

    def vendor_impl_run(self, tensor_mapping):
        gating_output = torch.mm(
            tensor_mapping["hidden_states"], 
            tensor_mapping["gating_weight"]
        ).to(self.dst_torch_dtype)
        return gating_output
