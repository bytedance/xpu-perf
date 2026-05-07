from functools import partial
import os

import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.moe_quant_group_gemm import MoeQuantGroupGemmOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype

from lightop import moe_groupgemm_marlin_w4a8, get_moe_groupgemm_config_w4a8


@ProviderRegistry.register_vendor_impl("moe_quant_group_gemm", "lightop")
class LightopMoeQuantGroupGemmOp(MoeQuantGroupGemmOp):
    def vendor_parser(self):
        """
        Bring legacy MoeQuantGroupGemmUp semantics into vendor implementation.
        K1/K2 and tile_k/tile_n are kernel/padding/packing-related parameters and
        therefore belong to the vendor layer (not the base op).
        """
        if self.dtype != "int8" or self.compute_dtype != "int8" or self.dst_dtype != "bfloat16":
            raise ValueError(
                f"{type(self).__name__} only supports int8/int8 -> bfloat16, "
                f"got dtype={self.dtype}, compute_dtype={self.compute_dtype}, dst_dtype={self.dst_dtype}"
            )

        # Match legacy xpu-perf MoeQuantGroupGemmUpOp behavior:
        # lightop w4a8 marlin kernels expect packed weights stored as int32.
        if self.w_dtype != "int32":
            raise ValueError(
                f"{type(self).__name__} only supports w_dtype=int32 (packed w4a8 weights), got {self.w_dtype}"
            )

        self.K1 = int(self.args_dict.get("K1", self.hidden_size // 2))
        self.K2 = int(self.args_dict.get("K2", self.new_hidden_size // 4))
        # For marlin w4a8 groupgemm kernels, the variant name often encodes tiling
        self.tile_k = int(self.args_dict.get("tile_k", 32))
        self.tile_n = int(self.args_dict.get("tile_n", 64))
        # Many lightop MoE groupgemm kernels assume each expert's token segment is aligned
        # (e.g., for vectorized memory access). Keep this vendor-specific.
        # Default to legacy xpu-perf behavior (no padding). You can opt-in per-case.
        self.token_align = int(self.args_dict.get("token_align", 1))
        if self.token_align <= 0:
            raise ValueError(f"{type(self).__name__} requires token_align > 0, got {self.token_align}")

        if self.hidden_size % self.tile_k != 0:
            raise ValueError(
                f"{type(self).__name__} requires hidden_size % tile_k == 0, got hidden_size={self.hidden_size}, tile_k={self.tile_k}"
            )
        if self.tile_n > 0 and (self.new_hidden_size % self.tile_n != 0):
            raise ValueError(
                f"{type(self).__name__} requires new_hidden_size % tile_n == 0 when tile_n>0, "
                f"got new_hidden_size={self.new_hidden_size}, tile_n={self.tile_n}"
            )

    def vendor_impl(self):
        self.extra_providers = ["lightop"]
        # Must be available during vendor_impl (called inside BasicOp.__init__).
        self.num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

        self.torch_dtype = get_torch_dtype(self.dtype)
        self.w_torch_dtype = get_torch_dtype(self.w_dtype)
        self.compute_torch_dtype = get_torch_dtype(self.compute_dtype)
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # packed tensor layout (copied from legacy MoeQuantGroupGemmUpOp)
        self.input_tensor_info = {}
        self.output_tensor_info = {}

        # Match legacy xpu-perf MoeQuantGroupGemmUpOp tensor semantics:
        # - no padding / alignment in generated token counts/offsets
        # - use dispatch_tokens as the leading dimension for input/output
        self.expert_dispatch_token_count = list(self.expert_dispatch_token_count)
        # lightop moe_groupgemm_marlin_w4a8 expects experts_offsets as a prefix-sum array.
        # Build a strict (E+1) offsets array from counts to avoid ambiguity.
        # offsets[i]..offsets[i+1] is expert i's segment.
        _offsets = [0]
        for c in self.expert_dispatch_token_count:
            _offsets.append(_offsets[-1] + int(c))
        self.expert_dispatch_token_offset = _offsets

        self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
            shape=[self.dispatch_tokens, self.hidden_size],
            dtype=self.torch_dtype,
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros,
        )
        self.input_tensor_info["per_token_scale"] = OpTensorInfo(
            shape=[self.dispatch_tokens, 1],
            dtype=torch.float32,
            device=self.backend.get_torch_device_name(),
            creator=torch.ones,
        )

        self.input_tensor_info["experts_weight"] = OpTensorInfo(
            shape=[
                self.num_experts_per_rank,
                self.hidden_size // self.tile_k,
                self.new_hidden_size * self.tile_k // 8,
            ],
            dtype=self.w_torch_dtype,
            device=self.backend.get_torch_device_name(),
            creator=torch.zeros,
        )
        self.input_tensor_info["experts_scale"] = OpTensorInfo(
            shape=[self.num_experts_per_rank, self.new_hidden_size, 1],
            dtype=torch.float32,
            device=self.backend.get_torch_device_name(),
            creator=torch.ones,
        )
        self.input_tensor_info["experts_token_count"] = OpTensorInfo(
            shape=[self.num_experts_per_rank],
            dtype=torch.int32,
            device=self.backend.get_torch_device_name(),
            creator=lambda size, dtype, device: torch.tensor(
                self.expert_dispatch_token_count, dtype=dtype, device=device
            ),
        )
        self.input_tensor_info["experts_token_offset"] = OpTensorInfo(
            shape=[self.num_experts_per_rank + 1],
            dtype=torch.int32,
            device=self.backend.get_torch_device_name(),
            creator=lambda size, dtype, device: torch.tensor(
                self.expert_dispatch_token_offset, dtype=dtype, device=device
            ),
        )

        self.output_tensor_info["y"] = OpTensorInfo(
            shape=[self.dispatch_tokens, self.new_hidden_size],
            dtype=self.dst_torch_dtype,
            device=self.backend.get_torch_device_name(),
        )

        self.input_tensor_size = sum(calc_tensor_size(info) for info in self.input_tensor_info.values())
        self.output_tensor_size = sum(calc_tensor_size(info) for info in self.output_tensor_info.values())
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self.calc_flops = 2 * self.dispatch_tokens * self.hidden_size * self.new_hidden_size

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )
        self._run_func = self.vendor_impl_run

        # kernel configs used by lightop implementation
        self.config1, self.config2, self.status = get_moe_groupgemm_config_w4a8(
            self.num_experts_per_rank,
            self.num_tokens,
            self.new_hidden_size,
            self.K1,
            self.hidden_size,
            self.K2,
            "DCU",
            str(self.num_cus),
            self.dst_dtype,
            self.tile_n if self.tile_n > 0 else None,
            self.tile_k if self.tile_k > 0 else None,
        )
        

    def vendor_impl_run(self, tensor_mapping):

        scatter_tokens = tensor_mapping["scatter_tokens"]
        experts_weight = tensor_mapping["experts_weight"]
        per_token_scale = tensor_mapping["per_token_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_scale = tensor_mapping["experts_scale"]
        experts_token_start = tensor_mapping["experts_token_offset"]
        gemm1_out = tensor_mapping["y"]

        moe_groupgemm_marlin_w4a8(scatter_tokens,
            experts_weight,
            gemm1_out,
            per_token_scale,
            experts_scale,
            experts_token_count,
            experts_token_start,
            self.config1
        )


        return gemm1_out
