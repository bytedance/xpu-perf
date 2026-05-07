from functools import partial
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.store_kv_cache import StoreKVCacheOp
from xpu_perf.micro_perf.core.utils import get_torch_dtype, static_quant
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype, get_torch_dtype_size

try:
    from custom_ops import store_kv_cache as _store_kv_cache

    @ProviderRegistry.register_vendor_impl("store_kv_cache", "custom_ops")
    class CustomopStoreKVCacheOp(StoreKVCacheOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["custom_ops"]

        def _run_paged_fallback(self, tensor_mapping):
            """
            First paged version in Python (correctness-first).
            Custom HIP kernel path is still linear-only.
            """
            packed_qkv = tensor_mapping["packed_qkv"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            block_table = tensor_mapping["block_table"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            k_head_start = self.q_head_num
            k_head_end = self.q_head_num + self.kv_head_num
            v_head_start = self.q_head_num + self.kv_head_num
            v_head_end = self.q_head_num + self.kv_head_num * 2

            # Prefer runtime tensor values so generated inputs and attrs stay consistent.
            for batch_idx in range(self.batch_size):
                q_len = int(q_lens[batch_idx].item())
                if q_len <= 0:
                    continue
                q_offset = int(accum_q_lens[batch_idx].item())
                cache_len = int(cache_lens[batch_idx].item())

                src_k = packed_qkv[q_offset:q_offset + q_len, k_head_start:k_head_end, :]
                src_v = packed_qkv[q_offset:q_offset + q_len, v_head_start:v_head_end, :]

                if self.use_quant:
                    src_k = static_quant(src_k, k_scale, self.cache_torch_dtype)
                    src_v = static_quant(src_v, v_scale, self.cache_torch_dtype)

                # [q_len, kv_head_num, head_dim] -> [kv_head_num, q_len, head_dim]
                src_k = src_k.contiguous().transpose(0, 1)
                src_v = src_v.contiguous().transpose(0, 1)

                for t in range(q_len):
                    token_pos = cache_len + t
                    block_idx = token_pos // self.block_size
                    offset_in_block = token_pos % self.block_size
                    physical_block = int(block_table[batch_idx, block_idx].item())
                    if physical_block < 0:
                        continue
                    k_cache[physical_block, :, offset_in_block, :].copy_(src_k[:, t, :])
                    v_cache[physical_block, :, offset_in_block, :].copy_(src_v[:, t, :])

            return k_cache, v_cache

        def vendor_impl(self):
            self.torch_dtype = get_torch_dtype(self.dtype)
            self.cache_torch_dtype = get_torch_dtype(self.cache_dtype)

            self.input_tensor_info = {}
            self.output_tensor_info = {}

            """
            Input QKV in packed / unsplitted layout.
            """
            self.input_tensor_info["packed_qkv"] = OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )


            """
            Build tensors describing how current num_tokens is composed (q/cache/kv lens metadata).
            """
            self.attn_info_tensors = {
                "q_lens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
                ),
                "cache_lens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
                ),
                "kv_lens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.kv_lens, dtype=dtype, device=device)
                ),
                "accum_q_lens": OpTensorInfo(
                    shape=[self.batch_size + 1],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
                ),
                "accum_cache_lens": OpTensorInfo(
                    shape=[self.batch_size + 1],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.accum_cache_lens, dtype=dtype, device=device)
                ),
                "accum_kv_lens": OpTensorInfo(
                    shape=[self.batch_size + 1],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.accum_kv_lens, dtype=dtype, device=device)
                ),
            }
            self.input_tensor_info.update(self.attn_info_tensors)


            """
            KV cache tensors; linear vs paged is determined by block_size / cache_type.
            """
            if self.cache_type == "linear":
                self.input_tensor_info["slot_mapping"] = OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: \
                        torch.tensor(self.slot_mapping, dtype=dtype, device=device)
                )
                cache_shape = [self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim]
            elif self.cache_type == "paged":
                self.input_tensor_info["block_table"] = OpTensorInfo(
                    shape=[self.target_batch_size, self.target_per_seq_num_block],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: \
                        torch.tensor(self.block_table, dtype=dtype, device=device)
                )
                cache_shape = [self.total_cache_blocks, self.kv_head_num, self.block_size, self.head_dim]
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=cache_shape,
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=cache_shape,
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )

            """
            Quantization parameters (only if kv cache is quantized).
            """
            if self.use_quant:
                quant_scale_shape = [self.kv_head_num, self.head_dim]
                self.input_tensor_info["k_scale"] = OpTensorInfo(
                    shape=quant_scale_shape,
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                )
                self.input_tensor_info["v_scale"] = OpTensorInfo(
                    shape=quant_scale_shape,
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                )

            # calculator
            self.input_tensor_size = sum(
                [calc_tensor_size(info) for info in self.input_tensor_info.values()]
            )
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            """
            Bandwidth accounting (aligned with bytemlperf semantics):
            - read_bytes: K/V components read from packed_qkv + indexing/length metadata + quant scales.
              Do NOT count the entire k/v cache as a "full read" (avoids double counting / omissions when
              mixed with tensor_size).
            - write_bytes: bytes actually written to k_cache / v_cache (scaled for linear/paged). Previously
              output_tensor_info was empty, which made write_bytes=0 and systematically under-reported mem_bw.
            """
            pq = self.input_tensor_info["packed_qkv"]
            self.read_bytes = (
                calc_tensor_size(pq) / self.total_head_num * (2 * self.kv_head_num)
                + calc_tensor_size(self.input_tensor_info["q_lens"])
                + calc_tensor_size(self.input_tensor_info["cache_lens"])
                + calc_tensor_size(self.input_tensor_info["accum_q_lens"])
                + calc_tensor_size(self.input_tensor_info["kv_lens"])
                + calc_tensor_size(self.input_tensor_info["accum_cache_lens"])
                + calc_tensor_size(self.input_tensor_info["accum_kv_lens"])
            )
            if self.cache_type == "linear":
                self.read_bytes += calc_tensor_size(self.input_tensor_info["slot_mapping"])
            elif self.cache_type == "paged":
                self.read_bytes += calc_tensor_size(self.input_tensor_info["block_table"])

            if self.use_quant:
                self.read_bytes += (
                    calc_tensor_size(self.input_tensor_info["k_scale"])
                    + calc_tensor_size(self.input_tensor_info["v_scale"])
                )

            sz_k = calc_tensor_size(self.input_tensor_info["k_cache"])
            sz_v = calc_tensor_size(self.input_tensor_info["v_cache"])
            if self.cache_type == "linear":
                self.write_bytes = (
                    sz_k / self.batch_size / self.max_kv_len * self.num_tokens
                    + sz_v / self.batch_size / self.max_kv_len * self.num_tokens
                )
            elif self.cache_type == "paged":
                self.write_bytes = (
                    sz_k / self.num_kv_blocks / self.block_size * self.num_tokens
                    + sz_v / self.num_kv_blocks / self.block_size * self.num_tokens
                    + calc_tensor_size(self.input_tensor_info["block_table"])
                    / self.batch_size
                    / self.max_block_num_per_seq
                    * self.num_q_blocks
                )

            self.io_bytes = self.read_bytes + self.write_bytes


            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=False
            )

            # run func
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            # get pre-allocated input tensors
            packed_qkv = tensor_mapping["packed_qkv"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            use_paged = self.cache_type == "paged"
            block_size = self.block_size if use_paged else 0

            if self.cache_type == "linear":
                slot_mapping = tensor_mapping["slot_mapping"]
                block_table = None
            elif self.cache_type == "paged":
                block_table = tensor_mapping["block_table"]
                slot_mapping = None

            try:
                _store_kv_cache(
                    packed_qkv,
                    k_cache,
                    v_cache,
                    tensor_mapping["q_lens"],
                    tensor_mapping["accum_q_lens"],
                    tensor_mapping["cache_lens"],
                    slot_mapping,
                    block_table,
                    use_paged,
                    block_size,
                    tensor_mapping.get("k_scale"),
                    tensor_mapping.get("v_scale"),
                    use_quant=self.use_quant,
                    q_head_num=self.q_head_num,
                    kv_head_num=self.kv_head_num,
                    head_dim=self.head_dim,
                    total_head_num=self.total_head_num,
                    max_kv_len=self.max_kv_len,
                )
            except TypeError:
                # Backward compatibility with older extension signature.
                if use_paged:
                    return self._run_paged_fallback(tensor_mapping)
                _store_kv_cache(
                    packed_qkv,
                    k_cache,
                    v_cache,
                    tensor_mapping["q_lens"],
                    tensor_mapping["accum_q_lens"],
                    tensor_mapping["cache_lens"],
                    slot_mapping,
                    tensor_mapping.get("k_scale"),
                    tensor_mapping.get("v_scale"),
                    use_quant=self.use_quant,
                    q_head_num=self.q_head_num,
                    kv_head_num=self.kv_head_num,
                    head_dim=self.head_dim,
                    total_head_num=self.total_head_num,
                    max_kv_len=self.max_kv_len,
                )
            return k_cache, v_cache
except:
    pass
