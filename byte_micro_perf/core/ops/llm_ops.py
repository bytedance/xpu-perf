import sys
import pathlib
import torch
import random
from functools import partial
from itertools import combinations


sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[2])
)

from core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype
from core.utils import precompute_freqs_cis, rotate, get_attn_info, get_moe_tokens_info
from core.utils import smooth_per_token_dynamic_quant, static_quant
from core.op import BasicOp



def fake_quant_gemm(
    tokens, per_token_scale, 
    weights, weight_scale, 
    dst_torch_dtype=torch.bfloat16, 
    trans_w=False,
):
    if trans_w:
        weights = weights.transpose(0, 1)
    fake_gemm_output = torch.matmul(
        tokens.type(torch.bfloat16), 
        weights.type(torch.bfloat16)
    )
    dequant_scale = torch.matmul(
        per_token_scale.unsqueeze(-1), 
        weight_scale.unsqueeze(0)
    )
    y = torch.mul(
        fake_gemm_output, 
        dequant_scale
    ).type(dst_torch_dtype)
    return y
    


"""
******************************************
Norm & Quant 算子
******************************************
"""


class ScaleDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError(f"arg_type {self.arg_type} is not supported")

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        src_avail_dtype = ["bfloat16"]
        if not self.dtype in src_avail_dtype:
            raise ValueError(f"dtype {self.dtype} is not supported")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        dst_avail_dtype = ["int8", "float8"]
        if not self.dst_dtype in dst_avail_dtype:
            raise ValueError(f"dst_dtype {self.dst_dtype} is not supported")
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name()
            )
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
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

        # run func
        self._run_func = self.scale_dynamic_quant_run


    def scale_dynamic_quant_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]

        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
            hidden_states, 
            smooth_scale, 
            self.dst_torch_dtype
        )

        return quant_tokens, per_token_scale



class HeadRMSNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.total_head_num = self.args_dict["total_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.norm_head_start = self.args_dict["norm_head_start"]
        self.norm_head_num = self.args_dict["norm_head_num"]
        self.norm_head_end = self.norm_head_start + self.norm_head_num

        self.eps = 1e-5

        # in-place
        self.input_tensor_info = {
            "token_data": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "norm_weight": OpTensorInfo(
                shape=[self.head_dim, ],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }
        self.output_tensor_info = {}

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_head_num * self.norm_head_num + \
            calc_tensor_size(self.input_tensor_info["norm_weight"])
        self.write_bytes = \
            calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_head_num * self.norm_head_num
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )

        # run func
        self._run_func = self.head_rms_norm_run


    def head_rms_norm_run(self, tensor_mapping):
        # get pre-allocated input tensors
        token_data = tensor_mapping["token_data"]
        norm_weight = tensor_mapping["norm_weight"]

        # in-place norm on specified heads
        head_data = token_data[:, self.norm_head_start:self.norm_head_end, :]
        head_data = torch.nn.functional.rms_norm(
            head_data, 
            normalized_shape=head_data.shape[-1:],
            weight=norm_weight,
            eps=self.eps
        )

        return head_data



class HeadRMSNormDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8", "float8"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.head_num = self.args_dict["head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.eps = 1e-5

        # out-place
        self.input_tensor_info = {
            "token_data": OpTensorInfo(
                shape=[self.num_tokens, self.head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "norm_weight": OpTensorInfo(
                shape=[self.head_dim, ],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.head_num * self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.head_num * self.head_dim], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name()
            )
        }

        # calculator
        self.input_tensor_size = 2 * sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.write_bytes = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )

        # run func
        self._run_func = self.head_rms_norm_dynamic_quant_run

    def head_rms_norm_dynamic_quant_run(self, tensor_mapping):
        # get pre-allocated input tensors
        token_data = tensor_mapping["token_data"]
        norm_weight = tensor_mapping["norm_weight"]
        smooth_scale = tensor_mapping["smooth_scale"]

        # per head rms_norm
        after_norm = torch.nn.functional.rms_norm(
            token_data, 
            normalized_shape=token_data.shape[-1:],
            weight=norm_weight,
            eps=self.eps
        )
        after_norm = after_norm.view(self.num_tokens, self.head_num * self.head_dim)

        # per token dynamic quant
        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
            after_norm, smooth_scale, self.dst_torch_dtype
        )

        return quant_tokens, per_token_scale
        


class AddRmsNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError("AddRmsNormOp only support llm arg_type")
        
        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError("AddRmsNormOp only support bfloat16 dtype")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        self.eps = 1e-5

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "residual": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "norm_weight": OpTensorInfo(
                shape=[self.hidden_size, ],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
        }
        self.output_tensor_info = {
            "after_res": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "output": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
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

        # run func
        self._run_func = self.add_rms_norm_run

    def add_rms_norm_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        residual = tensor_mapping["residual"]
        norm_weight = tensor_mapping["norm_weight"]

        output = torch.nn.functional.rms_norm(
            hidden_states + residual, 
            normalized_shape=hidden_states.shape[-1:],
            weight=norm_weight,
            eps=self.eps
        )

        return output

        




class AddRmsNormDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError
        
        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.add_residual = self.args_dict.get("add_residual", True)
        self.output_mode = self.args_dict.get("output_mode", "none")
        if not self.output_mode in ["none", "res", "norm"]:
            raise ValueError

        self.eps = 1e-5

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            # use 1 as smooth scale, fuse norm_weight
            "smooth_scale": OpTensorInfo(
                shape=[self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }
        if self.add_residual:
            self.input_tensor_info["residual"] = OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )

        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name()
            )
        }
        if self.output_mode == "res":
            self.output_tensor_info["after_res"] = OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        elif self.output_mode == "norm":
            self.output_tensor_info["after_norm"] = OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
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

        # run func
        self._run_func = self.add_rms_norm_dynamic_quant_run


    def add_rms_norm_dynamic_quant_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        residual = tensor_mapping.get("residual", None)
        smooth_scale = tensor_mapping["smooth_scale"]

        # add residual
        after_res = hidden_states
        if residual is not None:
            after_res = hidden_states + residual

        # rms norm
        after_norm = torch.nn.functional.rms_norm(
            after_res, 
            normalized_shape=after_res.shape[-1:], 
            eps=self.eps
        )

        # dynamic quant
        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(after_norm, smooth_scale)

        if self.output_mode == "none":
            return quant_tokens, per_token_scale
        elif self.output_mode == "res":
            return quant_tokens, per_token_scale, after_res
        elif self.output_mode == "norm":
            return quant_tokens, per_token_scale, after_norm




"""
******************************************
Attention & rope & kvcache 算子
******************************************
"""


class RotaryEmbeddingOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm", "batch_llm"]:
            raise ValueError
        
        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if not self.attn_mode in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)

        # src_dtype
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num
        self.head_dim = self.args_dict["head_dim"]
        self.rope_offset = self.args_dict.get("rope_offset", 0)
        self.rope_dim = self.args_dict["rope_dim"]

        cos_tensor, sin_tensor = precompute_freqs_cis(self.max_kv_len, self.rope_dim)

        self.input_tensor_info = {
            "packed_qkv": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "q_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ), 
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ), 
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
            ), 
            "cos": OpTensorInfo(
                shape=[self.max_kv_len, self.rope_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: cos_tensor.to(dtype=dtype, device=device)
            ), 
            "sin": OpTensorInfo(
                shape=[self.max_kv_len, self.rope_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: sin_tensor.to(dtype=dtype, device=device)
            ), 
        }
        self.output_tensor_info = {}

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) / self.total_head_num * (self.q_head_num + self.kv_head_num) + \
            calc_tensor_size(self.input_tensor_info["q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["accum_q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["cache_lens"]) + \
            calc_tensor_size(self.input_tensor_info["cos"]) + \
            calc_tensor_size(self.input_tensor_info["sin"])

        self.write_bytes = \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) / self.total_head_num * (self.q_head_num + self.kv_head_num)
        
        self.io_bytes = self.read_bytes + self.write_bytes


        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )

        # run func
        self._run_func = self.rotary_embedding_run


    def rotary_embedding_run(self, tensor_mapping):
        # get pre-allocated input tensors
        packed_qkv = tensor_mapping["packed_qkv"]
        q_lens = tensor_mapping["q_lens"]
        accum_q_lens = tensor_mapping["accum_q_lens"]
        cache_lens = tensor_mapping["cache_lens"]

        cos = tensor_mapping["cos"]
        sin = tensor_mapping["sin"]
        

        # for each batch
        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            q_offset = self.accum_q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]

            token_start = q_offset
            token_end = q_offset + q_len

            qk_head_start = 0
            qk_head_end = self.q_head_num + self.kv_head_num

            dim_start = self.rope_offset
            dim_end = self.rope_offset + self.rope_dim

            cache_start = cache_len
            cache_end = cache_len + q_len

            target_qk = packed_qkv[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end].contiguous()
            target_cos = cos[cache_start:cache_end]
            target_sin = sin[cache_start:cache_end]

            packed_qkv[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end].copy_(
                rotate(target_qk, target_cos, target_sin)
            )

        return packed_qkv


class StoreKVCacheOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm", "batch_llm"]:
            raise ValueError
        
        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if not self.attn_mode in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)
        

        """
        src_dtype / cache_dtype
        """
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        self.cache_dtype = self.args_dict.get("cache_dtype", "bfloat16")
        if not self.cache_dtype in ["bfloat16", "int8", "float8"]:
            raise ValueError
        self.cache_torch_dtype = get_torch_dtype(self.cache_dtype)

        if self.cache_dtype in ["int8", "float8"]:
            self.use_quant = True
        elif self.cache_dtype == "bfloat16":
            self.use_quant = False


        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num
        self.head_dim = self.args_dict["head_dim"]


        # target quant method for k_cache/v_cache
        self.quant_mode = self.args_dict.get("quant_mode", "static")
        if not self.quant_mode in ["static", "dynamic_per_token", "dynamic_per_block"]:
            raise ValueError

        # all tokens with same head/head_dim element pos share one scale
        if self.quant_mode == "static":
            self.quant_scale_shape = [
                self.kv_head_num, self.head_dim
            ]
        # head_dim per token share one scale
        elif self.quant_mode == "dynamic_per_token":
            self.quant_scale_shape = [
                self.batch_size, self.kv_head_num,self.max_kv_len
            ]
        # [seq_quant_block_size, head_dim_quant_block_size] share one scale
        elif self.quant_mode == "dynamic_per_block":
            self.seq_quant_block_size = self.args_dict.get("seq_quant_block_size", 16)
            self.head_dim_quant_block_size = self.args_dict.get("head_dim_quant_block_size", 16)

            self.quant_scale_shape = [
                self.batch_size, 
                self.kv_head_num, 
                self.max_kv_len // self.seq_quant_block_size, 
                self.head_dim // self.head_dim_quant_block_size
            ]





        self.input_tensor_info = {
            "packed_qkv": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "slot_mapping": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.slot_mapping, dtype=dtype, device=device)
            ), 

            "q_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ), 
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ), 
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
            )
        }

        if self.cache_type == "linear":
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.cache_torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.cache_torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
        elif self.cache_type == "paged":
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.batch_size, self.max_block_num_per_seq], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * -1
            )
        
        if self.use_quant:
            self.input_tensor_info["k_scale"] = OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            )
            self.input_tensor_info["v_scale"] = OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            )
        
        self.output_tensor_info = {}


        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) / self.total_head_num * 2 * self.kv_head_num + \
            calc_tensor_size(self.input_tensor_info["slot_mapping"]) + \
            calc_tensor_size(self.input_tensor_info["q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["accum_q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["cache_lens"])

        if self.use_quant:
            self.read_bytes += \
                calc_tensor_size(self.input_tensor_info["k_scale"]) + \
                calc_tensor_size(self.input_tensor_info["v_scale"])

        if self.cache_type == "linear":
            self.write_bytes = \
                calc_tensor_size(self.input_tensor_info["k_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens + \
                calc_tensor_size(self.input_tensor_info["v_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens
        elif self.cache_type == "paged":
            self.write_bytes = \
                calc_tensor_size(self.input_tensor_info["k_cache"]) / self.num_kv_blocks / self.block_size * self.num_tokens + \
                calc_tensor_size(self.input_tensor_info["v_cache"]) / self.num_kv_blocks / self.block_size * self.num_tokens + \
                calc_tensor_size(self.input_tensor_info["block_table"]) / self.batch_size / self.max_block_num_per_seq * self.num_q_blocks

        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )

        # run func
        self._run_func = self.store_kv_cache_run


    def store_kv_cache_run(self, tensor_mapping):
        # get pre-allocated input tensors
        packed_qkv = tensor_mapping["packed_qkv"]
        slot_mapping = tensor_mapping["slot_mapping"]

        q_lens = tensor_mapping["q_lens"]
        accum_q_lens = tensor_mapping["accum_q_lens"]
        cache_lens = tensor_mapping["cache_lens"]
        
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        k_scale = None if "k_scale" not in tensor_mapping else tensor_mapping["k_scale"]
        v_scale = None if "v_scale" not in tensor_mapping else tensor_mapping["v_scale"]

        """
        参考linear cache的实现改写成paged cache的实现
        """
        if self.cache_type == "paged":
            raise NotImplementedError("StoreKVCacheOp paged cache not implemented yet.")
        
        if self.cache_type == "linear":
            for batch_idx in range(self.batch_size):
                kv_slot_id = self.slot_mapping[batch_idx]
                q_len = self.q_lens[batch_idx]
                q_offset = self.accum_q_lens[batch_idx]
                cache_len = self.cache_lens[batch_idx]

                token_start = q_offset
                token_end = q_offset + q_len

                k_head_start = self.q_head_num
                k_head_end = self.q_head_num + self.kv_head_num

                v_head_start = self.q_head_num + self.kv_head_num
                v_head_end = self.q_head_num + self.kv_head_num * 2

                cache_start = cache_len
                cache_end = cache_len + q_len

                # [q_len, kv_head_num, head_dim]
                # bfloat16
                src_k_data = packed_qkv[token_start:token_end, k_head_start:k_head_end, :]
                src_v_data = packed_qkv[token_start:token_end, v_head_start:v_head_end, :]

                # [kv_head_num, q_len, head_dim]
                # bfloat16 / int8 / float8
                dst_k_cache = k_cache[kv_slot_id, :, cache_start:cache_end, :]
                dst_v_cache = v_cache[kv_slot_id, :, cache_start:cache_end, :]

                if self.use_quant:
                    dst_k_cache.copy_(static_quant(src_k_data, k_scale, self.cache_torch_dtype).contiguous().transpose(0, 1))
                    dst_v_cache.copy_(static_quant(src_v_data, v_scale, self.cache_torch_dtype).contiguous().transpose(0, 1))
                else:
                    dst_k_cache.copy_(src_k_data.contiguous().transpose(0, 1))
                    dst_v_cache.copy_(src_v_data.contiguous().transpose(0, 1))

        return k_cache, v_cache




class DequantKVCacheOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm", "batch_llm"]:
            raise ValueError
        
        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if not self.attn_mode in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)

        """
        src_dtype / cache_dtype
        """
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        self.cache_dtype = self.args_dict.get("cache_dtype", "bfloat16")
        if not self.cache_dtype in ["bfloat16", "int8", "float8"]:
            raise ValueError
        self.cache_torch_dtype = get_torch_dtype(self.cache_dtype)


        # pre-defined attrs
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]


        # target quant method for k_cache/v_cache
        self.quant_mode = self.args_dict.get("quant_mode", "static")
        if not self.quant_mode in ["static", "dynamic_per_token", "dynamic_per_block"]:
            raise ValueError

        # all tokens with same head/head_dim element pos share one scale
        if self.quant_mode == "static":
            self.quant_scale_shape = [
                self.kv_head_num, self.head_dim
            ]
        # head_dim per token share one scale
        elif self.quant_mode == "dynamic_per_token":
            self.quant_scale_shape = [
                self.batch_size, self.kv_head_num,self.max_kv_len
            ]
        # [seq_quant_block_size, head_dim_quant_block_size] share one scale
        elif self.quant_mode == "dynamic_per_block":
            self.seq_quant_block_size = self.args_dict.get("seq_quant_block_size", 16)
            self.head_dim_quant_block_size = self.args_dict.get("head_dim_quant_block_size", 16)

            self.quant_scale_shape = [
                self.batch_size, 
                self.kv_head_num, 
                self.max_kv_len // self.seq_quant_block_size, 
                self.head_dim // self.head_dim_quant_block_size
            ]



        self.input_tensor_info = {
            "slot_mapping": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.slot_mapping, dtype=dtype, device=device)
            ), 
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.kv_lens, dtype=dtype, device=device)
            ), 
            "k_scale": OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            ), 
            "v_scale": OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            ),
        }
        self.output_tensor_info = {}

        if self.cache_type == "linear":
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )

            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )


        elif self.cache_type == "paged":
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.batch_size, self.max_block_num_per_seq], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * -1
            )

            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim], 
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim], 
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["slot_mapping"]) + \
            calc_tensor_size(self.input_tensor_info["kv_lens"]) + \
            calc_tensor_size(self.input_tensor_info["k_scale"]) + \
            calc_tensor_size(self.input_tensor_info["v_scale"])

        if self.cache_type == "linear":
            self.read_bytes += \
                calc_tensor_size(self.input_tensor_info["k_cache"]) / self.batch_size / self.max_kv_len * self.num_kv_tokens + \
                calc_tensor_size(self.input_tensor_info["v_cache"]) / self.batch_size / self.max_kv_len * self.num_kv_tokens
            
            self.write_bytes = \
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"]) / self.batch_size / self.max_kv_len * self.num_kv_tokens + \
                calc_tensor_size(self.output_tensor_info["dequant_v_cache"]) / self.batch_size / self.max_kv_len * self.num_kv_tokens
        elif self.cache_type == "paged":
            self.read_bytes += \
                calc_tensor_size(self.input_tensor_info["k_cache"]) / self.num_kv_blocks / self.block_size * self.num_kv_tokens + \
                calc_tensor_size(self.input_tensor_info["v_cache"]) / self.num_kv_blocks / self.block_size * self.num_kv_tokens + \
                calc_tensor_size(self.input_tensor_info["block_table"]) / self.batch_size / self.max_block_num_per_seq * self.num_kv_blocks
                
            self.write_bytes = \
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"]) / self.num_kv_blocks / self.block_size * self.num_kv_tokens + \
                calc_tensor_size(self.output_tensor_info["dequant_v_cache"]) / self.num_kv_blocks / self.block_size * self.num_kv_tokens
        
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )

        # run func
        self._run_func = self.dequant_kv_cache_run


    def dequant_kv_cache_run(self, tensor_mapping):
        slot_mapping = tensor_mapping["slot_mapping"]
        kv_lens = tensor_mapping["kv_lens"]

        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]
        k_scale = tensor_mapping["k_scale"]
        v_scale = tensor_mapping["v_scale"]
        dequant_k_cache = tensor_mapping["dequant_k_cache"]
        dequant_v_cache = tensor_mapping["dequant_v_cache"]

        """
        参考linear cache的实现改写成paged cache的实现
        """
        if self.cache_type == "paged":
            raise NotImplementedError("DequantKVCacheOp paged cache not implemented yet.")
        
        if self.cache_type == "linear":
            for batch_idx in range(self.batch_size):
                kv_slot_id = self.slot_mapping[batch_idx]
                kv_len = self.kv_lens[batch_idx]

                # [kv_head_num, kv_len, head_dim]
                # int8 / float8
                src_k_cache = k_cache[kv_slot_id, :, :kv_len, :].contiguous()
                src_v_cache = v_cache[kv_slot_id, :, :kv_len, :].contiguous()

                # [kv_head_num, kv_len, head_dim]
                # bfloat16
                dst_k_cache = dequant_k_cache[kv_slot_id, :, :kv_len, :]
                dst_v_cache = dequant_v_cache[kv_slot_id, :, :kv_len, :]

                dst_k_cache.copy_(torch.mul(src_k_cache.to(k_scale.dtype), k_scale.unsqueeze(1)).to(dtype=self.cache_torch_dtype))
                dst_v_cache.copy_(torch.mul(src_v_cache.to(v_scale.dtype), v_scale.unsqueeze(1)).to(dtype=self.cache_torch_dtype))

        return dequant_k_cache, dequant_v_cache





class FlashAttentionOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm", "batch_llm"]:
            raise ValueError
        
        # get attention info
        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if not self.attn_mode in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)
        

        # q/o dtype
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise ValueError(f"FlashAttentionOp dtype {self.dtype} not supported.")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # target qk and pv compute dtype
        self.compute_dtype = self.args_dict.get("compute_dtype", self.dtype)
        if not self.compute_dtype in ["bfloat16", "int8", "float8"]:
            raise ValueError(f"FlashAttentionOp compute_dtype {self.compute_dtype} not supported.")
        self.compute_torch_dtype = get_torch_dtype(self.compute_dtype)

        # k_cache/v_cache dtype
        self.cache_dtype = self.args_dict.get("cache_dtype", "bfloat16")
        if not self.cache_dtype in ["bfloat16", "int8", "float8"]:
            raise ValueError(f"FlashAttentionOp cache_dtype {self.cache_dtype} not supported.")
        self.cache_torch_dtype = get_torch_dtype(self.cache_dtype)
        if self.cache_dtype in ["int8", "float8"]:
            self.use_quant = True
        elif self.cache_dtype == "bfloat16":
            self.use_quant = False

        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.softmax_scale = self.head_dim ** (-0.5)
        self.is_causal = True


        # target quant method for k_cache/v_cache
        self.quant_mode = self.args_dict.get("quant_mode", "static")
        if not self.quant_mode in ["static", "dynamic_per_token", "dynamic_per_block"]:
            raise ValueError


        # all tokens with same head/head_dim element pos share one scale
        if self.quant_mode == "static":
            self.quant_scale_shape = [
                self.kv_head_num, self.head_dim
            ]
        # head_dim per token share one scale
        elif self.quant_mode == "dynamic_per_token":
            self.quant_scale_shape = [
                self.batch_size, self.kv_head_num,self.max_kv_len
            ]
        # [seq_quant_block_size, head_dim_quant_block_size] share one scale
        elif self.quant_mode == "dynamic_per_block":
            self.seq_quant_block_size = self.args_dict.get("seq_quant_block_size", 16)
            self.head_dim_quant_block_size = self.args_dict.get("head_dim_quant_block_size", 16)

            self.quant_scale_shape = [
                self.batch_size, 
                self.kv_head_num, 
                self.max_kv_len // self.seq_quant_block_size, 
                self.head_dim // self.head_dim_quant_block_size
            ]




        # define basic input/outpus, could be overrided
        self.input_tensor_info = {
            # slice from packed_qkv, stride on head_num
            "q": OpTensorInfo(
                shape=[self.num_tokens, self.q_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
            "slot_mapping": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.slot_mapping, dtype=dtype, device=device)
            ), 

            "q_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ), 
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ), 
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.kv_lens, dtype=dtype, device=device)
            ), 
        }

        if self.cache_type == "linear":
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.cache_torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim], 
                dtype=self.cache_torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
        elif self.cache_type == "paged":
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.batch_size, self.max_block_num_per_seq], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * -1
            )

        if self.use_quant:
            self.input_tensor_info["k_scale"] = OpTensorInfo(
                shape=self.quant_scale_shape, 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            )
            self.input_tensor_info["v_scale"] = OpTensorInfo(
                shape=self.quant_scale_shape, 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            )


        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.num_tokens, self.q_head_num, self.head_dim], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name()
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

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["q"]) + \
            calc_tensor_size(self.input_tensor_info["slot_mapping"]) + \
            calc_tensor_size(self.input_tensor_info["q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["accum_q_lens"]) + \
            calc_tensor_size(self.input_tensor_info["kv_lens"])
        
        if self.use_quant:
            self.read_bytes += \
                calc_tensor_size(self.input_tensor_info["k_scale"]) + \
                calc_tensor_size(self.input_tensor_info["v_scale"])

        if self.cache_type == "linear":
            self.read_bytes = \
                calc_tensor_size(self.input_tensor_info["k_cache"]) / self.batch_size / self.max_kv_len * self.num_kv_tokens + \
                calc_tensor_size(self.input_tensor_info["v_cache"]) / self.batch_size / self.max_kv_len * self.num_kv_tokens
        elif self.cache_type == "paged":
            self.read_bytes = \
                calc_tensor_size(self.input_tensor_info["k_cache"]) / self.num_kv_blocks / self.block_size * self.num_kv_tokens + \
                calc_tensor_size(self.input_tensor_info["v_cache"]) / self.num_kv_blocks / self.block_size * self.num_kv_tokens + \
                calc_tensor_size(self.input_tensor_info["block_table"]) / self.batch_size / self.max_block_num_per_seq * self.num_kv_blocks

        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes


        self.calc_flops = 0
        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]
            kv_len = self.kv_lens[batch_idx]

            """
            q_len = 8, cache_len = 0, kv_len = 8
            total = kv_len * kv_len = 64
            valid = (cache_len + 1 + kv_len) * q_len / 2 = 36
            ratio = 36 / 64 = 0.5625
            * - - - - - - - 
            * * - - - - - - 
            * * * - - - - - 
            * * * * - - - - 
            * * * * * - - - 
            * * * * * * - - 
            * * * * * * * - 
            * * * * * * * *

            q_len = 4, cache_len = 4, kv_len = 8
            total = kv_len * kv_len = 64
            valid = (cache_len + 1 + kv_len) * q_len / 2 = 26
            ratio = 26 / 64 = 0.40625
            - - - - - - - - 
            - - - - - - - - 
            - - - - - - - - 
            - - - - - - - - 
            * * * * * - - - 
            * * * * * * - - 
            * * * * * * * - 
            * * * * * * * *
            """

            valid_parts = kv_len * kv_len
            if self.is_causal:
                valid_parts = (cache_len + 1 + kv_len) * q_len / 2
            else:
                valid_parts = q_len * kv_len

            # p = q * v, bf16/int8/fp8 batch_gemm
            # o = p * v, bf16/int8/fp8 batch_gemm
            self.calc_flops += 2 * (self.q_head_num * self.head_dim * valid_parts * 2)


        # specify create input/output tensors func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )
        
        # specify run function
        self._run_func = self.flash_attention_run


    def flash_attention_run(self, tensor_mapping):
        raise NotImplementedError






"""
******************************************
gemm & group_gemm & moe_ops
******************************************
"""


class MoeGatingGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16", "float32"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        self.dst_dtype = self.args_dict.get("dst_dtype", "float32")
        if not self.dst_dtype in ["float32"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.num_experts = self.args_dict["num_experts"]

        self.trans_w = self.args_dict.get("trans_w", False)
        
        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "gating_weight": OpTensorInfo(
                shape=[self.num_experts, self.hidden_size] if self.trans_w \
                    else [self.hidden_size, self.num_experts], 
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

        # run func
        self._run_func = self.moe_gating_gemm_run


    def moe_gating_gemm_run(self, tensor_mapping):
        gating_output = torch.mm(
            tensor_mapping["hidden_states"], 
            tensor_mapping["gating_weight"].t() if self.trans_w \
                else tensor_mapping["gating_weight"]
        ).type(self.dst_torch_dtype)
        return gating_output



class QuantMatmulOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError("QuantMatmulOp only support llm arg_type")

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8", "float8"]:
            raise ValueError(f"QuantMatmulOp only support int8, float8_e4m3, float8_e5m2, but got {self.dtype}")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["bfloat16"]:
            raise ValueError(f"QuantMatmulOp only support bfloat16 dst_dtype, but got {self.dst_dtype}")
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]
        self.trans_w = self.args_dict.get("trans_w", False)
        self.transpose_o = self.args_dict.get("transpose_o", False)

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "expert_weight": OpTensorInfo(
                shape=[self.new_hidden_size, self.hidden_size] \
                    if self.trans_w else [self.hidden_size, self.new_hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "expert_scale": OpTensorInfo(
                shape=[self.new_hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            )
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(
                shape=[self.num_tokens, self.new_hidden_size], 
                dtype=self.dst_torch_dtype
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

        self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.new_hidden_size


        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )

        # run func
        self._run_func = self.quant_matmul_run


    def quant_matmul_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        per_token_scale = tensor_mapping["per_token_scale"]
        expert_weight = tensor_mapping["expert_weight"]
        expert_scale = tensor_mapping["expert_scale"]

        # [num_tokens // sp_size, new_hidden_size]
        # [num_tokens // sp_size, sp_size, new_hidden_size // sp_size]
        # [sp_size, new_hidden_size // sp_size, new_hidden_size // sp_size]
        y = fake_quant_gemm(
            hidden_states, per_token_scale, 
            expert_weight, expert_scale, 
            dst_torch_dtype=self.dst_torch_dtype, 
            trans_w=self.trans_w
        )
        if self.transpose_o and self.sp_size > 1:
            y = y.view(self.num_tokens, self.sp_size, self.new_hidden_size // self.sp_size)
            y = y.transpose(0, 1).contiguous().view(self.num_tokens, self.new_hidden_size)
        return y


class QuantGroupGemmReduceSumOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError("QuantGroupGemmReduceSumOp only support llm arg_type")
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8", "float8"]:
            raise ValueError(f"QuantGroupGemmReduceSumOp only support int8, but got {self.dtype}")
        self.torch_dtype = get_torch_dtype(self.dtype)

        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["bfloat16"]:
            raise ValueError(f"QuantGroupGemmReduceSumOp only support bfloat16 dst_dtype, but got {self.dst_dtype}")
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]
        self.trans_w = self.args_dict.get("trans_w", False)

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.sp_size, self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.sp_size, self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "weight": OpTensorInfo(
                shape=[self.sp_size, self.new_hidden_size, self.hidden_size] \
                    if self.trans_w else [self.sp_size, self.hidden_size, self.new_hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "weight_scale": OpTensorInfo(
                shape=[self.sp_size, self.new_hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
        }
        self.output_tensor_info = {
            "output": OpTensorInfo(
                shape=[self.num_tokens, self.new_hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name()
            ), 
        }

        # calculator
        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ]) + calc_tensor_size(self.output_tensor_info["output"])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 2 * self.sp_size * self.num_tokens * self.new_hidden_size * self.hidden_size

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )

        # run func
        self._run_func = self.quant_group_gemm_reduce_sum_run

    def quant_group_gemm_reduce_sum_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        per_token_scale = tensor_mapping["per_token_scale"]
        weight = tensor_mapping["weight"]
        weight_scale = tensor_mapping["weight_scale"]

        # quant group gemm
        temp_tensor = torch.empty(
            [self.sp_size, self.num_tokens, self.new_hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=hidden_states.device
        )

        for sp_idx in range(self.sp_size):
            cur_tokens = hidden_states[sp_idx]
            cur_tokens_scale = per_token_scale[sp_idx]
            cur_weight = weight[sp_idx]
            cur_weight_scale = weight_scale[sp_idx]

            temp_tensor[sp_idx] = fake_quant_gemm(
                cur_tokens, cur_tokens_scale, 
                cur_weight, cur_weight_scale, 
                dst_torch_dtype=self.dst_torch_dtype, 
                trans_w=self.trans_w,
            )

        # reduce sum
        output = torch.sum(temp_tensor, dim=0, keepdim=False)

        return output
            

        
        




class MoeQuantGroupGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8", "float8", "w4a8"]:
            raise ValueError(f"MoeQuantGroupGemmOp only support int8, w4a8 dtype, but got {self.dtype}")
        if self.dtype == "w4a8":
            self.torch_dtype = torch.int8
        else:
            self.torch_dtype = get_torch_dtype(self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["bfloat16"]:
            raise ValueError(f"MoeQuantGroupGemmOp only support bfloat16 dst_dtype, but got {self.dst_dtype}")
        self.dst_torch_dtype = get_torch_dtype(self.dtype)


        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]
        self.trans_w = self.args_dict.get("trans_w", False)

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

        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "experts_weight": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.new_hidden_size, self.hidden_size] if self.trans_w else [self.num_experts_per_rank, self.hidden_size, self.new_hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "experts_scale": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.new_hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "experts_token_count": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_count, dtype=dtype, device=device)
            ), 
            "experts_token_offset": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset, dtype=dtype, device=device)
            )
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(
                shape=[self.dispatch_tokens, self.new_hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
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

        self.calc_flops = 2 * self.dispatch_tokens * self.hidden_size * self.new_hidden_size


        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )

        # run func
        self._run_func = self.moe_quant_group_gemm_run



    def moe_quant_group_gemm_run(self, tensor_mapping):
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
                trans_w=self.trans_w
            )
        return y



class MoeQuantGroupGemmCombineOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError
        
        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8", "float8"]:
            raise ValueError(f"MoeQuantGroupGemmCombineOp only support int8, float8 dtype, but got {self.dtype}")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["bfloat16"]:
            raise ValueError(f"MoeQuantGroupGemmCombineOp only support bfloat16 dst_dtype, but got {self.dst_dtype}")
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]
        self.trans_w = self.args_dict.get("trans_w", False)

        # moe info
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        # parallel info
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.ep_rank = self.args_dict.get("ep_rank", 0)


        # resiual info
        self.sp_size = self.args_dict.get("sp_size", None)
        self.sp_rank = self.args_dict.get("sp_rank", 0)
        self.res_scale = self.args_dict.get("res_scale", 1.0)
        self.has_residual = True if self.sp_size is not None else False

        self.num_res_tokens_per_rank = (self.num_tokens + self.sp_size - 1) // self.sp_size
        self.res_token_start = self.sp_rank * self.num_res_tokens_per_rank
        self.res_token_end = min(self.res_token_start + self.num_res_tokens_per_rank, self.num_tokens)


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


        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.zeros
            ), 
            "experts_weight": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.new_hidden_size, self.hidden_size] \
                    if self.trans_w else [self.num_experts_per_rank, self.hidden_size, self.new_hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            ), 
            "experts_scale": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.new_hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "experts_token_count": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_count, dtype=dtype, device=device)
            ), 
            "experts_token_offset": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset, dtype=dtype, device=device)
            ), 
            "scatter_token_id": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_id, 
                    dtype=dtype, device=device)
            ),
            "scatter_token_weight": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_weight, 
                    dtype=dtype, device=device)
            ),
        }
        if self.has_residual:
            self.input_tensor_info["residual_tokens"] = OpTensorInfo(
                shape=[self.num_res_tokens_per_rank, self.new_hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )

        self.output_tensor_info = {
            # init zero
            "convergent_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.new_hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ),
        }




        new_scatter_tokens_bytes = calc_tensor_size(
            OpTensorInfo(
                shape=[self.dispatch_tokens, self.new_hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )
        )

        # reserved memory
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()]) + \
            new_scatter_tokens_bytes
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        # in out bytes
        self.read_bytes = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.read_bytes += new_scatter_tokens_bytes
        self.write_bytes += new_scatter_tokens_bytes
        self.io_bytes = self.read_bytes + self.write_bytes

        # flops
        self.calc_flops = 2 * self.dispatch_tokens * self.hidden_size * self.new_hidden_size

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )

        self._run_func = self.moe_quant_group_gemm_combine_run

    def moe_quant_group_gemm_combine_run(self, tensor_mapping):
        scatter_tokens = tensor_mapping["scatter_tokens"]
        experts_weight = tensor_mapping["experts_weight"]
        per_token_scale = tensor_mapping["per_token_scale"]
        experts_scale = tensor_mapping["experts_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]
        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]
        
        new_scatter_tokens = torch.empty(
            size=[self.dispatch_tokens, self.new_hidden_size], 
            dtype=self.dst_torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )
        for expert_idx in range(self.num_experts_per_rank):
            cur_token_start = experts_token_offset[expert_idx]
            cur_token_end = cur_token_start + experts_token_count[expert_idx]

            cur_tokens = scatter_tokens[cur_token_start:cur_token_end]
            cur_tokens_scale = per_token_scale[cur_token_start:cur_token_end]

            cur_weight = experts_weight[expert_idx]
            cur_weight_scale = experts_scale[expert_idx]

            new_scatter_tokens[cur_token_start:cur_token_end] = fake_quant_gemm(
                cur_tokens, cur_tokens_scale, 
                cur_weight, cur_weight_scale, 
                dst_torch_dtype=self.dst_torch_dtype,
                trans_w=self.trans_w
            )

        residual_tokens = tensor_mapping.get("residual_tokens", None)
        convergent_tokens = tensor_mapping["convergent_tokens"]
        if residual_tokens is not None:
            convergent_tokens[self.res_token_start:self.res_token_end] += residual_tokens

        convergent_tokens.index_add_(
            0, scatter_token_id, 
            (new_scatter_tokens * scatter_token_weight.unsqueeze(-1)).to(self.dst_torch_dtype)
        )

        return convergent_tokens





class MoeSoftmaxTopkOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32"]:
            raise ValueError
        self.torch_dtype = getattr(torch, self.dtype)

        # pre-defined attrs
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        self.compute_mode = self.args_dict["compute_mode"]
        if not self.compute_mode in ["pre-softmax", "post-softmax"]:
            raise ValueError

        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        
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
                dtype=self.torch_dtype, 
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

        # run func
        self._run_func = self.moe_softmax_topk_run


    def moe_softmax_topk_run(self, tensor_mapping):
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
        else:
            raise ValueError



class MoeScatterDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8", "float8"]:
            raise ValueError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

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


        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "experts_smooth_scale": OpTensorInfo(
                shape=[self.num_experts, self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "selected_experts": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.all_select_experts, dtype=dtype, device=device)
            ), 
            # complete moe_weights
            "moe_weights": OpTensorInfo(
                shape=[self.num_tokens, self.topk], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.all_select_weights, dtype=dtype, device=device)
            ), 
        }
        self.output_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ), 
            "scatter_per_token_scale": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.ones
            ), 
            "scatter_token_id": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_id, dtype=dtype, device=device)
            ), 
            "scatter_token_weight": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_weight, dtype=dtype, device=device)
            ), 
            "experts_token_count": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_count, dtype=dtype, device=device)
            ), 
            "experts_token_offset": OpTensorInfo(
                shape=[self.num_experts_per_rank + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset, dtype=dtype, device=device)
            )
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = \
            calc_tensor_size(self.input_tensor_info["hidden_states"]) / self.num_tokens * self.used_src_tokens + \
            calc_tensor_size(self.input_tensor_info["experts_smooth_scale"]) + \
            calc_tensor_size(self.input_tensor_info["selected_experts"]) + \
            calc_tensor_size(self.input_tensor_info["moe_weights"])
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )

        # run func
        self._run_func = self.moe_scatter_dynamic_quant_run


    def moe_scatter_dynamic_quant_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        experts_smooth_scale = tensor_mapping["experts_smooth_scale"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]
        
        # get pre-allocated output tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]

        # For ease of reference in code demonstration, 
        # all the following tensors are precomputed. 
        # Vendors are required to implement the corresponding computation logic during integration.
        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]
        
        dst_token_id = 0
        for expert_idx in range(self.num_experts_per_rank):
            cur_token_start = self.expert_dispatch_token_offset[expert_idx]
            cur_token_end = cur_token_start + self.expert_dispatch_token_count[expert_idx]
            src_token_indices = scatter_token_id[cur_token_start:cur_token_end]

            scatter_tokens[cur_token_start:cur_token_end], \
            scatter_per_token_scale[cur_token_start:cur_token_end] = \
                smooth_per_token_dynamic_quant(
                    hidden_states[src_token_indices], 
                    experts_smooth_scale[expert_idx], 
                    dst_torch_dtype=self.dst_torch_dtype
                )
            
        return scatter_tokens, scatter_per_token_scale, \
               scatter_token_id, scatter_token_weight, \
               experts_token_count, experts_token_offset



class MoeSwigluDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8", "float8"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

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
        
        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size * 2], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "experts_smooth_scale": OpTensorInfo(
                shape=[self.num_experts_per_rank, self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            "experts_token_count": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_count, dtype=dtype, device=device)
            ), 
            "experts_token_offset": OpTensorInfo(
                shape=[self.num_experts_per_rank], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.expert_dispatch_token_offset, dtype=dtype, device=device)
            )
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
            ),
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
            create_outputs=True
        )

        # run func
        self._run_func = self.moe_swiglu_dynamic_quant_run


    def moe_swiglu_dynamic_quant_run(self, tensor_mapping): 
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        experts_smooth_scale = tensor_mapping["experts_smooth_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_offset = tensor_mapping["experts_token_offset"]

        # get per-allocated output tensors
        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]


        # swiglu, x1 used as gating, x2 used as up
        x1, x2 = torch.chunk(scatter_tokens, 2, dim=-1)
        swiglu_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

        # per expert dynamic quant
        for expert_idx in range(self.num_experts_per_rank):
            cur_token_start = self.expert_dispatch_token_offset[expert_idx]
            cur_token_end = cur_token_start + self.expert_dispatch_token_count[expert_idx]

            quant_tokens[cur_token_start:cur_token_end], \
            per_token_scale[cur_token_start:cur_token_end] = \
                smooth_per_token_dynamic_quant(
                    swiglu_tokens[cur_token_start:cur_token_end], 
                    experts_smooth_scale[expert_idx], 
                    dst_torch_dtype=self.dst_torch_dtype
                )

        return quant_tokens, per_token_scale



class SwigluDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8", "float8"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # predefined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]


        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size * 2], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "smooth_scale": OpTensorInfo(
                shape=[self.hidden_size], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
            ),
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

        # run func
        self._run_func = self.swiglu_dynamic_quant_run


    def swiglu_dynamic_quant_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]

        # swiglu, x1 used as gating, x2 used as up
        x1, x2 = torch.chunk(hidden_states, 2, dim=-1)
        swiglu_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
            swiglu_tokens, smooth_scale, 
            dst_torch_dtype=self.dst_torch_dtype
        )

        return quant_tokens, per_token_scale



class MoeGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = getattr(torch, self.dtype)

        # predefined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        # moe info
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        # parallel info
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.ep_rank = self.args_dict.get("ep_rank", 0)


        # residual info
        self.sp_size = self.args_dict.get("sp_size", None)
        self.sp_rank = self.args_dict.get("sp_rank", 0)
        self.res_scale = self.args_dict.get("res_scale", 1.0)
        self.has_residual = True if self.sp_size is not None else False

        self.num_res_tokens_per_rank = (self.num_tokens + self.sp_size - 1) // self.sp_size
        self.res_token_start = self.sp_rank * self.num_res_tokens_per_rank
        self.res_token_end = min(self.res_token_start + self.num_res_tokens_per_rank, self.num_tokens)

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

        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.dispatch_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "scatter_token_id": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_id, 
                    dtype=dtype, device=device)
            ),
            "scatter_token_weight": OpTensorInfo(
                shape=[self.dispatch_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.scatter_token_weight, 
                    dtype=dtype, device=device)
            ),
        }
        if self.has_residual:
            self.input_tensor_info["residual_tokens"] = OpTensorInfo(
                shape=[self.num_res_tokens_per_rank, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
            )

        self.output_tensor_info = {
            # init zero
            "convergent_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], 
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size


        scatter_tokens_bytes = calc_tensor_size(self.input_tensor_info["scatter_tokens"])
        self.read_bytes = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.read_bytes += scatter_tokens_bytes
        self.write_bytes = scatter_tokens_bytes
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )

        # run func
        self._run_func = self.moe_gather_run


    def moe_gather_run(self, tensor_mapping):
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]

        # get pre-allocated output tensors
        convergent_tokens = tensor_mapping["convergent_tokens"]

        if self.has_residual:
            residual_tokens = tensor_mapping["residual_tokens"]
            convergent_tokens[self.res_token_start:self.res_token_end] += residual_tokens * self.res_scale

        # [dispatch_tokens, hidden_size] --> [num_tokens, hidden_size]
        convergent_tokens.index_add_(
            0, scatter_token_id, 
            (scatter_tokens * scatter_token_weight.unsqueeze(-1)).to(self.torch_dtype)
        )

        return convergent_tokens

