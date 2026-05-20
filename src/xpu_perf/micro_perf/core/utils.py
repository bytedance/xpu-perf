import os
import re
import sys
import itertools
from functools import partial
from typing import List, Dict, Any
from dataclasses import dataclass, field

import torch

from xpu_perf.micro_perf.core.common_utils import (
    logger, setup_logger,
    parse_csv_file, parse_json_file, get_cartesian_product,
)

def bool_creator(size, dtype, device):
    return torch.ones(size, dtype=dtype).to(device=device)


def int_creator(size, dtype, device, cast_dtype=True):
    tensor = torch.randint(
        low=-4, 
        high=3, 
        size=size, 
        dtype=torch.int8
    ).to(device=device)

    if cast_dtype:
        tensor = tensor.to(dtype=dtype)

    return tensor


def uint_creator(size, dtype, device, cast_dtype=True):
    tensor = torch.randint(
        low=0, 
        high=3, 
        size=size, 
        dtype=torch.int8
    ).to(device=device)

    if cast_dtype:
        tensor = tensor.to(dtype=dtype)

    return tensor

def float_creator(size, dtype, device):
    return torch.randn(
        size=size, 
        dtype=torch.float32
    ).to(device=device).to(dtype=dtype)




@dataclass
class TorchDtypeInfo:
    torch_dtype: torch.dtype
    dtype_size: int
    creator: callable



TORCH_DTYPE_MAPPING = {
    "bool": TorchDtypeInfo(torch.bool, 1, bool_creator),

    "int64": TorchDtypeInfo(torch.int64, 8, int_creator),
    "int32": TorchDtypeInfo(torch.int32, 4, int_creator),
    "int16": TorchDtypeInfo(torch.int16, 2, int_creator),
    "int8": TorchDtypeInfo(torch.int8, 1, int_creator),

    "uint64": TorchDtypeInfo(torch.uint64, 8, uint_creator),
    "uint32": TorchDtypeInfo(torch.uint32, 4, uint_creator),
    "uint16": TorchDtypeInfo(torch.uint16, 2, uint_creator),
    "uint8": TorchDtypeInfo(torch.uint8, 1, uint_creator),

    "float": TorchDtypeInfo(torch.float32, 4, float_creator),
    "float32": TorchDtypeInfo(torch.float32, 4, float_creator),
    "tfloat32": TorchDtypeInfo(torch.float32, 4, float_creator), 
    "half": TorchDtypeInfo(torch.float16, 2, float_creator),
    "float16": TorchDtypeInfo(torch.float16, 2, float_creator),
    "bfloat16": TorchDtypeInfo(torch.bfloat16, 2, float_creator)
}


if hasattr(torch, "float8_e4m3fn"):
    TORCH_DTYPE_MAPPING.update({
        "float8": TorchDtypeInfo(torch.float8_e4m3fn, 1, float_creator),
        "float8_e4m3": TorchDtypeInfo(torch.float8_e4m3fn, 1, float_creator),
        "float8_e5m2": TorchDtypeInfo(torch.float8_e5m2, 1, float_creator),

        "mxfloat8": TorchDtypeInfo(torch.float8_e4m3fn, 1, float_creator),
        "mxfloat8_e4m3": TorchDtypeInfo(torch.float8_e4m3fn, 1, float_creator),
        "mxfloat8_e5m2": TorchDtypeInfo(torch.float8_e5m2, 1, float_creator),
    })

if hasattr(torch, "float8_e8m0fnu"):
    TORCH_DTYPE_MAPPING.update({
        "float8_e8m0": TorchDtypeInfo(torch.float8_e8m0fnu, 1, partial(int_creator, cast_dtype=False)),
    })

if hasattr(torch, "float4_e2m1fn_x2"):
    TORCH_DTYPE_MAPPING.update({
        "mxfloat4": TorchDtypeInfo(torch.float4_e2m1fn_x2, 0.5, partial(int_creator, cast_dtype=False)),
        "mxfloat4_e2m1": TorchDtypeInfo(torch.float4_e2m1fn_x2, 0.5, partial(int_creator, cast_dtype=False)),
    })


DTYPE_SIZE_MAPPING = {}
for dtype_str, dtype_info in TORCH_DTYPE_MAPPING.items():
    DTYPE_SIZE_MAPPING[dtype_info.torch_dtype] = dtype_info.dtype_size

CREATOR_MAPPING = {}
for dtype_str, dtype_info in TORCH_DTYPE_MAPPING.items():
    CREATOR_MAPPING[dtype_info.torch_dtype] = dtype_info.creator


def get_torch_dtype(dtype: str) -> torch.dtype:
    return TORCH_DTYPE_MAPPING[dtype].torch_dtype

def get_torch_dtype_size(dtype: torch.dtype) -> int | float:
    return DTYPE_SIZE_MAPPING[dtype]

def default_creator(size, dtype, device):
    return CREATOR_MAPPING[dtype](size, dtype, device)




@dataclass
class OpTensorInfo:
    shape: List[int] = field(default_factory=list)
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    creator: callable = default_creator


def create_from_list(size, dtype, device, data):
    full_tensor = torch.zeros(size, dtype=dtype, device=device)
    data_len = len(data)
    full_tensor.view(-1)[:data_len] = torch.tensor(data, dtype=dtype, device=device)
    return full_tensor
    



def calc_tensor_element_count(tensor_info: OpTensorInfo):
    element_count = 1
    for dim in tensor_info.shape:
        element_count *= dim
    return element_count


def calc_tensor_size(tensor_info: OpTensorInfo):
    element_count = 1
    for dim in tensor_info.shape:
        element_count *= dim
    dtype_size = get_torch_dtype_size(tensor_info.dtype)
    return element_count * dtype_size


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y




class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def _parse_cpulist(cpulist_str):
    cores = []
    for part in cpulist_str.split(","):
        bounds = part.split("-")
        if len(bounds) == 2:
            cores.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            cores.append(int(bounds[0]))
    return sorted(cores)


def _cores_to_range_str(cores):
    if not cores:
        return ""
    ranges = []
    start = prev = cores[0]
    for c in cores[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def get_numa_info():
    numa_nodes = {}

    if "CUSTOM_NUMA_CONFIG" in os.environ:
        numa_config_str = os.environ["CUSTOM_NUMA_CONFIG"]
        for i, sub_numa_str in enumerate(numa_config_str.split(";")):
            numa_nodes[i] = sub_numa_str
    else:
        numa_root = "/sys/devices/system/node/"
        if not os.path.exists(numa_root):
            raise FileNotFoundError(f"未找到 NUMA 目录 {numa_root}，系统可能不支持 NUMA 或未启用")
    
        for dirname in os.listdir(numa_root):
            if re.match(r'^node\d+$', dirname):
                node_id = dirname.lstrip('node')
                cpulist_path = os.path.join(numa_root, dirname, "cpulist")
                try:
                    with open(cpulist_path, 'r') as f:
                        cpu_list = f.read().strip()
                        numa_nodes[int(node_id)] = cpu_list
                except Exception as e:
                    print(f"警告：读取节点 {node_id} 的 CPU 列表失败：{e}")
                    continue
    sorted_nodes = dict(sorted(numa_nodes.items()))

    online_cpus = os.sched_getaffinity(0)

    final_numa_configs = []
    for node_config_str in sorted_nodes.values():
        core_list = _parse_cpulist(node_config_str)
        core_list = sorted(c for c in core_list if c in online_cpus)
        final_numa_configs.append(core_list)

    if final_numa_configs:
        min_cores = min(len(cores) for cores in final_numa_configs)
        final_numa_configs = [cores[:min_cores] for cores in final_numa_configs]

    rebuilt_nodes = {}
    for i, cores in enumerate(final_numa_configs):
        rebuilt_nodes[i] = _cores_to_range_str(cores)

    return rebuilt_nodes, final_numa_configs




def get_moe_tokens_info(
    num_tokens, num_experts, topk, 
    ep_size=1, ep_rank=0
):
    # split tokens / experts
    num_scatter_tokens = num_tokens * topk
    num_scatter_tokens_per_rank = num_scatter_tokens // ep_size
    num_experts_per_rank = num_experts // ep_size

    experts_start_idx = ep_rank * num_experts_per_rank
    experts_end_idx = experts_start_idx + num_experts_per_rank



    experts_idx_for_each_rank = []
    for rank_idx in range(ep_size):
        start_idx = rank_idx * num_experts_per_rank
        end_idx = start_idx + num_experts_per_rank
        experts_idx_for_each_rank.append(list(range(start_idx, end_idx)))
    transpose_experts = [list(row) for row in zip(*experts_idx_for_each_rank)]
    experts_array = [num for row in transpose_experts for num in row]


    # for each input token, choose topk experts and corresponding weights
    all_select_experts = []
    all_select_weights = []

    cur_expert = 0
    for token_idx in range(num_tokens):
        cur_token_selections = []
        for topk_idx in range(topk):
            cur_token_selections.append(experts_array[cur_expert])
            cur_expert += 1
            if cur_expert >= num_experts:
                cur_expert = 0
        all_select_experts.append(cur_token_selections)
        all_select_weights.append([1 / topk for _ in range(topk)])


    # 当前rank上，每一个input_token对应的dispatch到当前rank对应的experts
    cur_rank_tokens = {}
    cur_rank_weights = {}
    dispatch_tokens = 0

    # for each ep_rank, find corresponding tokens
    for token_idx in range(num_tokens):
        cur_token_dispatch_experts = []
        cur_token_dispatch_weights = []
        for expert_idx, expert_weight in zip(all_select_experts[token_idx], all_select_weights[token_idx]):
            if expert_idx >= experts_start_idx and expert_idx < experts_end_idx:
                cur_token_dispatch_experts.append(expert_idx)
                cur_token_dispatch_weights.append(expert_weight)
        if cur_token_dispatch_experts:
            cur_rank_tokens[token_idx] = cur_token_dispatch_experts
            cur_rank_weights[token_idx] = cur_token_dispatch_weights
            dispatch_tokens += len(cur_token_dispatch_experts)

    
    used_src_tokens = len(cur_rank_tokens)


    expert_dispatch_tokens = [[] for _ in range(experts_start_idx, experts_end_idx)]
    expert_dispatch_weights = [[] for _ in range(experts_start_idx, experts_end_idx)]
    expert_dispatch_token_count = [0 for _ in range(experts_start_idx, experts_end_idx)]
    expert_dispatch_token_offset = [0 for _ in range(experts_start_idx, experts_end_idx)]

    for token_idx in cur_rank_tokens:
        for expert_idx, weight in zip(cur_rank_tokens[token_idx], cur_rank_weights[token_idx]):
            expert_dispatch_tokens[expert_idx - experts_start_idx].append(token_idx)
            expert_dispatch_weights[expert_idx - experts_start_idx].append(weight)
            expert_dispatch_token_count[expert_idx - experts_start_idx] += 1
    expert_dispatch_token_offset = ([0] + list(itertools.accumulate(expert_dispatch_token_count)))[:num_experts_per_rank]

    
    scatter_token_id = []
    scatter_token_weight = []
    for expert_idx, tokens in enumerate(expert_dispatch_tokens):
        weights = expert_dispatch_weights[expert_idx]
        for target_token, target_weight in zip(tokens, weights):
            scatter_token_id.append(target_token)
            scatter_token_weight.append(target_weight)

    return (
        num_scatter_tokens, 
        num_scatter_tokens_per_rank, 
        num_experts_per_rank, 
        experts_start_idx, 
        experts_end_idx, 

        all_select_experts, 
        all_select_weights, 
        dispatch_tokens, 
        used_src_tokens, 
        expert_dispatch_tokens, 
        expert_dispatch_weights, 
        scatter_token_id, 
        scatter_token_weight, 
        expert_dispatch_token_count, 
        expert_dispatch_token_offset
    )



def get_attn_info(arg_type, attn_mode, args_dict, op_cls=None):
    if arg_type == "llm":
        if attn_mode == "prefill":
            batch_size = args_dict.get("batch_size", 1)
            cache_len = args_dict["cache_len"]
            q_len = args_dict["q_len"]

        elif attn_mode == "decode":
            batch_size = args_dict["batch_size"]
            cache_len = args_dict["cache_len"]
            q_len = args_dict.get("q_len", 1)

        cache_lens = [cache_len] * batch_size
        q_lens = [q_len] * batch_size
        kv_lens = [cache_len + q_len for cache_len, q_len in zip(cache_lens, q_lens)]

    elif arg_type == "batch_llm":
        q_lens = args_dict["q_lens"]
        cache_lens = args_dict["cache_lens"]
        kv_lens = [cache_len + q_len for cache_len, q_len in zip(cache_lens, q_lens)]

    else:
        raise ValueError(f"Unsupported arg_type: {arg_type}")

    batch_size = len(q_lens)

    max_cache_len = max(cache_lens)
    max_q_len = max(q_lens)
    max_kv_len = max(kv_lens)

    accum_cache_lens = [0] + list(itertools.accumulate(cache_lens))
    accum_q_lens = [0] + list(itertools.accumulate(q_lens))
    accum_kv_lens = [0] + list(itertools.accumulate(kv_lens))

    num_cache_tokens = accum_cache_lens[-1]
    num_tokens = accum_q_lens[-1]
    num_kv_tokens = accum_kv_lens[-1]


    block_size = args_dict.get("block_size", 0)
    if block_size == 0:
        cache_type = "linear"

        if "slot_mapping" in args_dict:
            slot_mapping = args_dict["slot_mapping"]
        else:
            slot_mapping = list(range(batch_size))

    elif block_size > 0:
        cache_type = "paged"
        
        # 期望的每个序列的最大的 block 数
        max_block_num_per_seq = (max_kv_len + block_size - 1) // block_size

        # 期望的有效的 q_blocks/cache_blocks/kv_blocks
        cache_blocks = [(cache_len + (block_size - 1)) // block_size for cache_len in cache_lens]
        q_blocks = [(q_len + (block_size - 1)) // block_size for q_len in q_lens]
        kv_blocks = [(kv_len + (block_size - 1)) // block_size for kv_len in kv_lens]

        num_cache_blocks = sum(cache_blocks)
        num_q_blocks = sum(q_blocks)
        num_kv_blocks = sum(kv_blocks)

        if "block_table" in args_dict:
            block_table = args_dict["block_table"]
            
            target_batch_size = len(block_table)
            target_per_seq_num_block = max([len(seq_blocks) for seq_blocks in block_table])
            if target_batch_size < batch_size or target_per_seq_num_block < max_block_num_per_seq:
                raise ValueError(f"block_table 中 batch_size 为 {target_batch_size}，而期望的 batch_size 为 {batch_size}；"
                                 f"每个序列的最大 block 数为 {target_per_seq_num_block}，而期望的最大 block 数为 {max_block_num_per_seq}")

            for seq_blocks in block_table:
                if len(seq_blocks) < target_per_seq_num_block:
                    seq_blocks.extend([-1] * (target_per_seq_num_block - len(seq_blocks)))

        else:
            block_table = []
            block_idx = 0
            for batch_idx in range(batch_size):
                seq_blocks = [-1] * max_block_num_per_seq
                for seq_block_id in range(kv_blocks[batch_idx]):
                    seq_blocks[seq_block_id] = block_idx
                    block_idx += 1
                block_table.append(seq_blocks)

            target_batch_size = batch_size
            target_per_seq_num_block = max_block_num_per_seq

        total_cache_blocks = target_batch_size * target_per_seq_num_block
    else:
        raise ValueError
    

    if op_cls is not None:
        op_cls.cache_type = cache_type

        op_cls.batch_size = batch_size
        op_cls.q_lens = q_lens
        op_cls.cache_lens = cache_lens
        op_cls.kv_lens = kv_lens

        op_cls.max_q_len = max_q_len
        op_cls.max_cache_len = max_cache_len
        op_cls.max_kv_len = max_kv_len

        op_cls.accum_q_lens = accum_q_lens
        op_cls.accum_cache_lens = accum_cache_lens
        op_cls.accum_kv_lens = accum_kv_lens

        op_cls.num_tokens = num_tokens
        op_cls.num_cache_tokens = num_cache_tokens
        op_cls.num_kv_tokens = num_kv_tokens

        if cache_type == "linear":
            op_cls.slot_mapping = slot_mapping

        elif cache_type == "paged":
            op_cls.block_size = block_size
            op_cls.q_blocks = q_blocks
            op_cls.cache_blocks = cache_blocks
            op_cls.kv_blocks = kv_blocks
            op_cls.num_q_blocks = num_q_blocks
            op_cls.num_cache_blocks = num_cache_blocks
            op_cls.num_kv_blocks = num_kv_blocks

            # [target_batch_size, target_per_seq_num_block]
            op_cls.block_table = block_table
            op_cls.target_batch_size = target_batch_size
            op_cls.target_per_seq_num_block = target_per_seq_num_block
            op_cls.total_cache_blocks = total_cache_blocks

            

    return_dict = {
        "cache_type": cache_type, 
        
        "batch_size": batch_size,
        "q_lens": q_lens,
        "cache_lens": cache_lens,
        "kv_lens": kv_lens,

        "max_q_len": max_q_len,
        "max_cache_len": max_cache_len,
        "max_kv_len": max_kv_len,

        "accum_q_lens": accum_q_lens,
        "accum_cache_lens": accum_cache_lens,
        "accum_kv_lens": accum_kv_lens,

        "num_tokens": num_tokens,
        "num_cache_tokens": num_cache_tokens,
        "num_kv_tokens": num_kv_tokens,
    }

    if cache_type == "linear":
        return_dict.update({
            "slot_mapping": slot_mapping,
        })

    elif cache_type == "paged":
        return_dict.update({
            "block_size": block_size,
            "q_blocks": q_blocks,
            "cache_blocks": cache_blocks,
            "kv_blocks": kv_blocks,
            "num_q_blocks": num_q_blocks,
            "num_cache_blocks": num_cache_blocks,
            "num_kv_blocks": num_kv_blocks,

            "block_table": block_table,
            "target_batch_size": target_batch_size,
            "target_per_seq_num_block": target_per_seq_num_block,
            "total_cache_blocks": total_cache_blocks,
        })

    return return_dict






def precompute_freqs_cis(
    max_seq_len, dim, 
    theta: float = 10000.0
):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2).float() / dim)
    )
    t = torch.arange(max_seq_len, device=inv_freq.device).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=1)
    cos = freqs.cos().bfloat16()
    sin = freqs.sin().bfloat16()
    return cos, sin



def rotate(qk, cos, sin):
    # [q_len, q_head_num + kv_head_num, rope_dim]
    rope_dim = qk.size(-1)
    left_part = qk[:, :, :rope_dim//2]
    right_part = qk[:, :, rope_dim//2:]
    output = torch.cat([left_part, right_part], dim=-1) * cos.unsqueeze(1) + \
             torch.cat([-right_part, left_part], dim=-1) * sin.unsqueeze(1)
    return output



def smooth_per_token_dynamic_quant(
    hidden_states : torch.Tensor, 
    smooth_scale : torch.Tensor, 
    dst_torch_dtype=torch.int8
):
    max_dtype_val = 1.0
    if dst_torch_dtype == torch.int8:
        max_dtype_val = 127.0
    elif dst_torch_dtype == torch.float8_e4m3fn:
        max_dtype_val = 448.0
    else:
        raise ValueError(f"dst_torch_dtype {dst_torch_dtype} is not supported")

    # [num_tokens, hidden_size]
    ori_shape = hidden_states.shape
    hidden_states = hidden_states.contiguous().view(ori_shape[0], -1).to(torch.float32)

    # [1, hidden_size]
    smooth_scale = smooth_scale.contiguous().view(1, -1)

    # [num_tokens, hidden_size]
    smoothed_input = torch.mul(hidden_states, smooth_scale)

    # [num_tokens, 1], 1 / max
    per_token_max = torch.max(smoothed_input.abs(), -1, keepdim=True)[0].reciprocal()

    # [num_tokens, 1], max_dtype_val / max
    per_token_scale = per_token_max * max_dtype_val

    # [num_tokens, hidden_size], quantized
    quant_tokens_fp32 = torch.mul(smoothed_input, per_token_scale).clamp(-max_dtype_val, max_dtype_val)
    if dst_torch_dtype == torch.int8:
        quant_tokens_fp32 = quant_tokens_fp32.round()

    # float32 --> int8 / float8
    quant_tokens = quant_tokens_fp32.type(dst_torch_dtype).view(ori_shape)

    # max_dtype_val / max --> max / max_dtype_val
    per_token_scale = per_token_scale.reciprocal().view(ori_shape[0])

    return quant_tokens, per_token_scale


def static_quant(
    hidden_states : torch.Tensor, 
    quant_scale : torch.Tensor, 
    dst_torch_dtype=torch.int8
):
    max_dtype_val = 1.0
    if dst_torch_dtype == torch.int8:
        max_dtype_val = 127.0
    elif dst_torch_dtype == torch.float8_e4m3fn:
        max_dtype_val = 448.0
    else:
        raise ValueError(f"dst_torch_dtype {dst_torch_dtype} is not supported")

    # [num_tokens, hidden_size]
    ori_shape = hidden_states.shape
    hidden_states = hidden_states.contiguous().view(ori_shape[0], -1).to(torch.float32)

    # [1, hidden_size]
    quant_scale = quant_scale.contiguous().view(1, -1)

    # [num_tokens, hidden_size], quantized
    quant_tokens_fp32 = torch.mul(hidden_states, quant_scale).clamp(-max_dtype_val, max_dtype_val)
    if dst_torch_dtype == torch.int8:
        quant_tokens_fp32 = quant_tokens_fp32.round()

    quant_tokens = quant_tokens_fp32.type(dst_torch_dtype).view(ori_shape)

    return quant_tokens

    
"""
使用 bfloat16 的 GEMM 来模拟 int8/float8 量化的 GEMM
厂商需要实现自己的 GEMM 算子
"""
def fake_quant_gemm(
    tokens, per_token_scale, 
    weights, weight_scale, 
    dst_torch_dtype=torch.bfloat16, 
    trans_w=False,
):
    if trans_w:
        weights = weights.transpose(0, 1)
    fake_gemm_output = torch.matmul(
        tokens.to(torch.bfloat16), 
        weights.to(torch.bfloat16)
    ).to(torch.float32)
    fake_gemm_output = torch.mul(
        fake_gemm_output, 
        per_token_scale.unsqueeze(-1)
    )
    fake_gemm_output = torch.mul(
        fake_gemm_output, 
        weight_scale.unsqueeze(0)
    )
    return fake_gemm_output.to(dst_torch_dtype)
    