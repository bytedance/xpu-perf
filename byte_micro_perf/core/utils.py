import os
import re
import sys
import csv
import json
import random
import pathlib
import logging
import itertools
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import namedtuple

import torch

# logger functions
logger = logging.getLogger("bytemlperf_micro_perf")
def setup_logger(loglevel: str):
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel.upper())
    logger.propagate = False



TORCH_DTYPE_MAPPING = {
    "float32": torch.float32,
    "float": torch.float32,

    "float16": torch.float16,
    "half": torch.float16,

    "bfloat16": torch.bfloat16,

    "float8": torch.float8_e4m3fn,

    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,

    # 使用int8表示两个int4
    "int4": torch.int8
}




def get_torch_dtype(dtype: str) -> torch.dtype:
    return TORCH_DTYPE_MAPPING[dtype]






def default_creator(size, dtype, device):
    if dtype in [
        torch.float64, 
        torch.float32, 
        torch.float16, 
        torch.bfloat16, 
        torch.float8_e4m3fn, 
        torch.float8_e5m2
    ]:
        return torch.randn(
            size=size, 
            dtype=dtype, 
            device=device
        )
    elif dtype in [
        torch.int64, 
        torch.int32, 
        torch.int16, 
        torch.int8
    ]:
        return torch.randint(
            low=-16, 
            high=17, 
            size=size, 
            dtype=torch.int32, 
            device=device
        ).to(dtype=dtype)
    elif dtype in [
        torch.uint64, 
        torch.uint32, 
        torch.uint16, 
        torch.uint8
    ]:
        return torch.randnint(
            low=0, 
            high=17, 
            size=size, 
            dtype=dtype, 
            device=device
        )
    else:
        raise NotImplementedError




# shape: list or tuple
# dtype: torch.dtype
# device: str
# creator: func, default is torch.zeros
OpTensorInfo = namedtuple(
    "OpTensorInfo", 
    ["shape", "dtype", "device", "creator"], 
    defaults=[torch.float32, "cpu", default_creator]
)




def calc_tensor_size(tensor_info: OpTensorInfo):
    tensor_size = 1
    for dim in tensor_info.shape:
        tensor_size *= dim
    dtype_size = torch.tensor([], dtype=tensor_info.dtype).element_size()
    tensor_size *= dtype_size
    return tensor_size


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


"""
task_name,arg1,arg2,arg3
name1, value1, value2, value3
"""
def parse_csv_file(csv_file : pathlib.Path):
    task_dict : Dict[str, List[Dict[str, Any]]] = {}

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        
        # 默认有一列指定 task_name
        if "task_name" not in reader.fieldnames:
            logger.error(f"CSV file {csv_file} must contain task_name column")
            sys.exit(1)

        for data in reader:
            task_name = data.pop("task_name")

            if task_name not in task_dict:
                task_dict[task_name] = []


            task_dict[task_name].append(data)

    return task_dict


"""
{task_name}.json
{
    "cases": [
        {
            "arg1": ["value1"],
            "arg2": ["value2"],
            "arg3": ["value3"],
        }
    ]
}
--> task_name, arg1, arg2, arg3
--> name1, value1, value2, value3


{any_name}.json
{
    "name1": [
        {
            "arg1": ["value1"],
            "arg2": ["value2"],
            "arg3": ["value3"],
        }
    ], 
    "name2": [
        {
            "arg1": ["value1"],
            "arg2": ["value2"],
            "arg3": ["value3"],
        }
    ]
}
"""

def get_cartesian_product(test_cases: List[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
    if not test_cases:
        return []

    return_datas = []
    for argument_case in test_cases:
        if "batch" in argument_case["arg_type"]:
            return_datas.append(argument_case)
            continue
            
        keys = list(argument_case.keys())
        values = list(argument_case.values())
        for i, value in enumerate(values):
            if not isinstance(value, list):
                values[i] = [value]
        total_cases = list(itertools.product(*values))
        for case in total_cases:
            case_dict = dict(zip(keys, case))

            added_key_value = {}
            removed_key = []
            for key in case_dict:
                if "." in key:
                    split_keys = key.split(".")
                    for i, split_key in enumerate(split_keys):
                        added_key_value[split_key] = case_dict[key][i]
                    removed_key.append(key)
            for key in removed_key:
                del case_dict[key]
            case_dict.update(added_key_value)
            return_datas.append(case_dict)
    return return_datas

    




def parse_json_file(json_file : pathlib.Path):
    task_dict : Dict[str, List[Dict[str, Any]]] = {}

    parsed_data = json.loads(json_file.read_text())

    if "cases" in parsed_data:
        task_name = json_file.stem
        task_dict[task_name] = get_cartesian_product(parsed_data["cases"])
    else:
        for task_name, test_cases in parsed_data.items():
            task_dict[task_name] = get_cartesian_product(test_cases)

    return task_dict
            


def get_numa_info():
    numa_nodes = {}

    if "CUSTOM_NUMA_CONFIG" in os.environ:
        # 0-15;80-95;160-175;240-255
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


    final_numa_configs = []
    for node_config_str in sorted_nodes.values():
        core_list = []
        for sub_node_config_str in node_config_str.split(","):
            split_cores = sub_node_config_str.split("-")
            core_list.extend(range(int(split_cores[0]), int(split_cores[1])+1))
        final_numa_configs.append(core_list)

    return sorted_nodes, final_numa_configs




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
        for topk_idx, expert_idx in enumerate(cur_rank_tokens[token_idx]):
            expert_dispatch_tokens[expert_idx - experts_start_idx].append(token_idx)
            expert_dispatch_weights[expert_idx - experts_start_idx].append(cur_rank_weights[token_idx][topk_idx])
            expert_dispatch_token_count[expert_idx - experts_start_idx] += 1
    expert_dispatch_token_offset = ([0] + list(itertools.accumulate(expert_dispatch_token_count)))[:num_experts_per_rank]

    
    expert_dispatch_tokens_flatten = [token for tokens in expert_dispatch_tokens for token in tokens]
    expert_dispatch_weights_flatten = [weight for weights in expert_dispatch_weights for weight in weights]


    # for expert_idx, (tokens, weights, num_tokens, token_offset) in enumerate(zip(expert_dispatch_tokens, expert_dispatch_weights, expert_dispatch_token_count, expert_dispatch_token_offset)):
    #     print(f"expert {expert_idx + experts_start_idx} dispatch tokens: {tokens}")
    #     print(f"expert {expert_idx + experts_start_idx} dispatch weights: {weights}")
    #     print(f"expert {expert_idx + experts_start_idx} dispatch token num: {num_tokens}")
    #     print(f"expert {expert_idx + experts_start_idx} dispatch token offset: {token_offset}")
    #     print("")

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
        expert_dispatch_tokens_flatten, 
        expert_dispatch_weights_flatten, 
        expert_dispatch_token_count, 
        expert_dispatch_token_offset
    )



def get_attn_info(arg_type, attn_mode, args_dict, op_cls=None):
    if arg_type == "llm":
        if attn_mode == "prefill":
            batch_size = args_dict.get("batch_size", 1)
            q_len = args_dict["q_len"]
            cache_len = args_dict["cache_len"]
        elif attn_mode == "decode":
            batch_size = args_dict["batch_size"]
            q_len = args_dict.get("q_len", 1)
            cache_len = args_dict["cache_len"]

        random_seed = args_dict.get("random_seed", 42)
        random.seed(random_seed)

        unbalance_q = args_dict.get("unbalance_q", 0)
        unbalance_cache = args_dict.get("unbalance_cache", 0)

        if not (unbalance_q >= 0 and unbalance_q <= 50):
            raise ValueError("unbalance_q must be in [0, 50]")
        
        if not (unbalance_cache >= 0 and unbalance_cache <= 50):
            raise ValueError("unbalance_cache must be in [0, 50]")
        
        q_lens = [q_len + int(q_len * (random.randint(-unbalance_q, unbalance_q) / 100)) for i in range(batch_size)]
        cache_lens = [cache_len + int(cache_len * (random.randint(-unbalance_cache, unbalance_cache) / 100)) for i in range(batch_size)]
        kv_lens = [q_len + cache_len for q_len, cache_len in zip(q_lens, cache_lens)]


    elif arg_type == "batch_llm":
        q_lens = args_dict["q_lens"]
        cache_lens = args_dict["cache_lens"]
        kv_lens = [q_len + cache_len for q_len, cache_len in zip(q_lens, cache_lens)]

    else:
        raise ValueError(f"Unsupported arg_type: {arg_type}")

    batch_size = len(q_lens)
    if "slot_mapping" in args_dict:
        slot_mapping = args_dict["slot_mapping"]
    else:
        slot_mapping = list(range(batch_size))

    max_q_len = max(q_lens)
    max_cache_len = max(cache_lens)
    max_kv_len = max(kv_lens)

    accum_q_lens = [0] + list(itertools.accumulate(q_lens))
    accum_cache_lens = [0] + list(itertools.accumulate(cache_lens))
    accum_kv_lens = [0] + list(itertools.accumulate(kv_lens))

    num_tokens = accum_q_lens[-1]
    num_cache_tokens = accum_cache_lens[-1]
    num_kv_tokens = accum_kv_lens[-1]


    block_size = args_dict.get("block_size", 0)
    if block_size == 0:
        cache_type = "linear"
    elif block_size > 0:
        cache_type = "paged"
        
        max_block_num_per_seq = (max_kv_len + block_size - 1) // block_size
        q_blocks = [(q_len + (block_size - 1)) // block_size for q_len in q_lens]
        cache_blocks = [(cache_len + (block_size - 1)) // block_size for cache_len in cache_lens]
        kv_blocks = [(kv_len + (block_size - 1)) // block_size for kv_len in kv_lens]
        
        num_q_blocks = sum(q_blocks)
        num_cache_blocks = sum(cache_blocks)
        num_kv_blocks = sum(kv_blocks)


        if "block_table" in args_dict:
            block_table = args_dict["block_table"]
        else:
            block_table = []
            block_idx = 0
            for batch_idx in range(batch_size):
                seq_blocks = [-1] * max_block_num_per_seq
                for seq_block_id in range(kv_blocks[batch_idx]):
                    seq_blocks[seq_block_id] = block_idx
                    block_idx += 1
                block_table.append(seq_blocks)
    else:
        raise ValueError
    

    if op_cls is not None:
        op_cls.cache_type = cache_type
        op_cls.slot_mapping = slot_mapping

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

        if cache_type == "paged":
            op_cls.block_size = block_size
            op_cls.max_block_num_per_seq = max_block_num_per_seq
            op_cls.q_blocks = q_blocks
            op_cls.cache_blocks = cache_blocks
            op_cls.kv_blocks = kv_blocks

            op_cls.num_q_blocks = num_q_blocks
            op_cls.num_cache_blocks = num_cache_blocks
            op_cls.num_kv_blocks = num_kv_blocks

            op_cls.block_table = block_table

    return_dict = {
        "cache_type": cache_type, 
        "slot_mapping": slot_mapping, 
        
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

    if cache_type == "paged":
        return_dict.update({
            "block_size": block_size,
            "max_block_num_per_seq": max_block_num_per_seq,
            "q_blocks": q_blocks,
            "cache_blocks": cache_blocks,
            "kv_blocks": kv_blocks,

            "num_q_blocks": num_q_blocks,
            "num_cache_blocks": num_cache_blocks,
            "num_kv_blocks": num_kv_blocks,

            "block_table": block_table,
        })

    return return_dict





def precompute_freqs_cis(dim, max_seq_len, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.real(freqs_cis), torch.imag(freqs_cis)




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
    quant_tokens_fp32 = torch.mul(smooth_scale, per_token_scale).clamp(-max_dtype_val, max_dtype_val)
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



if __name__ == "__main__":

    get_attn_info(
        "llm", "prefill", 
        {
            "batch_size": 16, 
            "q_len": 1024,
            "cache_len": 1024, 
            "unbalance_q": 10,
            "unbalance_cache": 10,
            "block_size": 128
        }
    )


    # get_moe_tokens_info(
    #     num_tokens=128,
    #     num_experts=128,
    #     topk=6,
    #     ep_size=8
    # )











