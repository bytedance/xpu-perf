



def num_tokens_set_template(
    workload, bench_info
):
    workload["arg_type"] = "llm"

    if {"cache_lens", "q_lens"}.issubset(bench_info.keys()):
        q_lens = bench_info["q_lens"]
        num_tokens = sum(q_lens)

    elif {"batch_size", "cache_len", "q_len"}.issubset(bench_info.keys()):
        batch_size = bench_info["batch_size"]
        q_len = bench_info["q_len"]
        num_tokens = batch_size * q_len

    else:
        raise ValueError("bench_info must contain either cache_lens, q_lens or batch_size, cache_len, q_len")

    workload["num_tokens"] = num_tokens





def mode_bs_cache_q_set_template(workload, bench_info):
    workload["attn_mode"] = bench_info["run_mode"]
    
    if {"cache_lens", "q_lens"}.issubset(bench_info.keys()):
        cache_lens = bench_info["cache_lens"]
        q_lens = bench_info["q_lens"]

        workload["arg_type"] = "batch_llm"
        workload["cache_lens"] = cache_lens
        workload["q_lens"] = q_lens
        
    elif {"batch_size", "cache_len", "q_len"}.issubset(bench_info.keys()):
        batch_size = bench_info["batch_size"]
        cache_len = bench_info["cache_len"]
        q_len = bench_info["q_len"]

        workload["arg_type"] = "llm"
        workload["batch_size"] = batch_size
        workload["cache_len"] = cache_len
        workload["q_len"] = q_len

    workload["block_size"] = bench_info["block_size"]
    if workload["block_size"] > 0:
        if "slot_mapping" in workload:
            workload.pop("slot_mapping")
        workload["block_table"] = bench_info["block_table"]
    else:
        if "block_table" in workload:
            workload.pop("block_table")
        workload["slot_mapping"] = bench_info["slot_mapping"]




OP_ZOO = {
    "gemm": num_tokens_set_template, 

    "add_rms_norm_dynamic_quant": num_tokens_set_template, 
    "add_rms_norm": num_tokens_set_template, 
    "scale_dynamic_quant": num_tokens_set_template, 
    
    "moe_softmax_topk": num_tokens_set_template, 
    "moe_scatter_dynamic_quant": num_tokens_set_template, 
    
    "moe_gather": num_tokens_set_template, 

    "qk_rms_norm": num_tokens_set_template, 
    "head_rms_norm": num_tokens_set_template, 
    "head_rms_norm_dynamic_quant": num_tokens_set_template, 


    "swiglu": num_tokens_set_template, 
    "swiglu_dynamic_quant": num_tokens_set_template, 
    "moe_swiglu": num_tokens_set_template, 
    "moe_swiglu_dynamic_quant": num_tokens_set_template, 

    "rotary_embedding": mode_bs_cache_q_set_template, 
    "store_kv_cache": mode_bs_cache_q_set_template, 
    "flash_attention": mode_bs_cache_q_set_template, 

    "moe_gating_gemm": num_tokens_set_template, 
    "quant_matmul": num_tokens_set_template, 
    "moe_quant_group_gemm": num_tokens_set_template, 
    "moe_quant_group_gemm_combine": num_tokens_set_template, 
    "quant_group_gemm_reduce_sum": num_tokens_set_template, 

    "all_reduce": num_tokens_set_template, 
    "reduce_scatter": num_tokens_set_template, 
    "all_gather": num_tokens_set_template, 
    "all_to_all": num_tokens_set_template, 
}




# from dataclasses import dataclass, field
# from typing import List
# from enum import Enum, auto


# # 用于标记 gemm / group_gemm 的计算类型
# # 对应芯片的不同规格算力


# """
# 标记 gemm/group_gemm/attn 的计算类型
# """
# class GemmComputeDType(Enum):
#     FP32 = auto()
#     BF16 = auto()
#     FP16 = auto()
#     W4A16 = auto()
#     W8A8 = auto()
#     W4A8 = auto()
#     W4A4 = auto()

# class GemmDstDtype(Enum):
#     FP32 = auto()
#     BF16 = auto()
#     FP16 = auto()


    
# """
# 量化粒度

# NONE:           不量化
# PER_TENSOR:     整个tensor共用一个scale
# PER_TOKEN:      每个token共用一个scale, 一般用于激活值
# PER_CHANNEL:    每个output_channel共用一个scale, 一般用于权重值
# PER_HEAD_PER_DIM: 每个head的每个dim共用一个scale, 一般用于KV
# PER_BLOCK:      用于统一描述per_block和per_group
# PER_BLOCK_PER_TENSOR:    用于描述类似于NVFP4这种二级scale的方式
# """
# class QuantGranularity(Enum):
#     NONE = auto()
#     PER_TENSOR = auto()
#     PER_TOKEN = auto()
#     PER_CHANNEL = auto()
#     PER_HEAD_PER_DIM = auto()
#     PER_BLOCK = auto()
#     PER_BLOCK_PER_TENSOR = auto()



# """
# 量化模式

# OFFLINE:    静态量化参数, 离线量化
# STATIC:     静态量化参数, 在线量化
# DYNAMIC:    动态量化参数, 在线量化
# """
# class QuantMode(Enum):
#     STATIC = auto()
#     DYNAMIC = auto()


# class QuantScaleDType(Enum):
#     FP32 = auto()
#     TF32 = auto()
#     FP16 = auto()
#     BF16 = auto()
#     FP8_E4M3 = auto()
#     E8M0 = auto()



# @dataclass
# class QuantInfo:
#     granularity : QuantGranularity = QuantGranularity.NONE
#     mode : QuantMode = QuantMode.STATIC
#     scale_dtype : QuantScaleDType = QuantScaleDType.FP32
#     aux_scale_dtype : QuantScaleDType = QuantScaleDType.FP32
#     scale_dim_0 : int = 32
#     scale_dim_1 : int = 32



# @dataclass
# class GemmQuantConfig:
#     compute_dtype: GemmComputeDType = GemmComputeDType.BF16
#     dst_dtype: GemmDstDtype = GemmDstDtype.BF16

#     w_quant_info : QuantInfo = QuantInfo()
#     a_quant_info : QuantInfo = QuantInfo()

#     @classmethod
#     def from_dict(
#         cls, 
#         config_dict, 
#         dst_dtype
#     ):

#         compute_dtype = config_dict.get("compute_dtype", GemmComputeDType.BF16)




# @dataclass
# class AttnQuantConfig:
#     compute_dtype: GemmComputeDType = GemmComputeDType.BF16
#     dst_dtype: GemmDstDtype = GemmDstDtype.BF16

#     q_quant_info : QuantInfo = QuantInfo()
#     k_quant_info : QuantInfo = QuantInfo()
#     p_quant_info : QuantInfo = QuantInfo()
#     v_quant_info : QuantInfo = QuantInfo()


