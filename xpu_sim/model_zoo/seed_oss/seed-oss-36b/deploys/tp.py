import os
import sys
import pathlib
from typing import Dict
from transformers import SeedOssConfig

# model_type / model_name / deploy / deploy_0.py
DEPLOY_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_NAME_DIR = DEPLOY_DIR.parent
MODEL_TYPE_DIR = MODEL_NAME_DIR.parent
MODEL_ZOO_DIR = MODEL_TYPE_DIR.parent

sys.path.insert(0, str(MODEL_ZOO_DIR.parent))

from model_zoo import OpTopologyDAG, DistributionInfo

"""
tp 模式
- tp_size devices
- w8a8 gemm 
- bf16_c8 flash_attention
"""
def generate(
    model_config: SeedOssConfig, 
    bench_config: Dict
):
    # parse model params
    hidden_size = model_config.hidden_size
    q_head_num = model_config.num_attention_heads
    kv_head_num = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    attention_bias = model_config.attention_bias
    attention_out_bias = model_config.attention_out_bias
    mlp_bias = model_config.mlp_bias

    
    # parse distribution info
    dist_info = DistributionInfo.from_bench_config(bench_config["parallel_config"])

    split_q_head_num = q_head_num // dist_info.tp_size \
        if q_head_num >= dist_info.tp_size \
        else 1
    split_kv_head_num = kv_head_num // dist_info.tp_size \
        if kv_head_num >= dist_info.tp_size \
        else 1


    # 获取默认数据类型
    default_dtype = bench_config.get("dtype_config", {}).get("default_dtype", "bfloat16")

    qkvo_config = bench_config["dtype_config"]["qkvo"]
    attn_config = bench_config["dtype_config"]["attn"]
    mlp_config = bench_config["dtype_config"]["mlp"]


    model_topo = OpTopologyDAG()

    model_topo.op_process_wrapper(
        "add_rms_norm_dynamic_quant", "add_rms_norm_0", 
        {
            "dtype": default_dtype, 
            "dst_dtype": qkvo_config["dtype"], 
            "hidden_size": hidden_size, 
            "add_residual": True, 
            "output_mode": "res"
        }
    )

    model_topo.op_process_wrapper(
        "quant_matmul", "qkv_gemm", 
        {
            "dtype": qkvo_config["dtype"], 
            "w_dtype": qkvo_config["w_dtype"], 
            "compute_dtype": qkvo_config["compute_dtype"], 
            "dst_dtype": default_dtype, 
            "has_bias": attention_bias, 
            "hidden_size": hidden_size, 
            "new_hidden_size": (split_q_head_num + 2 * split_kv_head_num) * head_dim
        }
    )

    model_topo.op_process_wrapper(
        "rotary_embedding", "rotary_embedding", 
        {
            "dtype": default_dtype, 
            "q_head_num": split_q_head_num, 
            "kv_head_num": split_kv_head_num, 
            "head_dim": head_dim, 
            "rope_offset": 0, 
            "rope_dim": head_dim, 
        }
    )

    model_topo.op_process_wrapper(
        "store_kv_cache", "store_kv_cache", 
        {
            "dtype": default_dtype, 
            "cache_dtype": attn_config["cache_dtype"], 
            "q_head_num": split_q_head_num, 
            "kv_head_num": split_kv_head_num, 
            "head_dim": head_dim, 
        }
    )

    model_topo.op_process_wrapper(
        "flash_attention", "flash_attention", 
        {
            "dtype": default_dtype, 
            "cache_dtype": attn_config["cache_dtype"], 
            "qk_compute_dtype": attn_config["qk_compute_dtype"], 
            "pv_compute_dtype": attn_config["pv_compute_dtype"], 
            "q_head_num": split_q_head_num, 
            "kv_head_num": split_kv_head_num, 
            "head_dim": head_dim, 
        }
    )

    model_topo.op_process_wrapper(
        "quant_matmul", "attn_out_gemm", 
        {
            "dtype": qkvo_config["dtype"], 
            "w_dtype": qkvo_config["w_dtype"], 
            "compute_dtype": qkvo_config["compute_dtype"], 
            "dst_dtype": default_dtype, 
            "has_bias": attention_out_bias, 
            "hidden_size": split_q_head_num * head_dim, 
            "new_hidden_size": hidden_size, 
        }
    )

    model_topo.op_process_wrapper(
        "all_reduce", "all_reduce_0", 
        {
            "world_size": dist_info.tp_size, 
            "dtype": default_dtype, 
            "hidden_size": hidden_size, 
        }
    )

    model_topo.op_process_wrapper(
        "add_rms_norm_dynamic_quant", "add_rms_norm_1", 
        {
            "dtype": default_dtype, 
            "dst_dtype": qkvo_config["dtype"], 
            "hidden_size": hidden_size, 
            "add_residual": True, 
            "output_mode": "res"
        }
    )


    model_topo.op_process_wrapper(
        "quant_matmul", "up_gemm", 
        {
            "dtype": mlp_config["dtype"], 
            "w_dtype": mlp_config["w_dtype"], 
            "compute_dtype": mlp_config["compute_dtype"], 
            "dst_dtype": default_dtype, 
            "has_bias": mlp_bias, 
            "hidden_size": hidden_size, 
            "new_hidden_size": model_config.intermediate_size // dist_info.tp_size * 2, 
        }
    )

    model_topo.op_process_wrapper(
        "swiglu_dynamic_quant", "swiglu_dynamic_quant", 
        {
            "dtype": default_dtype, 
            "dst_dtype": mlp_config["dtype"], 
            "hidden_size": model_config.intermediate_size // dist_info.tp_size, 
        }
    )

    model_topo.op_process_wrapper(
        "quant_matmul", "down_gemm", 
        {
            "dtype": mlp_config["dtype"], 
            "w_dtype": mlp_config["w_dtype"], 
            "compute_dtype": mlp_config["compute_dtype"], 
            "dst_dtype": default_dtype, 
            "has_bias": mlp_bias, 
            "hidden_size": model_config.intermediate_size // dist_info.tp_size, 
            "new_hidden_size": hidden_size, 
        }
    )

    model_topo.op_process_wrapper(
        "all_reduce", "all_reduce_1", 
        {
            "world_size": dist_info.tp_size, 
            "dtype": default_dtype, 
            "hidden_size": hidden_size, 
        }
    )


    return model_topo