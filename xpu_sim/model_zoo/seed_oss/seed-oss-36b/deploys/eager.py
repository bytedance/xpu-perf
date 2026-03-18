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

from model_zoo import OpTopologyDAG

"""
eager 模式
- bf16
- single device
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

    intermediate_size = model_config.intermediate_size

    
    default_dtype = "bfloat16"
    


    model_topo = OpTopologyDAG()
    model_topo.op_process_wrapper(
        "add_rms_norm", "add_rms_norm_0", 
        {
            "dtype": default_dtype, 
            "dst_dtype": default_dtype, 
            "hidden_size": hidden_size
        }
    )

    model_topo.op_process_wrapper(
        "gemm", "qkv_gemm", 
        {
            "dtype": default_dtype, 
            "dst_dtype": default_dtype, 
            "hidden_size": hidden_size, 
            "new_hidden_size": (q_head_num + 2 * kv_head_num) * head_dim
        }
    )

    model_topo.op_process_wrapper(
        "rotary_embedding", "rotary_embedding", 
        {
            "dtype": default_dtype, 
            "q_head_num": q_head_num, 
            "kv_head_num": kv_head_num, 
            "head_dim": head_dim, 
            "rope_offset": 0, 
            "rope_dim": head_dim, 
        }
    )

    model_topo.op_process_wrapper(
        "store_kv_cache", "store_kv_cache", 
        {
            "dtype": default_dtype, 
            "cache_dtype": default_dtype, 
            "q_head_num": q_head_num, 
            "kv_head_num": kv_head_num, 
            "head_dim": head_dim, 
        }
    )

    model_topo.op_process_wrapper(
        "flash_attention", "flash_attention", 
        {
            "dtype": default_dtype, 
            "cache_dtype": default_dtype, 
            "qk_compute_dtype": default_dtype, 
            "pv_compute_dtype": default_dtype, 
            "q_head_num": q_head_num, 
            "kv_head_num": kv_head_num, 
            "head_dim": head_dim
        }
    )

    model_topo.op_process_wrapper(
        "gemm", "attn_out_gemm", 
        {
            "dtype": default_dtype, 
            "dst_dtype": default_dtype, 
            "hidden_size": q_head_num * head_dim, 
            "new_hidden_size": hidden_size
        }
    )

    model_topo.op_process_wrapper(
        "add_rms_norm", "add_rms_norm_1", 
        {
            "dtype": default_dtype, 
            "dst_dtype": default_dtype, 
            "hidden_size": hidden_size
        }
    )

    model_topo.op_process_wrapper(
        "gemm", "up_gemm", 
        {
            "dtype": default_dtype, 
            "dst_dtype": default_dtype, 
            "hidden_size": hidden_size, 
            "new_hidden_size": intermediate_size * 2
        }
    )

    model_topo.op_process_wrapper(
        "swiglu", "swiglu",
        {
            "dtype": default_dtype, 
            "hidden_size": intermediate_size, 
        }
    )

    model_topo.op_process_wrapper(
        "gemm", "down_gemm", 
        {
            "dtype": default_dtype, 
            "dst_dtype": default_dtype, 
            "hidden_size": intermediate_size, 
            "new_hidden_size": hidden_size
        }
    )

    return model_topo
