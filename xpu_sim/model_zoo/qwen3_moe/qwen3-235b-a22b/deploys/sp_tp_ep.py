import os
import sys
import pathlib
from typing import Dict
from transformers import Qwen3MoeConfig

# model_type / model_name / deploy / deploy_0.py
DEPLOY_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_NAME_DIR = DEPLOY_DIR.parent
MODEL_TYPE_DIR = MODEL_NAME_DIR.parent
MODEL_ZOO_DIR = MODEL_TYPE_DIR.parent

sys.path.insert(0, str(MODEL_ZOO_DIR.parent))

from model_zoo import OpTopologyDAG, DistributionInfo
from model_zoo.topology import add_moe_graph

"""
sp-tp-ep 模式
"""
def generate(
    model_config: Qwen3MoeConfig, 
    bench_config: Dict
):
    # parse model params
    hidden_size = model_config.hidden_size
    q_head_num = model_config.num_attention_heads
    kv_head_num = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    attention_bias = model_config.attention_bias

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
    gating_config = bench_config["dtype_config"]["gating"]
    mlp_config = bench_config["dtype_config"]["mlp"]




    model_topo = OpTopologyDAG()


    # sp: [num_tokens // sp_size, hidden_size]
    model_topo.op_process_wrapper(
        "add_rms_norm_dynamic_quant", "add_rms_norm_0", 
        {
            "dtype": default_dtype, 
            "dst_dtype": qkvo_config["dtype"],
            "sp_size": dist_info.sp_size, 
            "hidden_size": hidden_size, 
            "add_residual": True, 
            "output_mode": "res"
        }
    )

    
    """
    ***************************************
    TODO: 这部分可以做一个 通算融合算子
    ***************************************
    """

    """
    由于 sp-->tp 的过程中的 all_to_all 只能对外层的维度进行切分
    因此, 这里的 gemm 再写回数据的时候需要重排布一下, 不影响性能
    
    [num_tokens // sp_size, hidden_size]
    --> [num_tokens // sp_size, (q_head_num + 2 * kv_head_num) * head_dim]
    --> [num_tokens // sp_size, tp_size, (q_head_num + 2 * kv_head_num) // tp_size * head_dim]
    --> [tp_size, num_tokens // sp_size, (q_head_num + 2 * kv_head_num) // tp_size * head_dim]
    """
    model_topo.op_process_wrapper(
        "quant_matmul", "qkv_gemm", 
        {
            "dtype": qkvo_config["dtype"], 
            "w_dtype": qkvo_config["w_dtype"], 
            "compute_dtype": qkvo_config["compute_dtype"], 
            "dst_dtype": default_dtype, 
            "sp_size": dist_info.sp_size, 
            "has_bias": attention_bias, 
            "transpose_o": True, 
            "hidden_size": hidden_size, 
            "new_hidden_size": (split_q_head_num + 2 * split_kv_head_num) \
                                * dist_info.tp_size * head_dim
        }
    )


    """
    sp --> tp
    [tp_size, num_tokens // sp_size, (q_head_num + 2 * kv_head_num) // tp_size * head_dim]
    --> [sp_size, num_tokens // sp_size, (q_head_num + 2 * kv_head_num) // tp_size * head_dim]
    --> [num_tokens, (q_head_num + 2 * kv_head_num) // tp_size * head_dim]
    """
    model_topo.op_process_wrapper(
        "all_to_all", "all_to_all_0", 
        {
            "dtype": default_dtype, 
            "world_size": dist_info.sp_size, 
            "hidden_size": (split_q_head_num + 2 * split_kv_head_num) * head_dim, 
        }
    )
    """
    ***************************************
    这部分可以做一个 通算融合算子
    ***************************************
    """






    # tp: [num_token, (q_head_num + 2 * kv_head_num) // tp_size, head_dim]
    model_topo.op_process_wrapper(
        "qk_rms_norm", "qk_norm", 
        {
            "dtype": default_dtype, 
            "q_head_num": split_q_head_num, 
            "kv_head_num": split_kv_head_num, 
            "qk_head_dim": head_dim, 
            "v_head_dim": head_dim
        }
    )

    # tp: [num_token, (q_head_num + 2 * kv_head_num) // tp_size, head_dim]
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

    # tp: [num_token, (q_head_num + 2 * kv_head_num) // tp_size, head_dim]
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

    """
    tp: [num_token, (q_head_num + 2 * kv_head_num) // tp_size, head_dim]
    --> tp: [num_token, q_head_num // tp_size, head_dim]
    """
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




    """
    ***************************************
    TODO: 这部分可以做一个 通算融合算子
    ***************************************
    """

    """
    对 attntion 输出进行量化
    - in:   [num_tokens, q_head_num // tp_size * head_dim]
    - out:  [num_tokens, q_head_num // tp_size * head_dim], 
    - out:  [num_tokens, ]
    """
    model_topo.op_process_wrapper(
        "scale_dynamic_quant", "attn_out_quant", 
        {
            "dtype": default_dtype, 
            "dst_dtype": qkvo_config["dtype"], 
            "hidden_size": split_q_head_num * head_dim, 
        }
    )

    """
    SP --> TP
    - all_to_all_0: 交换 quant_value
        * [sp_size, num_tokens // sp_size, q_head_num // tp_size * head_dim]
    - all_to_all_1: 交换 per_token_scale
        * [sp_size, num_tokens // sp_size, 1]
    """
    model_topo.op_process_wrapper(
        "all_to_all", "all_to_all_1", 
        {
            "dtype": qkvo_config["dtype"], 
            "world_size": dist_info.sp_size, 
            "hidden_size": split_q_head_num * head_dim
        }
    )
    model_topo.op_process_wrapper(
        "all_to_all", "all_to_all_2", 
        {
            "dtype": "float32", 
            "world_size": dist_info.sp_size, 
            "hidden_size": 1
        }
    )

    """
    ***************************************
    这部分可以做一个 通算融合算子
    ***************************************
    """




    """
    attn_out_gemm
    由于 all_to_all 交换回来的同一个token的多部分的head的量化参数各自独立
    所以这里需要通过 group_gemm 实现再进行多个group的reduce
    此外, all_to_all 交换回来的 token 数据在 head 维度也不连续
    """
    model_topo.op_process_wrapper(
        "quant_group_gemm_reduce_sum", "attn_out_gemm", 
        {
            "dtype": qkvo_config["dtype"], 
            "dst_dtype": default_dtype, 
            "sp_size": dist_info.sp_size, 
            "hidden_size": split_q_head_num * head_dim, 
            "new_hidden_size": hidden_size
        }
    )

    """
    到这里后, 就都是 sp 并行了, [num_tokens // sp_size, hidden_size]
    有两种实现: 
    1. 本地 gating + all_gather + 本地 dispatch
    2. 本地 gating + deepep (all_to_all_v)

    再后面就是 group_gemm + combine
    """
    pre_moe = model_topo.op_process_wrapper(
        "add_rms_norm", "pre_moe_norm", 
        {
            "dtype": default_dtype, 
            "sp_size": dist_info.sp_size, 
            "hidden_size": hidden_size
        }
    )



    moe_output_node = add_moe_graph(
        model_topo, 
        pre_node=pre_moe, 
        hidden_size=hidden_size, 

        num_experts=model_config.num_experts, 
        moe_topk=model_config.num_experts_per_tok, 
        moe_intermediate_size=model_config.moe_intermediate_size, 
        moe_mlp_config=mlp_config, 

        # qwen3-moe models have no share experts
        num_share_experts=0, 
        share_intermediate_size=model_config.moe_intermediate_size, 
        share_mlp_config=qkvo_config,

        is_pre_softmax=True, 

        sp_size=dist_info.sp_size, 
        ep_size=dist_info.ep_size, 
        
        use_deepep=False, 
        fuse_combine=True,

        default_dtype=default_dtype, 
        module_name="qwen3_moe"
    )

    return model_topo
    

