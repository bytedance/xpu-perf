import os
from typing import Dict

from transformers import Qwen3MoeConfig

from model_zoo.topology import OpTopologyDAG
from xpu_perf.model_perf.utils import DistributionInfo

"""
tp-ep 模式 (ep_size == tp_size)
Attention 部分使用纯 TP 并行，MoE 部分在 TP 基础上加入 EP 并行通信
"""


def generate(
    model_config: Qwen3MoeConfig,
    bench_config: Dict,
):
    # parse model params
    hidden_size = model_config.hidden_size
    q_head_num = model_config.num_attention_heads
    kv_head_num = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    attention_bias = model_config.attention_bias

    moe_intermediate_size = model_config.moe_intermediate_size
    num_experts = model_config.num_experts
    num_experts_per_tok = model_config.num_experts_per_tok

    # parse distribution info
    dist_info = DistributionInfo.from_bench_config(bench_config["parallel_config"])

    split_q_head_num = q_head_num // dist_info.tp_size if q_head_num >= dist_info.tp_size else 1
    split_kv_head_num = kv_head_num // dist_info.tp_size if kv_head_num >= dist_info.tp_size else 1

    # 获取默认数据类型
    default_dtype = bench_config.get("dtype_config", {}).get("default_dtype", "bfloat16")

    qkvo_config = bench_config["dtype_config"]["qkvo"]
    attn_config = bench_config["dtype_config"]["attn"]
    gating_config = bench_config["dtype_config"]["gating"]
    mlp_config = bench_config["dtype_config"]["mlp"]
    extra_config = bench_config.get("extra_config", {})
    moe_gather_res_scale = float(extra_config.get("moe_gather_res_scale", 1.0))

    model_topo = OpTopologyDAG()

    # ============================================================
    # Attention 部分 (纯 TP 并行)
    # ============================================================
    model_topo.op_process_wrapper(
        "add_rms_norm_dynamic_quant",
        "add_rms_norm_0",
        {
            "dtype": default_dtype,
            "dst_dtype": qkvo_config["dtype"],
            "hidden_size": hidden_size,
            "add_residual": True,
            "output_mode": "res",
        },
    )

    model_topo.op_process_wrapper(
        "quant_matmul",
        "qkv_gemm",
        {
            "dtype": qkvo_config["dtype"],
            "w_dtype": qkvo_config["w_dtype"],
            "compute_dtype": qkvo_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "has_bias": attention_bias,
            "hidden_size": hidden_size,
            "new_hidden_size": (split_q_head_num + 2 * split_kv_head_num) * head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "qk_rms_norm",
        "qk_norm",
        {
            "dtype": default_dtype,
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "qk_head_dim": head_dim,
            "v_head_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "rotary_embedding",
        "rotary_embedding",
        {
            "dtype": default_dtype,
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "head_dim": head_dim,
            "rope_offset": 0,
            "rope_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "store_kv_cache",
        "store_kv_cache",
        {
            "dtype": default_dtype,
            "cache_dtype": attn_config["cache_dtype"],
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "head_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "flash_attention",
        "flash_attention",
        {
            "dtype": default_dtype,
            "cache_dtype": attn_config["cache_dtype"],
            "qk_compute_dtype": attn_config["qk_compute_dtype"],
            "pv_compute_dtype": attn_config["pv_compute_dtype"],
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "head_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "quant_matmul",
        "attn_out_gemm",
        {
            "dtype": qkvo_config["dtype"],
            "w_dtype": qkvo_config["w_dtype"],
            "compute_dtype": qkvo_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "has_bias": attention_bias,
            "hidden_size": split_q_head_num * head_dim,
            "new_hidden_size": hidden_size,
        },
    )

    model_topo.op_process_wrapper(
        "all_reduce",
        "all_reduce_0",
        {
            "world_size": dist_info.tp_size,
            "dtype": default_dtype,
            "hidden_size": hidden_size,
        },
    )

    # ============================================================
    # MoE 部分 (TP + EP 并行)
    # ============================================================
    pre_moe = model_topo.op_process_wrapper(
        "add_rms_norm",
        "qwen3_pre_moe_norm",
        {"dtype": default_dtype, "hidden_size": hidden_size},
    )

    a2a0_node = model_topo.op_process_wrapper(
        "all_to_all",
        "qwen3_moe_a2a0",
        {
            "dtype": default_dtype,
            "world_size": dist_info.ep_size,
            "hidden_size": hidden_size,
        },
        src=pre_moe,
    )

    model_topo.op_process_wrapper(
        "moe_gating_gemm",
        "qwen3_moe_gating",
        {
            "dtype": gating_config["dtype"],
            "compute_dtype": gating_config["compute_dtype"],
            "dst_dtype": gating_config.get("dst_dtype", "float32"),
            "num_experts": num_experts,
            "hidden_size": hidden_size,
        },
        src=pre_moe,
    )

    topk_node = model_topo.op_process_wrapper(
        "moe_softmax_topk",
        "qwen3_moe_softmax_topk",
        {
            "dtype": "float32",
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "compute_mode": extra_config.get("compute_mode", "pre-softmax"),
        },
    )

    scatter_node = model_topo.op_process_wrapper(
        "moe_scatter_dynamic_quant",
        "qwen3_moe_scatter",
        {
            "dtype": default_dtype,
            "dst_dtype": mlp_config["dtype"],
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "hidden_size": hidden_size,
        },
        src=[a2a0_node, topk_node],
    )

    tile_k = int(extra_config.get("tile_k", 0))
    tile_n = int(extra_config.get("tile_n", 0))

    model_topo.op_process_wrapper(
        "moe_quant_group_gemm_up",
        "qwen3_moe_up_gemm",
        {
            "dtype": mlp_config["dtype"],
            "w_dtype": mlp_config["w_dtype"],
            "compute_dtype": mlp_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "hidden_size": hidden_size,
            "new_hidden_size": moe_intermediate_size * 2,
        },
        src=scatter_node,
    )

    model_topo.op_process_wrapper(
        "moe_swiglu_dynamic_quant",
        "qwen3_moe_swiglu",
        {
            "dtype": default_dtype,
            "dst_dtype": mlp_config["dtype"],
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "hidden_size": moe_intermediate_size,
        },
    )

    model_topo.op_process_wrapper(
        "moe_quant_group_gemm_down",
        "qwen3_moe_down_gemm",
        {
            "dtype": mlp_config["dtype"],
            "w_dtype": mlp_config["w_dtype"],
            "compute_dtype": mlp_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "hidden_size": moe_intermediate_size,
            "new_hidden_size": hidden_size,
        },
    )

    model_topo.op_process_wrapper(
        "all_to_all",
        "qwen3_moe_a2a1",
        {"dtype": default_dtype, "world_size": dist_info.ep_size, "hidden_size": hidden_size},
    )

    model_topo.op_process_wrapper(
        "moe_gather",
        "qwen3_moe_gather",
        {
            "dtype": default_dtype,
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "hidden_size": hidden_size,
            "res_scale": moe_gather_res_scale,
        },
    )

    return model_topo

import os
from typing import Dict

from transformers import Qwen3MoeConfig

from model_zoo.topology import OpTopologyDAG
from xpu_perf.model_perf.utils import DistributionInfo

"""
tp-ep 模式 (ep_size == tp_size)
Attention 部分使用纯 TP 并行，MoE 部分在 TP 基础上加入 EP 并行通信
"""


def generate(
    model_config: Qwen3MoeConfig,
    bench_config: Dict,
):
    # parse model params
    hidden_size = model_config.hidden_size
    q_head_num = model_config.num_attention_heads
    kv_head_num = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    attention_bias = model_config.attention_bias

    moe_intermediate_size = model_config.moe_intermediate_size
    num_experts = model_config.num_experts
    num_experts_per_tok = model_config.num_experts_per_tok

    # parse distribution info
    dist_info = DistributionInfo.from_bench_config(bench_config["parallel_config"])

    split_q_head_num = q_head_num // dist_info.tp_size if q_head_num >= dist_info.tp_size else 1
    split_kv_head_num = kv_head_num // dist_info.tp_size if kv_head_num >= dist_info.tp_size else 1

    # 获取默认数据类型
    default_dtype = bench_config.get("dtype_config", {}).get("default_dtype", "bfloat16")

    qkvo_config = bench_config["dtype_config"]["qkvo"]
    attn_config = bench_config["dtype_config"]["attn"]
    gating_config = bench_config["dtype_config"]["gating"]
    mlp_config = bench_config["dtype_config"]["mlp"]
    extra_config = bench_config.get("extra_config", {})
    moe_gather_res_scale = float(extra_config.get("moe_gather_res_scale", 1.0))

    model_topo = OpTopologyDAG()

    # ============================================================
    # Attention 部分 (纯 TP 并行)
    # ============================================================
    model_topo.op_process_wrapper(
        "add_rms_norm_dynamic_quant",
        "add_rms_norm_0",
        {
            "dtype": default_dtype,
            "dst_dtype": qkvo_config["dtype"],
            "hidden_size": hidden_size,
            "add_residual": True,
            "output_mode": "res",
        },
    )

    model_topo.op_process_wrapper(
        "quant_matmul",
        "qkv_gemm",
        {
            "dtype": qkvo_config["dtype"],
            "w_dtype": qkvo_config["w_dtype"],
            "compute_dtype": qkvo_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "has_bias": attention_bias,
            "hidden_size": hidden_size,
            "new_hidden_size": (split_q_head_num + 2 * split_kv_head_num) * head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "qk_rms_norm",
        "qk_norm",
        {
            "dtype": default_dtype,
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "qk_head_dim": head_dim,
            "v_head_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "rotary_embedding",
        "rotary_embedding",
        {
            "dtype": default_dtype,
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "head_dim": head_dim,
            "rope_offset": 0,
            "rope_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "store_kv_cache",
        "store_kv_cache",
        {
            "dtype": default_dtype,
            "cache_dtype": attn_config["cache_dtype"],
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "head_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "flash_attention",
        "flash_attention",
        {
            "dtype": default_dtype,
            "cache_dtype": attn_config["cache_dtype"],
            "qk_compute_dtype": attn_config["qk_compute_dtype"],
            "pv_compute_dtype": attn_config["pv_compute_dtype"],
            "q_head_num": split_q_head_num,
            "kv_head_num": split_kv_head_num,
            "head_dim": head_dim,
        },
    )

    model_topo.op_process_wrapper(
        "quant_matmul",
        "attn_out_gemm",
        {
            "dtype": qkvo_config["dtype"],
            "w_dtype": qkvo_config["w_dtype"],
            "compute_dtype": qkvo_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "has_bias": attention_bias,
            "hidden_size": split_q_head_num * head_dim,
            "new_hidden_size": hidden_size,
        },
    )

    model_topo.op_process_wrapper(
        "all_reduce",
        "all_reduce_0",
        {
            "world_size": dist_info.tp_size,
            "dtype": default_dtype,
            "hidden_size": hidden_size,
        },
    )

    # ============================================================
    # MoE 部分 (TP + EP 并行)
    # ============================================================
    pre_moe = model_topo.op_process_wrapper(
        "add_rms_norm",
        "qwen3_pre_moe_norm",
        {"dtype": default_dtype, "hidden_size": hidden_size},
    )

    a2a0_node = model_topo.op_process_wrapper(
        "all_to_all",
        "qwen3_moe_a2a0",
        {
            "dtype": default_dtype,
            "world_size": dist_info.ep_size,
            "hidden_size": hidden_size,
        },
        src=pre_moe,
    )

    model_topo.op_process_wrapper(
        "moe_gating_gemm",
        "qwen3_moe_gating",
        {
            "dtype": gating_config["dtype"],
            "compute_dtype": gating_config["compute_dtype"],
            "dst_dtype": gating_config.get("dst_dtype", "float32"),
            "num_experts": num_experts,
            "hidden_size": hidden_size,
        },
        src=pre_moe,
    )

    topk_node = model_topo.op_process_wrapper(
        "moe_softmax_topk",
        "qwen3_moe_softmax_topk",
        {
            "dtype": "float32",
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "compute_mode": extra_config.get("compute_mode", "pre-softmax"),
        },
    )

    scatter_node = model_topo.op_process_wrapper(
        "moe_scatter_dynamic_quant",
        "qwen3_moe_scatter",
        {
            "dtype": default_dtype,
            "dst_dtype": mlp_config["dtype"],
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "hidden_size": hidden_size,
        },
        src=[a2a0_node, topk_node],
    )

    tile_k = int(extra_config.get("tile_k", 0))
    tile_n = int(extra_config.get("tile_n", 0))

    model_topo.op_process_wrapper(
        "moe_quant_group_gemm_up",
        "qwen3_moe_up_gemm",
        {
            "dtype": mlp_config["dtype"],
            "w_dtype": mlp_config["w_dtype"],
            "compute_dtype": mlp_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "hidden_size": hidden_size,
            "new_hidden_size": moe_intermediate_size * 2,
        },
        src=scatter_node,
    )

    model_topo.op_process_wrapper(
        "moe_swiglu_dynamic_quant",
        "qwen3_moe_swiglu",
        {
            "dtype": default_dtype,
            "dst_dtype": mlp_config["dtype"],
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "hidden_size": moe_intermediate_size,
        },
    )

    model_topo.op_process_wrapper(
        "moe_quant_group_gemm_down",
        "qwen3_moe_down_gemm",
        {
            "dtype": mlp_config["dtype"],
            "w_dtype": mlp_config["w_dtype"],
            "compute_dtype": mlp_config["compute_dtype"],
            "dst_dtype": default_dtype,
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "tile_k": tile_k,
            "tile_n": tile_n,
            "hidden_size": moe_intermediate_size,
            "new_hidden_size": hidden_size,
        },
    )

    model_topo.op_process_wrapper(
        "all_to_all",
        "qwen3_moe_a2a1",
        {"dtype": default_dtype, "world_size": dist_info.ep_size, "hidden_size": hidden_size},
    )

    model_topo.op_process_wrapper(
        "moe_gather",
        "qwen3_moe_gather",
        {
            "dtype": default_dtype,
            "ep_size": dist_info.ep_size,
            "num_experts": num_experts,
            "topk": num_experts_per_tok,
            "hidden_size": hidden_size,
            "res_scale": moe_gather_res_scale,
        },
    )

    return model_topo

