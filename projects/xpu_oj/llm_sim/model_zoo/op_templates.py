from typing import List, Dict
from xpu_perf.model_perf.utils import BenchTestCase


"""
num_tokens_set_template
只关心有多少新增 tokens
"""
def num_tokens_set_template(
    workload: Dict, 
    test_case: BenchTestCase, 
    **kwargs
):
    workload["arg_type"] = "llm"
    workload["num_tokens"] = test_case.num_tokens


"""
mode_bs_cache_q_set_template
需要关心:
1. 是batch_size/cache_len/q_len 还是 cache_lens/q_lens
2. 是linear cache 还是 paged cache
"""
def mode_bs_cache_q_set_template(
    workload, 
    test_case: BenchTestCase, 
    run_mode: str = "prefill", 
    **kwargs
):
    workload["attn_mode"] = run_mode
    
    if test_case.is_var_input:
        workload["arg_type"] = "batch_llm"
        workload["cache_lens"] = test_case.cache_lens
        workload["q_lens"] = test_case.q_lens
    else:
        workload["arg_type"] = "llm"
        workload["batch_size"] = test_case.batch_size
        workload["cache_len"] = test_case.cache_len
        workload["q_len"] = test_case.q_len

    workload["block_size"] = test_case.block_size
    if test_case.cache_type == "linear":
        workload["slot_mapping"] = test_case.slot_mapping
    else:
        workload["block_table"] = test_case.block_table




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
    # Compatibility split-ops for DCU/lightop tp-ep templates.
    "moe_quant_group_gemm_up": num_tokens_set_template,
    "moe_quant_group_gemm_down": num_tokens_set_template,
    "moe_quant_group_gemm_combine": num_tokens_set_template, 
    "quant_group_gemm_reduce_sum": num_tokens_set_template, 

    "all_reduce": num_tokens_set_template, 
    "reduce_scatter": num_tokens_set_template, 
    "all_gather": num_tokens_set_template, 
    "all_to_all": num_tokens_set_template, 
}


