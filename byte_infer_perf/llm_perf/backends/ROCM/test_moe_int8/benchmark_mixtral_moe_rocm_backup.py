import argparse
import json
import os
import sys
import unittest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm
_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, f'{_path}/../')
#import vllm._moe_C as moe_kernels
import rocmKernels as ops
from rocm_kernels.fused_moe_int8_a8w8 import (fused_moe_int8_a8w8,
                                    scaled_int8_quant)
from rocm_kernels.fused_moe_custom import (fused_moe_int8_a8w8_custom,
                                    get_config_file_name,
                                    triton_dynamic_quantize)


def main(args):
    os.environ["HIP_VISIBLE_DEVICES"] = args.GPUID
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    os.environ["DEBUG_CLR_GRAPH_PACKET_CAPTURE"] = "1"
    os.environ["OPTIMIZE_EPILOGUE"] = "1"

    for bs in [
#           1,
#           2,
#           4,
#           8,
            16,
#           24,
#           32,
#           48,
#           64,
#           96,
#           128,
#           256,
#           512,
#           1024,
#           1536,
#           2048,
#           3072,
#           4096,
    ]:
        run_grid(bs, model=args.model, TP=args.TP)


## Utilize method from rocm/Triton tuning script
def get_full_tuning_space():
    configs = []

    #block_mn_range = [16, 32, 64, 128, 256]
    block_mn_range = [32]
    #block_k_range = [32, 64, 128, 256]
    block_k_range = [32]
    block_mn_range = [16]
    # block_k_range = [64]
    #split_k_range = [1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    split_k_range = [4]
    #num_warps_range = [1, 2, 4, 8]
    num_warps_range = [2]
    # num_warps_range = [1]
    #group_m_range = [1, 4, 8, 16, 32]
    group_m_range = [1]
    # group_m_range = [1]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [0]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16]
    #matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [ 2]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        # for split_k in split_k_range:
                        for num_stages in num_stage_range:
                            for waves_per_eu in waves_per_eu_range:
                                for (matrix_instr_nonkdim
                                     ) in matrix_instr_nonkdim_range:
                                    for kpack in kpack_range:
                                        configs.append({
                                            "BLOCK_SIZE_M": block_m,
                                            "BLOCK_SIZE_N": block_n,
                                            "BLOCK_SIZE_K": block_k,
                                            "GROUP_SIZE_M": group_m,
                                            "num_warps": num_warps,
                                            "num_stages": num_stages,                          
                                        })

    return configs


## Utilize method from rocm/Triton tuning script
def prune_configs(M, N, K, configs):
    pruned_configs = []
    elemBytes_a = 2  # [DV Note] Hard-coded for float16 (2 bytes)
    elemBytes_b = 2  # [DV Note] Hard-coded for float16 (2 bytes)

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = 1  # config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a +
               BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b)
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


def union_of_list_of_dicts(l1, l2):
    result = []
    temp_list = l1.copy()
    temp_list.extend(l2)
    for myDict in temp_list:
        if myDict not in result:
            result.append(myDict)

    return result


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024

def torch_moe(hidden_states, w1, w2, score, topk):
    #print("in side torch moe w1, w2", w1, w2, hidden_states)
    B, D = hidden_states.shape
    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(
        B * topk,
        w2.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    #print("torch topk_weight",topk_weight,"topk_ids",topk_ids)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            silu_input = hidden_states[mask] @ (w1[i].transpose(0, 1))
            #
            d = silu_input.shape[-1] // 2
            silu_output_shape = silu_input.shape[:-1] + (d,)
            silu_out = torch.empty(
                silu_output_shape, dtype=silu_input.dtype, device=silu_input.device
            )
            ops.silu_and_mul(silu_out, silu_input)
            #
            out[mask] = silu_out @ (w2[i].transpose(0, 1))
    #out = out + 2.0
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)
def dynamic_quan_torch_impl(input):
    max_input = input.abs().max(-1, keepdim=True)[0]
    scale = max_input / 127.0
    out = torch.round(input / scale)
    return out.to(torch.int8), scale.half().squeeze(-1)
def run_grid(bs, model, TP):
    if model == '8x7B':
        d_model = 4096
        #d_model = 32
        model_intermediate_size = 14336
    elif model == '8x22B':
        d_model = 6144
        model_intermediate_size = 16384
    else:
        raise ValueError(f'Unsupported Mixtral model {model}')

    num_total_experts = 8
    top_k = 2
    tp_size = TP
    num_calls = 100

    num_warmup_trials = 1
    num_trials = 1

    full_configs = get_full_tuning_space()
    M1 = bs * 2
    N1 = model_intermediate_size * 2 // tp_size
    K1 = d_model
    prune_configs_1 = prune_configs(M1, N1, K1, full_configs)

    M2 = bs * 2
    N2 = d_model
    K2 = model_intermediate_size // tp_size
    prune_configs_2 = prune_configs(M2, N2, K2, full_configs)

    configs = union_of_list_of_dicts(prune_configs_1, prune_configs_2)
    print(f"{bs=} || {len(full_configs)=} | {len(prune_configs_1)=} | \
            {len(prune_configs_2)=} | {len(configs)=}")

    best_config = None
    best_time_us = 1e20

    for config in tqdm(configs):
        print("have config")
        # warmup
        try:
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_calls,
                    bs=bs,
                    d_model=d_model,
                    num_total_experts=num_total_experts,
                    top_k=top_k,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    config=config,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # benchmark
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                num_total_experts=num_total_experts,
                top_k=top_k,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                config=config,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            # model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

    filename = get_config_file_name(num_total_experts,
                                    model_intermediate_size // tp_size,
                                    dtype=None)
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def run_timing(
    num_calls: int,
    bs: int,
    d_model: int,
    num_total_experts: int,
    top_k: int,
    tp_size: int,
    model_intermediate_size: int,
    config,
) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size
    print("run timing")
    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda",
        dtype=torch.float16,
    )

    w1 = torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )/100

    w2 = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )/100
    a2_scales = torch.rand((hidden_states.shape[1]),
            device = hidden_states.device,
            dtype=hidden_states.dtype)
    gating_output = F.softmax(
        torch.rand(
            # (num_calls, bs, num_total_experts), # THIS
            (bs, num_total_experts),
            device=hidden_states.device,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    ###### Stuff from fused moe ######
    hidden_states_quant,hidden_states_scales = dynamic_quan_torch_impl(hidden_states)
    w1_quant, w1_scales = dynamic_quan_torch_impl(w1)
    w2_quant, w2_scales = dynamic_quan_torch_impl(w2)
    assert (hidden_states.shape[0] == gating_output.shape[0]
            ), "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output = fused_moe_int8_a8w8(hidden_states_quant,
            w1_quant,
            w2_quant,
            gating_output,
            w1_scales,
            w2_scales,
            hidden_states_scales,
            a2_scales,
            top_k,
            renormalize=False,
            inplace=False)
    output_custom = fused_moe_int8_a8w8_custom(hidden_states_quant,
            hidden_states_scales,
            w1_quant,
            w1_scales,
            w2_quant,
            w2_scales,
            gating_output,
            top_k,
            renormalize=False,
            inplace=False)
    hidden_states_dequant = hidden_states_quant * hidden_states_scales[:, None]
    w1_dequant = w1_quant * w1_scales[:, :, None]
    w2_dequant = w2_quant * w2_scales[:, :, None]
    out_ref = torch_moe(hidden_states_dequant,
            w1_dequant,
            w2_dequant,
            gating_output,
            top_k,
            )
    diff = ~torch.isclose(
            output.half().cpu(), out_ref.half().cpu(), rtol=1, atol=1
        )
    print("output:",output)
#    print("output custom:",output_custom)
    print("out_ref:",out_ref)
    assert(diff.sum() < 10)
    print("diff sum :",diff.sum())
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_mixtral_moe_rocm",
        description="Tune the fused_moe kernel for mixtral.")
    parser.add_argument(
        "--TP",
        type=int,
        choices=[8, 4, 2, 1],
        help="Specify the TP value that the actual model will run on",
        required=True,
    )
    parser.add_argument(
        "--GPUID",
        type=str,
        help="This script uses single GPU. Specify the GPU to use for tuning",
        default="0",
    )
    parser.add_argument('--model',
                        type=str,
                        choices=['8x7B', '8x22B'],
                        help='The Mixtral model to benchmark')

    args = parser.parse_args()

    print(f"Running tuning for {args.model} model")
    print(f"TP is set to: {args.TP}")
    print(f"GPU-ID being used for tuning: {args.GPUID}")
    sys.exit(main(args))
