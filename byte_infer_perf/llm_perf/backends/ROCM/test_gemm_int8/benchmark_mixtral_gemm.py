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
# print(ops.__file__)
# exit()
from rocmKernels import gemm_a8w8
def torch_gemma8w8(a, b, alpha_row, alpha_col):
    b = b.transpose(0, 1)
    x = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    scale = torch.matmul(alpha_row, alpha_col)
    out = torch.mul(x, scale)
    return out.to(torch.half)
def get_MNK_shapes():
    MNK_SHAPES = [
        (1,4608,3584),
        (32,4608,3584),
        (64,4608,3584),
        (128,4608,3584),
        (256,4608,3584),
        (512,4608,3584),
        (1024,4608,3584),
        (2048,4608,3584),
        (4096,4608,3584),
        (8192,4608,3584),
        (16384,4608,3584),
        (20480,4608,3584),
        (1,3584,3584),
        (32,3584,3584),
        (64,3584,3584),
        (128,3584,3584),
        (256,3584,3584),
        (512,3584,3584),
        (1024,3584,3584),
        (2048,3584,3584),
        (4096,3584,3584),
        (8192,3584,3584),
        (16384,3584,3584),
        (20480,3584,3584),
        (1,3584,20480),
        (32,3584,20480),
        (64,3584,20480),
        (128,3584,20480),
        (256,3584,20480),
        (512,3584,20480),
        (1024,3584,20480),
        (2048,3584,20480),
        (4096,3584,20480),
        (8192,3584,20480),
        (16384,3584,20480),
        (20480,3584,20480),
        (1,40960,3584),
        (32,40960,3584),
        (64,40960,3584),
        (128,40960,3584),
        (256,40960,3584),
        (512,40960,3584),
        (1024,40960,3584),
        (2048,40960,3584),
        (4096,40960,3584),
        (8192,40960,3584),
        (16384,40960,3584),
        (20480,40960,3584),
    ]
    return MNK_SHAPES


def get_M_shapes():
    # Start M from 1 and init gemm_a8w8 at very beginning will cause inf error
    M_SHAPES = [2**i for i in range(4, 13)] + [1, 10, 20, 30, 40]
    # M_SHAPES = [2**i for i in range(4, model_config.exponent_of_max_seq_len + 1)]
    return M_SHAPES
def main():
    for m, n,k in get_MNK_shapes():
       # start_event = torch.cuda.Event(enable_timing=True)
       # end_event = torch.cuda.Event(enable_timing=True)
       # start_event.record()
        a_s = []
        b_s = []
        num_calls = 200
        for i in range(num_calls):
            a = torch.randint(-20, 20, (m, k), dtype=torch.int8).cuda()
            b = torch.randint(-20, 20, (n, k), dtype=torch.int8).cuda()
            a_s.append(a)
            b_s.append(b)

        alpha_row = torch.rand([m, 1], dtype=torch.half).cuda()
        alpha_col = torch.rand([1, n], dtype=torch.half).cuda()
        out_gemm = torch.empty([m,n],dtype = torch.half).cuda()
        out_ref = torch.empty([m,n],dtype = torch.half).cuda()
        print("self m",m,"n",n,"k",k)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ], on_trace_ready=torch.profiler.tensorboard_trace_handler("./"), with_modules=False, with_stack=False) as p:
            for i in range(num_calls):
                gemm_a8w8(a_s[i],b_s[i],alpha_row,alpha_col,out_gemm)
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        #end_event.record()
        #end_event.synchronize()
        #dur_us = start_event.elapsed_time(end_event)*1000 / num_calls
        #print("m",m,"n",n,"k",k, "dur_us",dur_us)
        out_ref = torch_gemma8w8(a,b,alpha_row,alpha_col)
        assert torch.allclose(
        out_ref, out_gemm, 1e-03, 1000), f"torch and ck results are not close\n{out_ref.shape}\n{out_ref}\n{out_gemm.shape}\n{out_gemm}\nmax delta:{(out_gemm-out_ref).max()}"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_mixtral_gemm",
        description="Tune the gemm kernel for mixtral.")
    parser.add_argument(
        "--TP",
        type=int,
        choices=[8, 4, 2, 1],
        help="Specify the TP value that the actual model will run on",
        default="0",
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
                        default="8x7B",
                        )

    args = parser.parse_args()

    print(f"Running tuning for {args.model} model")
    print(f"TP is set to: {args.TP}")
    print(f"GPU-ID being used for tuning: {args.GPUID}")
    sys.exit(main())

 

