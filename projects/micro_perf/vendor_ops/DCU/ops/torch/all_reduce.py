from functools import partial
import torch
import torch.distributed as dist

from xpu_perf_provider_base_ops.basic_ops.xccl_ops import AllReduceOp

OP_MAPPING = {
    "torch": AllReduceOp
}
