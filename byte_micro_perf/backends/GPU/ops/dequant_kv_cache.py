import sys
import pathlib
from functools import partial

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import DequantKVCacheOp
from core.utils import OpTensorInfo, calc_tensor_size


OP_MAPPING = {
    "torch": DequantKVCacheOp,
}