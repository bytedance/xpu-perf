from .llm_ops import *
from .tensor_gemm_ops import *
from .vector_activation_ops import *
from .vector_index_ops import *
from .vector_linear_ops import *
from .vector_norm_ops import *
from .vector_reduction_ops import *
from .vector_sfu_ops import *
from .xccl_ops import *




SUPPORTED_OPS = {
    "XCCLEngine": {
        "all_reduce": AllReduceOp, 
        "reduce_scatter": ReduceScatterOp, 
        "all_gather": AllGatherOp, 
        "all_to_all": AlltoAllOp, 
        "broadcast": BroadcastOp, 
        "p2p": P2POp, 

        "all_reduce_h2d": AllReduce_H2D_Op, 

        "host2device": Host2DeviceOp, 
        "device2host": Device2HostOp, 
    }, 
    "ComputeEngine": {
        "device2device": Device2DeviceOp, 

        # vector_linear_ops
        "add": AddOp,
        "sub": SubOp,
        "mul": MulOp,
        "cast": CastOp,

        # vector_sfu_ops
        "div": DivOp,
        "sin": SinOp,
        "cos": CosOp,
        "exp": ExpOp,
        "log": LogOp,
        "sqrt": SqrtOp,

        # vector_reduction_ops
        "reduce_max": ReduceMaxOp,
        "reduce_min": ReduceMinOp,
        "reduce_sum": ReduceSumOp,
        "topk": TopkOp,

        # vector_norm_ops
        "layer_norm": LayerNormOp,
        "rms_norm": RMSNormOp,
        "softmax": SoftmaxOp,

        # vector_activation_ops
        "gelu": GeluOp,
        "silu": SiluOp,

        # vector_index_ops
        "embedding": EmbeddingOp,
        "gather": GatherOp,
        "index_select": IndexSelectOp,
        "scatter": ScatterOp,
        "index_add": IndexAddOp,

        # tensor_gemm_ops
        "gemm": GemmOp,

        # llm: basic
        "scale_dynamic_quant": ScaleDynamicQuantOp,
        "add_rms_norm_dynamic_quant": AddRmsNormDynamicQuantOp,
        "add_rms_norm": AddRmsNormOp,

        # llm: MOE
        "moe_gating_gemm": MoeGatingGemmOp,
        "moe_softmax_topk": MoeSoftmaxTopkOp,
        "moe_scatter_dynamic_quant": MoeScatterDynamicQuantOp,
        "quant_matmul": QuantMatmulOp,
        "quant_group_gemm_reduce_sum": QuantGroupGemmReduceSumOp,
        "moe_quant_group_gemm": MoeQuantGroupGemmOp,
        "moe_quant_group_gemm_combine": MoeQuantGroupGemmCombineOp,
        "moe_gather": MoeGatherOp,

        # swiglu ops
        "swiglu": SwigluOp, 
        "swiglu_dynamic_quant": SwigluDynamicQuantOp,
        "moe_swiglu": MoeSwigluOp,
        "moe_swiglu_dynamic_quant": MoeSwigluDynamicQuantOp,        

        # llm: ATTN
        "qk_rms_norm": QKRMSNormOp, 
        "head_rms_norm": HeadRMSNormOp,
        "head_rms_norm_dynamic_quant": HeadRMSNormDynamicQuantOp,
        "rotary_embedding": RotaryEmbeddingOp,
        "store_kv_cache": StoreKVCacheOp,
        # "dequant_kv_cache": DequantKVCacheOp,
        "flash_attention": FlashAttentionOp,
    }
}

OP_ENGINE_MAPPING = {}
DEFAULT_OP_IMPL_MAPPING = {}
for engine_name, engine_ops in SUPPORTED_OPS.items():
    for op_name in engine_ops:
        OP_ENGINE_MAPPING[op_name] = engine_name
        DEFAULT_OP_IMPL_MAPPING[op_name] = engine_ops[op_name]



