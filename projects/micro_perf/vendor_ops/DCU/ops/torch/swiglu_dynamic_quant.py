from xpu_perf_provider_base_ops.llm_ops.swiglu_dynamic_quant import SwigluDynamicQuantOp

OP_MAPPING = {
    "torch": SwigluDynamicQuantOp
}
