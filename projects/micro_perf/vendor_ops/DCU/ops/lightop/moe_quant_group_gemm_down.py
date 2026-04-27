"""lightop vendor: moe_quant_group_gemm_down

Compatibility provider for legacy split-op workloads.
Currently it reuses the unified `moe_quant_group_gemm` lightop implementation.
"""

from xpu_perf.micro_perf.core.op import ProviderRegistry

from .moe_quant_group_gemm import LightopMoeQuantGroupGemmOp as _UnifiedLightopMoeQuantGroupGemmOp


@ProviderRegistry.register_vendor_impl("moe_quant_group_gemm_down", "lightop")
class LightopMoeQuantGroupGemmDownOp(_UnifiedLightopMoeQuantGroupGemmOp):
    pass
