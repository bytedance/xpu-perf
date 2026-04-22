import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "flash_attn"

try:
    import flash_attn
    ProviderRegistry.register_provider_info("flash_attn_v2", {
        "flash_attn": importlib.metadata.version("flash_attn")
    })
except Exception:
    pass

try:
    import flash_attn_interface
    ProviderRegistry.register_provider_info("flash_attn_v3", {
        "flash_attn": importlib.metadata.version("flash_attn")
    })
except Exception:
    pass
