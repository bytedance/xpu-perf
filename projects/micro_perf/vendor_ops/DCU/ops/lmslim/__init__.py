import importlib.metadata

from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "dcu_lmslim"

try:
    ProviderRegistry.register_provider_info(
        "lmslim", {"lmslim": importlib.metadata.version("lmslim")}
    )
except Exception:
    pass
