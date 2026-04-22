import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "flashinfer"

try:
    ProviderRegistry.register_provider_info("flashinfer", {
        "flashinfer": importlib.metadata.version("flashinfer-python")
    })
except Exception:
    pass
