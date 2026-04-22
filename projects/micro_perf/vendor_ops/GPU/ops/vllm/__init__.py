import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "vllm"

try:
    ProviderRegistry.register_provider_info("vllm", {
        "vllm": importlib.metadata.version("vllm")
    })
except Exception:
    pass
