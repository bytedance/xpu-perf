import importlib.metadata

from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "dcu_lightop"

try:
    ProviderRegistry.register_provider_info(
        "lightop", {"lightop": importlib.metadata.version("lightop")}
    )
except Exception:
    pass
