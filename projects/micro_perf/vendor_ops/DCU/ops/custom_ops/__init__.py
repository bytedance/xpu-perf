import importlib.metadata

from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "dcu_custom_ops"

try:
    ProviderRegistry.register_provider_info(
        "custom_ops", {"custom_ops": importlib.metadata.version("custom_ops")}
    )
except Exception:
    pass
