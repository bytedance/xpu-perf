import importlib.metadata

from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "dcu_flash_attn"

# Mirror GPU vendor + DCU environment package probes
try:
    import flash_attn  # noqa: F401

    ProviderRegistry.register_provider_info(
        "flash_attn_v2", {"flash_attn": importlib.metadata.version("flash_attn")}
    )
except Exception:
    pass
try:
    import flash_attn_interface  # noqa: F401

    ProviderRegistry.register_provider_info(
        "flash_attn_v3", {"flash_attn": importlib.metadata.version("flash_attn")}
    )
except Exception:
    pass
try:
    ProviderRegistry.register_provider_info(
        "vllm", {"vllm": importlib.metadata.version("vllm")}
    )
except Exception:
    pass
try:
    ProviderRegistry.register_provider_info(
        "flashinfer", {"flashinfer": importlib.metadata.version("flashinfer-python")}
    )
except Exception:
    pass
