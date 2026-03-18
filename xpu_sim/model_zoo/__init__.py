from .topology import OpTopologyDAG
from .utils import DistributionInfo


from .seed_oss import model_configs as seed_oss_configs
from .qwen3_dense import model_configs as qwen3_dense_configs
from .qwen3_moe import model_configs as qwen3_moe_configs


BASE_MODEL_MAPPING = {
    "seed_oss": seed_oss_configs,
    "qwen3_dense": qwen3_dense_configs, 
    "qwen3_moe": qwen3_moe_configs,
}

