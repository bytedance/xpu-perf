import pathlib
from transformers import Qwen3Config
from ..utils import CommonModelConfig

def trans_model_config(
    src_config: Qwen3Config
) -> CommonModelConfig:
    dst_config: CommonModelConfig = CommonModelConfig()

    dst_config.num_layers = [
        src_config.num_hidden_layers
    ]

    return dst_config


model_configs = {}
for sub_dir in pathlib.Path(__file__).parent.iterdir():
    if sub_dir.is_dir() and sub_dir.joinpath("config.json").exists():
        model_configs[sub_dir.name] = {}
        
        src_config = Qwen3Config.from_pretrained(sub_dir)
        dst_config = trans_model_config(src_config)
        
        model_configs[sub_dir.name]["common"] = dst_config
        model_configs[sub_dir.name]["source"] = src_config
