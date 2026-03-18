import pathlib
from transformers import Qwen3MoeConfig
from ..utils import CommonModelConfig

def trans_model_config(
    src_config: Qwen3MoeConfig
) -> CommonModelConfig:
    dst_config: CommonModelConfig = CommonModelConfig()

    # TODO: 虽然 qwen3-235b-a22b 没有 dense layer, 但是还是需要加上必要的判断
    dst_config.num_layers = [
        src_config.num_hidden_layers
    ]

    return dst_config


model_configs = {}
for sub_dir in pathlib.Path(__file__).parent.iterdir():
    if sub_dir.is_dir() and sub_dir.joinpath("config.json").exists():
        model_configs[sub_dir.name] = {}
        
        src_config = Qwen3MoeConfig.from_pretrained(sub_dir)
        dst_config = trans_model_config(src_config)
        
        model_configs[sub_dir.name]["common"] = dst_config
        model_configs[sub_dir.name]["source"] = src_config