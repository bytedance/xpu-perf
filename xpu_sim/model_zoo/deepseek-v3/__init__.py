import pathlib
from transformers import DeepseekV3Config


configs = {}
for sub_dir in pathlib.Path(__file__).parent.iterdir():
    if sub_dir.is_dir() and sub_dir.joinpath("config.json").exists():
        configs[sub_dir.name] = DeepseekV3Config.from_pretrained(sub_dir)
