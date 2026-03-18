### Meaning of name of json_file
Take mixtral-torch-bf16-8x22b as an example: 
- **mixtral** means model type, there may be other variances of model type.
- **torch** means model framework, we will only use torch as infer framework and prioritize using huggingface model format.
- **bf16** means source data type of model weights.
- **8x22b** means model size, 8 means experts num, 22b means model size for each expert.

### How to create new json_file
Every json file defines a individual llm model, and the name of json file will be a unique identifier.
Each json file contains following info: 
- model_name: **mixtral** for example.
- model_path: path to model, default to "llm_perf/model_zoo/sota/`repo_name`"
- model_interface: model module name.
- tokenizer: tokenizer info.
- network: model config.