## __all__ is a dict:
##   key is model_name in `model_zoo/chatglm-xx.json`
##   value is vendor specify model impl
# __all__ = {
#     "chatglm" : ChatGLMForConditionalGeneration,
#     "chatglm2" : ChatGLM2ForConditionalGeneration
# }

from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch_tpu

from .tpu_llama import TPULlama

from llm_perf.utils.logger import logger

__all__ = {
    "llama3.1": TPULlama,
}