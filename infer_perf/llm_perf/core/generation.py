import torch
import asyncio
from dataclasses import dataclass, field
from typing import List


@dataclass
class GenerateConfig:
    min_new_tokens: int = 0
    max_new_tokens: int = 0
    top_k: int = 0
    top_p: float = 1.0
    temperature: float = 1.0
    presence_penalty: float = 1.0
    eos_token_id: int = -1
    pad_token_id: int = -1
    get_input_logits: bool = False


@dataclass
class GenerateRequest:
    input_ids: List[int]
    generate_config: GenerateConfig


@dataclass
class GenerateResult:
    token_id: int
    finish_reason: str
    wait_time: float
    model_time: float
    post_process_time: float
    logits: torch.Tensor
    last_logits: torch.Tensor


class ResultQueue:
    def __init__(self):
        self._q = asyncio.Queue()
        try:
            self._loop = self._q._get_loop()
        except:
            self._loop = asyncio.get_running_loop()

    def put(self, item):
        self._loop.call_soon_threadsafe(self._q.put_nowait, item)

    async def get(self):
        return await self._q.get()
