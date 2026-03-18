import os
import time
import random
import threading
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, AsyncIterable, Dict, Iterable, List, Tuple, Union

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

from llm_perf.core.generation import (
    GenerateConfig,
    GenerateRequest,
    GenerateResult
)
from llm_perf.core.inferencer import CoreInferencer
from llm_perf.core.sampler import CoreSampler
from llm_perf.utils.logger import logger
from llm_perf.utils.reporter import calc_perplexity


class CoreScheduler(ABC):
    def __init__(
        self,
        inferencer: CoreInferencer,
        sampler: CoreSampler,
        task_cls=CoreInferencer.Task
    ) -> None:
        super().__init__()

        self.inferencer: CoreInferencer = inferencer
        self.sampler: CoreSampler = sampler

        self.Task = task_cls
        self.task_queue: Queue[self.Task] = Queue()

        self.started = False
        self.scheduler_thread = None


    def start(self):
        if not self.started:
            logger.info("start scheduler thread")
            self.started = True
            self.scheduler_thread = threading.Thread(target=self.scheduler_loop)
            self.scheduler_thread.start()

    def stop(self):
        if self.started:
            logger.info("stop scheduler thread")
            self.started = False
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=1.)


    @abstractmethod
    @torch.no_grad()
    def scheduler_loop(self):
        raise NotImplementedError

    async def generate(
        self, 
        req: GenerateRequest
    ) -> Union[
        AsyncIterable[GenerateResult], Tuple[AsyncIterable[GenerateResult], float, str]
    ]:
        task = self.Task(request=req)
        self.submit(task)

        async for result in self.get_packet_results(
            req.generate_config.get_input_logits, task
        ):
            yield result


    def submit(self, task):
        self.task_queue.put_nowait(task)


    async def get_packet_results(
        self, 
        get_input_logits: bool, 
        task: CoreInferencer.Task
    ):

        gen_index = 0

        while True:
            result = await task.get_result()
            if result is None:
                if task.exception:
                    raise task.exception
                break
            
            cur_input_tokens = task.request.input_ids + task.generate_ids[:gen_index]
            gen_index += 1

            task_results = {
                "result": result, 
            }
            if get_input_logits:
                await self.update_logits(result, task)

                cur_labels_tensor = torch.tensor(
                    cur_input_tokens, 
                    dtype=torch.int64, device='cpu'
                )

                input_logits_len = len(cur_input_tokens) - 1
                input_logits = task.all_logits[:input_logits_len]

                perplexity = calc_perplexity(input_logits, cur_labels_tensor)

                task_results["dump_file"] = ""
                task_results["perplexity"] = perplexity

            yield task_results

        task_results = {
            "result": None, 
            "perplexity": -1, 
        }

        if get_input_logits:
            dump_file = await self.dump_last_logits(task)
            task_results["dump_file"] = dump_file
            yield task_results
        return
        

    async def update_logits(self, result, task):
        # [8, num_vocab]
        if not hasattr(task, "all_logits"):
            task.all_logits = result.logits
        # [1, num_vocab]
        else:
            task.all_logits = torch.cat([task.all_logits, result.logits], dim=0)



    async def dump_last_logits(self, task: CoreInferencer.Task):
        tmp_dir = ".tmp_logits"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        import numpy as np

        dump_file = (
            tmp_dir
            + "/"
            + str(random.randint(0, 100))
            + "_"
            + str(int(time.time_ns()))
            + ".npy"
        )
        input_tokens_len = len(task.request.input_ids)
        gen_tokens_len = len(task.generate_ids)
        generate_logits = task.all_logits[-gen_tokens_len:].unsqueeze(0)
        np.save(dump_file, generate_logits.numpy())
        return dump_file