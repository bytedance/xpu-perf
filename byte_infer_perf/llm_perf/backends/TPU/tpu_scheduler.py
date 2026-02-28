import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set

import torch
import torch_tpu

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.core.inferencer import CoreInferencer
from llm_perf.core.sampler import CoreSampler
from llm_perf.backends.TPU.tpu_inferencer import TpuInferencer
from llm_perf.utils.logger import logger

class TpuScheduler(CoreScheduler):
    def __init__(
        self,
        inferencer: CoreInferencer,
        sampler: CoreSampler,
        xpu_cfg
    ) -> None:
        super().__init__(inferencer, sampler)
        self.max_batch_size = xpu_cfg["max_batch_size"]
        
        
    @torch.inference_mode()
    def scheduler_loop(self):
        task_slots: List[CoreInferencer.Task] = [None] * self.max_batch_size
        avail_slots: List[int] = [self.max_batch_size - 1 - i for i in range(self.max_batch_size)]
        context_slots: List[int] = []

        while self.started:
            while not self.task_queue.empty():
                if len(avail_slots) == 0:
                    break
                slot = avail_slots.pop()
                task_slots[slot] = self.task_queue.get()
                context_slots.append(slot)

            if len(avail_slots) == self.max_batch_size:
                with self.task_queue.not_empty:
                    self.task_queue.not_empty.wait(0.1)
                continue


            # context phase
            if len(context_slots) != 0:
                # do inference --> logits
                select_slot = context_slots.pop(0)
                select_slots= [
                    select_slot
                ]

                cur_task = task_slots[select_slot]
                cur_tasks = [
                    cur_task
                ]

                cur_task.update_st("model_start")

                outputs = self.inferencer.infer(
                    cur_tasks, 
                    is_context=True, 
                    valid_slot_ids=select_slots
                )

                cur_task.update_st("model_end")

                # sample logits --> tokens
                next_tokens, _ = self.sampler.sample(
                    tasks=cur_tasks, 
                    logits=outputs["last_logits"]
                )

                cur_task.update_st("process_end")

                # postprocess -> gen result
                generation_results = self.sampler.postprocess(
                    tasks=cur_tasks,
                    infer_outputs=outputs,
                    next_tokens=next_tokens,
                )

                # add result to task
                cur_task.add_result(generation_results[0])
                if generation_results[0].finish_reason:
                    cur_task.finish()

            
            # decode phase
            else:
                select_slots = []
                valid_tasks = []
                for i, task in enumerate(task_slots):
                    if task is not None:
                        select_slots.append(i)
                        valid_tasks.append(task)

                for task in valid_tasks:
                    task.update_st("model_start")

                outputs = self.inferencer.infer(
                    valid_tasks, 
                    is_context=False, 
                    valid_slot_ids=select_slots
                )

                for task in valid_tasks:
                    task.update_st("model_end")


                # sample logits --> tokens
                next_tokens, _ = self.sampler.sample(
                    tasks=valid_tasks, 
                    logits=outputs["last_logits"]
                )

                for task in valid_tasks:
                    task.update_st("process_end")

                # postprocess -> gen result
                generation_results = self.sampler.postprocess(
                    tasks=valid_tasks,
                    infer_outputs=outputs,
                    next_tokens=next_tokens,
                )

                # add result to task
                for i, gen_res in enumerate(generation_results):
                    valid_tasks[i].add_result(gen_res)
                    if gen_res.finish_reason:
                        valid_tasks[i].finish()
                    
            for i, task in enumerate(task_slots):
                if task is not None and task.is_finished():
                    avail_slots.append(i)
                    task_slots[i] = None
            
            avail_slots.sort(reverse=True)