import os
from typing import Dict, List, Any
from dataclasses import dataclass

from llm_perf.core.generation import GenerateRequest
from llm_perf.core.inferencer import CoreInferencer
from llm_perf.backends.ROCM.gpu_mp_engine import GpuMpEngine
from llm_perf.utils.logger import logger

class GpuInferencer(CoreInferencer):
    def __init__(self, model_impl, xpu_cfg):
        super().__init__()

        self.tp_size = xpu_cfg["tp_size"]
        self.pad_token_id = xpu_cfg["pad_token_id"]
        self.max_batch_size = xpu_cfg["max_batch_size"]
        self.mp_engine = GpuMpEngine(self.tp_size, model_impl, xpu_cfg)    

    def prepare_inputs(
        self, 
        tasks: List[CoreInferencer.Task], 
        **kwargs
    ):
        input_dict = {
            "input_ids": None, 
            "position_ids": None, 
            "attention_mask": None, 
            "all_q_len": None, 
            "all_kv_len": None, 
            "is_context": None, 
            "valid_slot_ids": None
        }

        is_context = kwargs.get("is_context") if "is_context" in kwargs.keys() else False
        valid_slot_ids = kwargs.get("valid_slot_ids") if "valid_slot_ids" in kwargs.keys() else [i for i in range(self.max_batch_size)]
    

        get_input_logits = False
        for task in tasks:
            if task.request.generate_config.get_input_logits:
                get_input_logits = True
                break

        input_dict["is_context"] = is_context
        input_dict["valid_slot_ids"] = valid_slot_ids
        input_dict["get_input_logits"] = get_input_logits

        if is_context:
            q_len = len(tasks[0].request.input_ids)
            kv_len = len(tasks[0].request.input_ids)

            input_dict["input_ids"] = [
                tasks[0].request.input_ids
            ]
            input_dict["position_ids"] = [
                [i for i in range(q_len)]
            ]
            input_dict["attention_mask"] = [
                [1 for _ in range(q_len)]
            ]
            input_dict["all_q_len"] = [
                q_len
            ]
            input_dict["all_kv_len"] = [
                kv_len
            ]
        else:
            all_input_ids = []
            all_position_ids = []
            all_attention_mask = []
            all_q_len = []
            all_kv_len = []

            for task in tasks:
                q_len = 1
                kv_len = 0

                if task is None:
                    kv_len = 1

                    input_ids = [
                        self.pad_token_id
                    ]
                    position_ids = [
                        0
                    ]
                    attention_mask = [
                        0
                    ]
                else:
                    kv_len = len(task.request.input_ids) + len(task.generate_ids) - 1

                    input_ids = [
                        task.generate_ids[-1]
                    ]
                    position_ids = [
                        kv_len
                    ]
                    attention_mask = [
                        1
                    ]
                all_input_ids.append(input_ids)
                all_position_ids.append(position_ids)
                all_attention_mask.append(attention_mask)
                all_q_len.append(q_len)
                all_kv_len.append(kv_len)

            input_dict["input_ids"] = all_input_ids
            input_dict["position_ids"] = all_position_ids
            input_dict["attention_mask"] = all_attention_mask
            input_dict["all_q_len"] = all_q_len
            input_dict["all_kv_len"] = all_kv_len

        return input_dict


    def infer(
        self, 
        tasks: List[CoreInferencer.Task],  
        **kwargs
    ):
        input_dict = self.prepare_inputs(tasks, **kwargs)
        output_dict = self.mp_engine.mp_forward(input_dict)
        
        logits = output_dict["logits"]
        next_token_logits = logits[:, -1, :].contiguous()
        infer_outputs = {
            "logits": logits, 
            "last_logits": next_token_logits
        }
        return infer_outputs
