# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company

import os
import json
import random
import logging
import subprocess
import pathlib
import shutil

from datetime import timedelta
import torch.distributed as dist

import torch

from core.backend import Backend

from dataclasses import dataclass

import habana_frameworks.torch as ht


@dataclass
class HPUDeviceProperties:
    total_memory: int


class BackendHPU(Backend):
    def __init__(self):
        super().__init__()

    def get_torch_device_name(self):
        return "hpu"

    def get_device_name(self, index=0):
        return "Gaudi2"

    def get_device_properties(self, index=0):
        # output of torch.hpu.get_device_properties() on Gaudi2:
        # '(sramBaseAddress=1153202979533225984, dramBaseAddress=1153203082662772736, sramSize=50331648, dramSize=102106132480, tpcEnabledMask=16777215, dramEnabled=1, fd=20, device_id=0, device_type=4)'
        dramSize = 102106132480
        return HPUDeviceProperties(dramSize)

    def get_mem_info(self, index=0):
        if hasattr(torch, 'hpu'):
            return torch.hpu.mem_get_info(index)
        else:
            return [self.get_device_properties().total_memory]*2

    def get_device_count(self):
        device_count = int(
            subprocess.check_output(
                "hl-smi -Q module_id -f csv,noheader | wc -l", shell=True, text=True
            )
        )
        return device_count, list(range(device_count))

    def set_device(self, device_index: int):
        try:
            os.environ["HLS_MODULE_ID"] = str(device_index)
        except Exception as e:
            print(str(e), flush=True)

        import habana_frameworks.torch as htorch
        torch.hpu.set_device("hpu:0")

    def get_device(self):
        return 0

    def device_synchronize(self):
        torch.hpu.synchronize()

    def empty_cache(self):
        if hasattr(torch, 'hpu'):
            torch.hpu.synchronize()
        return

    def get_backend_env(self):
        hl_smi_output= subprocess.run(
            ["hl-smi", "-Q", "driver_version", "-f", "csv,noheader"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        __driver_version = hl_smi_output.stdout.splitlines()[0]
        return {
            "torch": torch.__version__,
            "driver": __driver_version,
        }

    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "hccl"

    def core_perf(
        self, op_instance, 
        warmup_iterations, prefer_iterations, 
        tensor_list, 
        profiling=False
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        if profiling:
            raise NotImplementedError("Profiling mode is not supported for HPU")

        ht.core.mark_step()
        for i in range(warmup_iterations):
            index = random.randint(0, len(tensor_list) - 1)
            op_instance.core_run(tensor_list[index])
            ht.core.mark_step()
        start_event = torch.hpu.Event(enable_timing=True)
        end_event = torch.hpu.Event(enable_timing=True)

        self.device_synchronize()
        self.op_group_barrier(op_group=op_group, group_size=group_size)
        start_event.record()
        for i in range(prefer_iterations):
            _ = op_instance.core_run(tensor_list[i % len(tensor_list)])
            ht.core.mark_step()
        self.device_synchronize()
        end_event.record()
        end_event.synchronize()

        latency_us = start_event.elapsed_time(end_event) * 1e3 / prefer_iterations
        return latency_us, []
