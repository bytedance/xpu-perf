import os
import sys
import csv
import json
import pathlib
import random
import shutil
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
from core.utils import suppress_stdout_stderr
try:
    from backends.GPU.provider_gpu import GPU_PROVIDER
except:
    GPU_PROVIDER = {}
try:
    from backends.GPU.env_gpu import GPU_ENV
except:
    GPU_ENV = {}

from backends.GPU.provider_gpu import GPU_PROVIDER



class BackendGPU(Backend):
    def __init__(self):
        super().__init__()

    def get_backend_info(self):
        info_dict = {}

        # device相关
        info_dict["device_name"] = torch.cuda.get_device_name(0)
        info_dict["device_count"] = torch.cuda.device_count()

        device_properties = torch.cuda.get_device_properties(0)
        info_dict["device_memory_mb"] = device_properties.total_memory / (1024 ** 2)
        


        __torch_version = torch.__version__
        __cuda_version = torch.version.cuda
        __driver_version = ''
        nvidia_smi_output = subprocess.run(
            ['nvidia-smi', '-q', '-i', '0'], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        for line in nvidia_smi_output.stdout.split('\n'):
            if 'Driver Version' in line:
                __driver_version = line.split(':')[1].strip()
                break

        info_dict["torch_version"] = __torch_version
        info_dict["torch_cuda_version"] = __cuda_version
        info_dict["driver_version"] = __driver_version

        return info_dict
    
    def get_default_envs(self):
        return GPU_ENV.get(torch.cuda.get_device_name(0), {})
    
    def get_provider_info(self):
        return GPU_PROVIDER

    def clean_extra_files(self):
        PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling")
        if PROFILER_DIR.exists():
            shutil.rmtree(PROFILER_DIR)

        



    """
    device management related
    """
    def get_torch_device_name(self):
        return "cuda"
    
    def get_device_name(self, index = 0):
        return torch.cuda.get_device_name(index)
    
    def get_device_properties(self, index = 0):
        return torch.cuda.get_device_properties(index)

    def get_mem_info(self, index = 0):
        total_memory = torch.cuda.get_device_properties(index).total_memory
        allocated_memory = torch.cuda.memory_allocated(index)
        cached_memory = torch.cuda.memory_reserved(index)
        free_memory = (total_memory - allocated_memory)
        return (free_memory, total_memory)

    def get_device_count(self):
        device_count = torch.cuda.device_count()
        return device_count, list(range(device_count))
    
    def set_device(self, device_index : int):
        torch.cuda.set_device(device_index)

    def get_device(self):
        return torch.cuda.current_device()

    def device_synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()





    """
    ccl related
    """
    def get_dist_module(self):
        return dist
    
    def get_dist_backend(self):
        return "nccl"




    def core_perf(
        self, op_instance, 
        warmup_iterations, prefer_iterations, 
        tensor_list, 
        profiling=True
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        if not op_instance.is_concurrent and profiling:
            process_id = os.getpid()
            PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling", f"{process_id}")
            PROFILER_DIR.mkdir(parents=True, exist_ok=True)
            TRACE_FILE = PROFILER_DIR.joinpath("trace.json")

            # profiling
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], 
                schedule=torch.profiler.schedule(
                    wait=0, 
                    warmup=warmup_iterations, 
                    active=prefer_iterations, 
                    repeat=1
                ), 
                on_trace_ready=lambda prof: prof.export_chrome_trace(str(TRACE_FILE))
            ) as prof:
                for i in range(prefer_iterations + warmup_iterations):
                    op_instance.core_run(tensor_list[i % len(tensor_list)])
                    self.device_synchronize()
                    prof.step()

            # parse and delete profiling json file
            average_latency = 0.
            kernel_latency_list = {}
            if PROFILER_DIR.exists():
                json_files = list(PROFILER_DIR.glob("*.json"))
                if json_files:
                    profiling_data = json.load(open(json_files[0]))
                    for event in profiling_data["traceEvents"]:
                        if event.get("cat", None) in ["kernel", "gpu_memcpy"]:
                            kernel_name = event["name"]
                            kernel_latency = event["dur"]
                            if kernel_name not in kernel_latency_list:
                                kernel_latency_list[kernel_name] = []
                            kernel_latency_list[kernel_name].append(kernel_latency)

                    take_iters = prefer_iterations // 2
                    iters_offset = prefer_iterations - take_iters

                    removed_keys = []
                    for kernel in kernel_latency_list:
                        if len(kernel_latency_list[kernel]) != prefer_iterations:
                            removed_keys.append(kernel)
                        average_latency += sum(kernel_latency_list[kernel][iters_offset:])
                    for kernel in removed_keys:
                        kernel_latency_list.pop(kernel)

                    average_latency /= take_iters
                TRACE_FILE.unlink()
            return average_latency, list(kernel_latency_list.keys())
        
        else:
            for i in range(warmup_iterations):
                index = random.randint(0, len(tensor_list) - 1)
                op_instance.core_run(tensor_list[index])
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            self.device_synchronize()
            self.op_group_barrier(op_group=op_group, group_size=group_size)
            start_event.record()
            for i in range(prefer_iterations):
                op_instance.core_run(tensor_list[i % len(tensor_list)])
            end_event.record()
            end_event.synchronize()

            latency_us = start_event.elapsed_time(end_event) * 1e3 / prefer_iterations
            return latency_us, []
