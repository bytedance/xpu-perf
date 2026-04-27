import json
import os
import pathlib
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

from xpu_perf.micro_perf.core.backend import Backend


class BackendDCU(Backend):
    def __init__(
        self,
        backend,
        env_file=None,
        op_defs: Optional[pathlib.Path] = None,
        vendor_ops: Optional[List[pathlib.Path]] = None,
        **kwargs,
    ):
        if vendor_ops is None:
            vendor_ops = []
        super().__init__(
            backend=backend,
            env_file=env_file,
            op_defs=op_defs,
            vendor_ops=vendor_ops,
            **kwargs,
        )

    def get_backend_info(self):
        info_dict = {}
        device_name = torch.cuda.get_device_name(0)
        info_dict["device_name"] = device_name
        info_dict["device_count"] = torch.cuda.device_count()
        device_properties = torch.cuda.get_device_properties(0)
        info_dict["device_memory_mb"] = device_properties.total_memory / (1024**2)
        backend_env = self.get_backend_env()
        info_dict["torch_version"] = backend_env.get("torch", "")
        info_dict["torch_cuda_version"] = getattr(torch.version, "hip", None) or getattr(
            torch.version, "cuda", ""
        )
        info_dict["dtk_version"] = backend_env.get("dtk_version", "")
        info_dict["driver_version"] = backend_env.get("driver", "")
        return info_dict

    def perf(self, op_instance):
        """
        Keep upstream perf mechanism, but use legacy DCU iteration policy:
        fixed 32 iterations (for stable profiling/measurement) with a longer target window.
        """
        import math
        import time
        import traceback

        tensor_size = op_instance.tensor_size
        avail_memory = self.get_mem_info()[0]

        assume_avail_bytes = int(avail_memory * 0.9)
        assume_cache_size = 1 * (1024**3)

        latency_us = 0.0
        kernel_mapping = {}

        try:
            min_test_iters = 32
            max_test_iters = 32
            max_test_time = 1e6  # 1s in us
            max_data_cnt = 1
            if not op_instance.is_concurrent:
                if tensor_size > assume_avail_bytes:
                    raise RuntimeError("Not enough memory to run the op")
                elif 2 * tensor_size > assume_avail_bytes:
                    max_data_cnt = 1
                elif tensor_size > assume_cache_size:
                    max_data_cnt = 2
                else:
                    max_data_cnt = min(
                        math.floor(max(assume_avail_bytes, assume_cache_size) / tensor_size),
                        math.floor(assume_cache_size / tensor_size),
                    )

            tensor_list = op_instance.create_tensors(max_data_cnt)
            random.shuffle(tensor_list)

            latency_us, _ = self.core_perf(op_instance, 2, 2, tensor_list, profiling=False)
            prefer_iters = min(max(math.ceil(max_test_time / latency_us), min_test_iters), max_test_iters)

            if op_instance.group_size > 1:
                dist_module = self.get_dist_module()
                prefer_iters_list = [None for _ in range(op_instance.group_size)]
                dist_module.all_gather_object(prefer_iters_list, prefer_iters, group=op_instance.op_group)
                prefer_iters = max(prefer_iters_list)

            time.sleep(0.2)

            # DCU policy: enable profiling by default, unless explicitly disabled per-case.
            # This keeps upstream's vendor-controlled flag available as an override.
            require_profiling = op_instance.args_dict.get("require_profiling", True)
            op_instance.require_profiling = bool(require_profiling)
            actual_profiling = self.enable_profiling and bool(require_profiling)
            latency_us, kernel_mapping = self.core_perf(
                op_instance, 2, prefer_iters, tensor_list, profiling=actual_profiling
            )

            del tensor_list
            self.empty_cache()
        except Exception:
            traceback.print_exc()

        return op_instance.summary(latency_us, kernel_mapping)

    def clean_extra_files(self):
        prof_dir = pathlib.Path.cwd().joinpath("profiling")
        if prof_dir.exists() and not getattr(self, "keep_traces", False):
            shutil.rmtree(prof_dir)

    def get_torch_device_name(self):
        return "cuda"

    def get_device_name(self, index=0):
        return torch.cuda.get_device_name(index)

    def get_device_properties(self, index=0):
        return torch.cuda.get_device_properties(index)

    def get_mem_info(self, index=0):
        total_memory = torch.cuda.get_device_properties(index).total_memory
        allocated_memory = torch.cuda.memory_allocated(index)
        free_memory = total_memory - allocated_memory
        return (free_memory, total_memory)

    def get_device_count(self):
        device_count = torch.cuda.device_count()
        return device_count, list(range(device_count))

    def set_device(self, device_index: int):
        torch.cuda.set_device(device_index)

    def get_device(self):
        return torch.cuda.current_device()

    def device_synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()

    def get_rocm_version(self):
        hipcc_path = subprocess.run(
            ["which", "hipcc"], stdout=subprocess.PIPE, text=True
        ).stdout.strip()
        if not hipcc_path:
            return "N/A"
        dtk_root = str(Path(hipcc_path).parent.parent)
        version_path = os.path.join(dtk_root, ".info/rocm_version")
        try:
            with open(version_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return "N/A"

    def get_backend_env(self):
        __torch_version = torch.__version__
        __dtk_version = self.get_rocm_version()
        __driver_version = ""
        rocm_smi = subprocess.run(
            ["rocm-smi", "--showdriverversion"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        if rocm_smi.returncode == 0:
            for line in rocm_smi.stdout.split("\n"):
                if "Driver Version" in line:
                    __driver_version = line.split(":", 1)[1].strip()
                    break
        return {
            "torch": __torch_version,
            "dtk_version": __dtk_version,
            "driver": __driver_version,
        }

    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "nccl"

    def core_perf(
        self,
        op_instance,
        warmup_iterations,
        prefer_iterations,
        tensor_list,
        profiling=True,
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        if not op_instance.is_concurrent and profiling:
            process_id = os.getpid()
            prof_dir = pathlib.Path.cwd().joinpath("profiling", f"{process_id}")
            prof_dir.mkdir(parents=True, exist_ok=True)
            if getattr(self, "keep_traces", False):
                trace_file = prof_dir.joinpath(
                    f"trace_{op_instance.__class__.__name__}_{int(time.time() * 1000)}.json"
                )
            else:
                trace_file = prof_dir.joinpath("trace.json")

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=warmup_iterations,
                    active=prefer_iterations,
                    repeat=1,
                ),
                on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_file)),
            ) as prof:
                for i in range(prefer_iterations + warmup_iterations):
                    op_instance.core_run(tensor_list[i % len(tensor_list)])
                    self.device_synchronize()
                    prof.step()

            average_latency = 0.0
            kernel_latency_list = {}
            if prof_dir.exists():
                json_files = list(prof_dir.glob("*.json"))
                if json_files:
                    profiling_data = json.load(open(json_files[0], encoding="utf-8"))
                    for event in profiling_data.get("traceEvents", []):
                        if event.get("cat", None) in ["kernel", "gpu_memcpy"]:
                            kernel_name = event["name"]
                            kernel_latency = event["dur"]
                            kernel_latency_list.setdefault(kernel_name, []).append(
                                kernel_latency
                            )
                    take_iters = prefer_iterations // 2
                    iters_offset = prefer_iterations - take_iters
                    removed_keys = []
                    for kernel in list(kernel_latency_list.keys()):
                        if len(kernel_latency_list[kernel]) != prefer_iterations:
                            removed_keys.append(kernel)
                        else:
                            average_latency += sum(
                                kernel_latency_list[kernel][iters_offset:]
                            )
                    for k in removed_keys:
                        kernel_latency_list.pop(k, None)
                    if take_iters:
                        average_latency /= take_iters
                if not getattr(self, "keep_traces", False):
                    try:
                        trace_file.unlink()
                    except OSError:
                        pass
            return average_latency, list(kernel_latency_list.keys())

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
