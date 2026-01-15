import os
import re
import sys
import math
import time
import json
import copy
import random
import psutil
import pathlib
import platform
import traceback
import importlib
import prettytable
from datetime import timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Any


import torch
import torch.distributed as dist
import torch.multiprocessing as mp

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger, get_numa_info


# collect all backends OP_MAPPING
SUPPORTED_OPS = {
    # xccl_ops
    "all_reduce": {"default_engine": "XCCLEngine"},
    "reduce_scatter": {"default_engine": "XCCLEngine"},
    "all_gather": {"default_engine": "XCCLEngine"},
    "all_to_all": {"default_engine": "XCCLEngine"},
    "broadcast": {"default_engine": "XCCLEngine"},
    "p2p": {"default_engine": "P2PEngine"},

    "all_reduce_h2d": {"default_engine": "XCCLEngine"}, 

    "host2device": {"default_engine": "XCCLEngine"},
    "device2host": {"default_engine": "XCCLEngine"},
    "device2device": {"default_engine": "ComputeEngine"},

    # vector_linear_ops
    "add": {"default_engine": "ComputeEngine"},
    "sub": {"default_engine": "ComputeEngine"},
    "mul": {"default_engine": "ComputeEngine"},
    "cast": {"default_engine": "ComputeEngine"},

    # vector_sfu_ops
    "div": {"default_engine": "ComputeEngine"},
    "sin": {"default_engine": "ComputeEngine"},
    "cos": {"default_engine": "ComputeEngine"},
    "exp": {"default_engine": "ComputeEngine"},
    "log": {"default_engine": "ComputeEngine"},
    "sqrt": {"default_engine": "ComputeEngine"},

    # vector_reduction_ops
    "reduce_max": {"default_engine": "ComputeEngine"},
    "reduce_min": {"default_engine": "ComputeEngine"},
    "reduce_sum": {"default_engine": "ComputeEngine"},
    "topk": {"default_engine": "ComputeEngine"},

    # vector_norm_ops
    "layer_norm": {"default_engine": "ComputeEngine"},
    "rms_norm": {"default_engine": "ComputeEngine"},
    "softmax": {"default_engine": "ComputeEngine"},

    # vector_activation_ops
    "gelu": {"default_engine": "ComputeEngine"},
    "silu": {"default_engine": "ComputeEngine"},

    # vector_index_ops
    "sort": {"default_engine": "ComputeEngine"},
    "embedding": {"default_engine": "ComputeEngine"},
    "gather": {"default_engine": "ComputeEngine"},
    "index_select": {"default_engine": "ComputeEngine"},
    "scatter": {"default_engine": "ComputeEngine"},
    "index_add": {"default_engine": "ComputeEngine"},

    # tensor_gemm_ops
    "gemm": {"default_engine": "ComputeEngine"},

    # llm: basic
    "scale_dynamic_quant": {"default_engine": "ComputeEngine"},
    "add_rms_norm_dynamic_quant": {"default_engine": "ComputeEngine"},
    "add_rms_norm": {"default_engine": "ComputeEngine"},

    # llm: MOE
    "moe_gating_gemm": {"default_engine": "ComputeEngine"},
    "moe_softmax_topk": {"default_engine": "ComputeEngine"},
    "moe_scatter_dynamic_quant": {"default_engine": "ComputeEngine"},
    "quant_matmul": {"default_engine": "ComputeEngine"},
    "moe_quant_group_gemm": {"default_engine": "ComputeEngine"},
    "moe_swiglu_dynamic_quant": {"default_engine": "ComputeEngine"},
    "swiglu_dynamic_quant": {"default_engine": "ComputeEngine"},
    "moe_gather": {"default_engine": "ComputeEngine"},

    # llm: ATTN
    "head_rms_norm": {"default_engine": "ComputeEngine"},
    "head_rms_norm_dynamic_quant": {"default_engine": "ComputeEngine"},
    "rotary_embedding": {"default_engine": "ComputeEngine"},
    "store_kv_cache": {"default_engine": "ComputeEngine"},
    "dequant_kv_cache": {"default_engine": "ComputeEngine"},
    "flash_attention": {"default_engine": "ComputeEngine"},

}





class Backend(ABC):
    def __init__(self):
        self.numa_world_size = 1
        self.numa_rank = 0

        self.common_info = {}   # 系统的基本信息
        self.backend_info = {}  # backend相关信息
        self.default_envs = {}  # 特定backend特定device_name的默认环境变量
        self.override_envs = {} # 预先设置的环境变量，会覆盖默认环境变量
        self.provider_info = {} # 特定backend可以提供的算子provider
        
        # 获取必要的 backend 信息
        self.backend_info = self.get_backend_info()

        # 获取当前backend可以提供的算子provider
        self.provider_info = self.get_provider_info()

        # 获取必要的 env 信息
        self.default_envs = self.get_default_envs()
        self.override_envs = self.process_envs()

        # 获取系统的基本信息
        self.common_info = self.get_common_info()


        self.op_mapping = {}


    """
    获取系统的基本信息
    """
    def get_common_info(self):
        # CPU架构
        cpu_arch = ''
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            cpu_arch = "x86_64"
        elif machine in ("aarch64", "arm64"):
            cpu_arch = "aarch64"
        else:
            cpu_arch = machine

        # python版本
        python_version = platform.python_version()

        # numa配置
        numa_config_str, numa_configs = get_numa_info()

        return {
            "cpu_arch": cpu_arch,
            "python_version": python_version,
            "numa_config_str": numa_config_str, 
            "numa_configs": numa_configs
        }


    """
    获取和backend相关的信息
    """
    @abstractmethod
    def get_backend_info(self):
        raise NotImplementedError

    
    def get_provider_info(self):
        return {}

    """
    获取和backend相关的环境变量
    """
    def get_default_envs(self):
        return {}

        
    def process_envs(self):
        override_envs = {}
        if type(self.default_envs) != dict:
            raise ValueError(f"default_envs must be a dict, but got {type(self.default_envs)}")

        for env, val in self.default_envs.items():
            if env in os.environ:
                override_envs[env] = os.environ[env]
            else:
                os.environ[env] = val
        return override_envs


    """
    backend加载支持的算子的provider
    """
    def load_all_ops(self):
        """
        op0:
            provider0: ComputeEngine
            provider1: ComputeEngine
        op1:
            provider0: XCCLEngine
        
        """
        self.op_mapping = {}
        for op_name, op_config in SUPPORTED_OPS.items():
            default_engine = op_config["default_engine"]

            try:
                backend_ops = importlib.import_module(f"backends.{self.backend_type}.ops.{op_name}")
                op_providers = getattr(backend_ops, "OP_MAPPING")

                self.op_mapping[op_name] = {}

                for provider_name, op_cls in op_providers.items():
                    self.op_mapping[op_name][provider_name] = {
                        "op_cls": op_cls,
                        "engine_name": default_engine
                    }

            except Exception as e:
                logger.warning(f"load op {op_name} failed, error: {e}")
                continue


    """
    清楚与backend相关的bench过程中产生的results
    """
    def clean_extra_files(self):
        pass


    """
    device management related
    """
    @abstractmethod
    def get_torch_device_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_device_name(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_device_properties(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_mem_info(self, index: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_device_count(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_device(self, index: int):
        raise NotImplementedError
    
    @abstractmethod
    def get_device(self):
        raise NotImplementedError
    
    @abstractmethod
    def device_synchronize(self):
        raise NotImplementedError
    
    @abstractmethod
    def empty_cache(self):
        raise NotImplementedError






    
    """
    ccl related
    """
    @abstractmethod
    def get_dist_module(self):
        raise NotImplementedError

    @abstractmethod
    def get_dist_backend(self):
        raise NotImplementedError
    

    def initialize_ccl(self, rank: int, world_size: int):
        dist_module = self.get_dist_module()
        dist_backend_name = self.get_dist_backend()
        
        dist_module.init_process_group(
            backend=dist_backend_name,
            world_size=world_size,
            rank=rank, 
            timeout=timedelta(seconds=1800)
        )

        assigned_value = 1 if rank < world_size // 2 else -1
        data = torch.ones([1], dtype=torch.float32, device=self.get_torch_device_name()) * assigned_value
        dist_module.all_reduce(data, op=dist_module.ReduceOp.SUM)
        print(data)

        return True

    def new_group(self, ranks):
        dist_module = self.get_dist_module()
        
        if dist_module.is_initialized():
            return dist_module.new_group(ranks)
        else:
            return None


    def op_group_barrier(self, op_group=None, group_size=1):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized() and group_size > 1:
            dist_module.all_reduce(
                torch.tensor([1], dtype=torch.int32, device=self.get_torch_device_name()),
                op=dist_module.ReduceOp.SUM,
                group=op_group
            )

    def destroy_process_group(self):
        dist_module = self.get_dist_module()
        if dist_module.is_initialized():
            dist_module.destroy_process_group()
    


    def core_perf(
        self, op_instance, 
        warmup_iterations, prefer_iterations, tensor_list, 
        profiling=True
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        # nessary sync for host2device, device2host test
        self.op_group_barrier(op_group=op_group, group_size=group_size)
        self.device_synchronize()

        for i in range(warmup_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        self.device_synchronize()
        start_time = time.perf_counter_ns()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        self.device_synchronize()
        end_time = time.perf_counter_ns()

        latency_us = (end_time - start_time) / 1e3 / prefer_iterations
        return latency_us, []


    
    def perf(self, op_instance, profiling=True):        
        # op
        tensor_size = op_instance.tensor_size

        # device
        device_mem_info = self.get_mem_info()
        avail_memory = device_mem_info[0]

        # assume
        assume_avail_bytes = int(avail_memory * 0.9)
        assume_cache_size = 1 * (1024 ** 3)

        # preset return values
        latency_us = 0.
        kernel_mapping = {}

        try:
            min_test_iters = 10
            sleep_time = 0.2
            max_test_time = 1e6
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
                        math.floor(assume_cache_size / tensor_size)
                    )

            tensor_list = op_instance.create_tensors(max_data_cnt)
            random.shuffle(tensor_list)

            latency_us, _ = self.core_perf(op_instance, 2, 2, tensor_list, profiling=False)
            prefer_iters = min(max(int(max_test_time / latency_us), 2), min_test_iters)
            if op_instance.group_size > 1:
                dist_module = self.get_dist_module()
                prefer_iters_list = [None for _ in range(op_instance.group_size)]
                dist_module.all_gather_object(prefer_iters_list, prefer_iters, group=op_instance.op_group)
                prefer_iters = max(prefer_iters_list)
            time.sleep(sleep_time)
            latency_us, kernel_mapping = self.core_perf(op_instance, 2, prefer_iters, tensor_list, profiling=profiling)

            del tensor_list
            self.empty_cache()
        except Exception as e:
            traceback.print_exc()

        return op_instance.summary(latency_us, kernel_mapping)


    def fake_perf(self, group_size, op_group):
        if group_size > 1:
            dist_module = self.get_dist_module()

            self.op_group_barrier(op_group=op_group, group_size=group_size)

            prefer_iters_list = [None for _ in range(group_size)]
            dist_module.all_gather_object(prefer_iters_list, 0, group=op_group)

            self.op_group_barrier(op_group=op_group, group_size=group_size)


    def compute_infer_loop(
        self, 
        local_process_rank: int, 
        process_mapping: List[Dict[str, Any]],
        input_queue : mp.Queue,
        output_queue : mp.Queue
    ):
        cur_process_info = process_mapping[local_process_rank]

        # set cpu affinity
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cur_process_info["numa_cores"])

        # set device
        local_device_id = cur_process_info["device_id"]
        self.set_device(local_device_id)

        output_queue.put("success")

        while True:
            data = input_queue.get()
            if data is None:
                break

            case_idx: int = data[0]
            op_name, op_provider, op_cls = data[1]
            task_case : Dict[str, Any] = data[2]


            result_dict = {}

            op_instance = None
            try:
                op_instance = op_cls(task_case, self)
                op_instance.is_concurrent = False
            except Exception as e:
                print(f"Failed to create op {op_name} with provider {op_provider} with args {task_case} with error {e}")
                output_queue.put((case_idx, result_dict))
                continue

            try:
                target_dict = self.perf(op_instance, profiling=self.enable_profiling)
            except Exception as e:
                print(f"Failed to run op {op_name} with provider {op_provider} with args {task_case} with error {e}")
                output_queue.put((case_idx, result_dict))
                continue

            if target_dict:
                arguments_str = json.dumps(task_case)
                targets_str = json.dumps(target_dict, indent=4)

                pt = prettytable.PrettyTable()
                pt.field_names = ["key", "value"]
                pt.align = "l"
                pt.add_row(["op_name", op_name])
                pt.add_row(["op_provider", op_provider])
                pt.add_row(["rank", str(local_process_rank)])
                pt.add_row(["device_id", str(local_device_id)])
                pt.add_row(["idx", str(case_idx)])

                print(f"{pt}\n{arguments_str}\n{targets_str}\n")

                result_dict = {
                    "provider": op_provider,
                    "arguments": copy.deepcopy(task_case),
                    "targets": target_dict
                }

            output_queue.put((case_idx, result_dict))
        

        

    
    def xccl_infer_loop(
        self, 
        local_process_rank: int, 
        process_mapping: List[Dict[str, Any]],
        input_queue : mp.Queue,
        output_queue : mp.Queue, 
        master_addr: str,
        xccl_port: int, 
        node_world_size: int, 
        node_rank: int,
    ):
        cur_process_info = process_mapping[local_process_rank]

        # set cpu affinity
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cur_process_info["numa_cores"])

        # set device
        local_device_id = cur_process_info["device_id"]
        self.set_device(local_device_id)


        local_world_size = len(process_mapping)
        local_rank = local_process_rank
        world_size = local_world_size * node_world_size
        rank = local_rank + node_rank * local_world_size

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(xccl_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        print(f"device_id: {local_device_id}, local: {local_rank}/{local_world_size}, world: {rank}/{world_size}")

        if world_size > 1:
            self.initialize_ccl(rank, world_size)
            dist.barrier()
        
        if local_process_rank == 0:
            output_queue.put("success")

        dist_group_mapping = {}
        dist_group_mapping[world_size] = None

        while True:
            # 只有 rank0 在等待数据
            if rank == 0:
                data = input_queue.get()
            else:
                data = None
                
            exchange_area = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(
                    exchange_area, 
                    {"rank": rank, "data": data}
                )
            else:
                exchange_area[0] = {"rank": rank, "data": data}
            sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])

            data = sorted_exchange_area[0]["data"]
            if data is None:
                break
            
            case_idx: int = data[0]
            op_name, op_provider, op_cls = data[1]
            task_case : Dict[str, Any] = data[2]


            result_dict = {}
            
            task_world_size = task_case.get("world_size", 1)
            if task_world_size > world_size:
                if rank == 0:
                    output_queue.put((case_idx, result_dict))
                continue



            if task_world_size > 1 and task_world_size not in dist_group_mapping:
                dist_group_mapping[task_world_size] = dist.new_group(ranks=list(range(task_world_size)))

            # create op instancce on all target devices
            op_instance = None
            if rank < task_world_size:
                try:
                    op_instance = op_cls(
                        task_case, self, 
                        op_group=dist_group_mapping.get(task_world_size, None), 
                        group_size=task_world_size
                    )
                    op_instance.is_concurrent = True
                except Exception as e:
                    traceback.print_exc()

            # create current exchange area for all cooperative devices
            # maybe on different node or numa
            # check whether needed devices have created op instance
            exchange_area = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(
                    exchange_area, 
                    {"rank": rank, "result": op_instance is not None}
                )
            else:
                exchange_area[0] = {"rank": rank, "result": op_instance is not None}
            
            sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])[:task_world_size]
            if not all([x["result"] for x in sorted_exchange_area]):
                if rank == 0:
                    output_queue.put((case_idx, result_dict))
                continue
                

            # according to given task_world_size, 
            # some devices work, some devices sleep
            target_dict = {}
            if rank < task_world_size:
                try:
                    target_dict = self.perf(op_instance, profiling=True)
                except Exception as e:
                    traceback.print_exc()
            
            exchange_area = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(
                    exchange_area, 
                    {"rank": rank, "device_id": local_device_id, "target_dict": target_dict}
                )
            else:
                exchange_area[0] = {"rank": rank, "device_id": local_device_id, "target_dict": target_dict}
            sorted_exchange_area = sorted(exchange_area, key=lambda x: x["rank"])[:task_world_size]


            target_dict_list = [x["target_dict"] for x in sorted_exchange_area]
            if not all(target_dict_list):
                if rank == 0:
                    output_queue.put((case_idx, result_dict))
                continue
            rank_list = [x["rank"] for x in sorted_exchange_area]
            device_id_list = [x["device_id"] for x in sorted_exchange_area]


            if rank == 0:
                new_target_dict = op_instance.merge_summary(target_dict_list)
                arguments_str = json.dumps(task_case)
                targets_str = json.dumps(new_target_dict, indent=4)

                pt = prettytable.PrettyTable()
                pt.field_names = ["key", "value"]
                pt.align = "l"
                pt.add_row(["op_name", op_name])
                pt.add_row(["op_provider", op_provider])
                pt.add_row(["rank", str(rank_list)])
                pt.add_row(["device_id", str(device_id_list)])
                pt.add_row(["idx", str(case_idx)])

                print(f"{pt}\n{arguments_str}\n{targets_str}\n")

                result_dict = {
                    "provider": op_provider, 
                    "device_id": device_id_list,
                    "arguments": copy.deepcopy(task_case),
                    "targets": new_target_dict,
                }
                output_queue.put((case_idx, result_dict))
            
        if world_size > 1:
            dist.destroy_process_group()
            

    


        
