import os
import re
import sys
import math
import time
import json
import copy
import ctypes
import random
import psutil
import signal
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
from core.op import ProviderRegistry
from core.ops import SUPPORTED_OPS, DEFAULT_OP_IMPL_MAPPING




def path_to_prefixed_import(file_path, base_path, prefix="my_project"):
    """
    将文件路径转换为带前缀的 python import 路径
    """
    try:
        # 1. 获取相对于 base_path 的路径
        rel_path = pathlib.Path(file_path).relative_to(base_path)
        
        # 2. 去掉后缀并切分路径组件
        # parts 如 ('models', 'llama') 或 ('__init__',)
        parts = list(rel_path.with_suffix('').parts)
        
        # 3. 处理 __init__.py：如果是包本身，去掉末尾的 __init__
        if parts and parts[-1] == "__init__":
            parts.pop()
        
        # 4. 组合内部路径
        internal_import = ".".join(parts)
        
        # 5. 拼接前缀
        if not internal_import:
            return prefix  # 刚好是 base_path 下的 __init__.py
        
        # 确保前缀不以点结尾，然后连接
        return f"{prefix.rstrip('.')}.{internal_import}"
        
    except ValueError:
        return "Path is not under the base directory"


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

        self.enable_profiling = False


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
            provider0: op0_cls0
            provider1: op0_cls1
        op1:
            provider0: op1_cls0
        
        """
        self.op_mapping = {}


        self.provider_mapping = {}





        """
        直接加载 backend/ops/ 下面的所有python文件,
        如果有旧版本的 OP_MAPPING, 则加载里面的 provider
        没有的话也会默认调用 ProviderRegistry 注册的各种实现
        """
        target_backend_ops_dir = pathlib.Path(__file__).parents[1].joinpath("backends", self.backend_type, "ops").absolute()
        for op_file in target_backend_ops_dir.rglob("*.py"):
            import_op_name = path_to_prefixed_import(op_file, target_backend_ops_dir, prefix=f"backends.{self.backend_type}.ops")
            op_name = op_file.stem

            target_op_module = importlib.import_module(import_op_name)
            # OLD: 如果有OP_MAPPING, 加载所有 provider
            if hasattr(target_op_module, "OP_MAPPING"):
                self.op_mapping[op_name] = {}
                op_providers = getattr(target_op_module, "OP_MAPPING")
                for provider_name, op_cls in op_providers.items():
                    self.op_mapping[op_name][provider_name] = {
                        "op_cls": op_cls
                    }

                    self.provider_mapping[provider_name] = self.provider_mapping.get(provider_name, {})
                    self.provider_mapping[provider_name][op_name] = op_cls

            

        for _, engine_ops in SUPPORTED_OPS.items():
            for op_name in engine_ops:
                self.op_mapping[op_name] = self.op_mapping.get(op_name, {})
                try:
                    # NEW: 如果通过 ProviderRegistry 注册
                    if op_name in ProviderRegistry.OP_MAPPING:
                        for provider_name, op_cls in ProviderRegistry.OP_MAPPING[op_name].items():
                            self.op_mapping[op_name][provider_name] = {
                                "op_cls": op_cls
                            }   
                            self.provider_mapping[provider_name] = self.provider_mapping.get(provider_name, {})
                            self.provider_mapping[provider_name][op_name] = op_cls
                except Exception as e:
                    logger.warning(f"load op {op_name} failed, error: {e}")

                if len(self.op_mapping[op_name]) == 0:
                    default_op = DEFAULT_OP_IMPL_MAPPING.get(op_name, None)
                    if default_op is None:
                        logger.warning(f"op {op_name} not supported by backend {self.backend_type}")
                        continue
                    self.op_mapping[op_name] = {
                        "torch": {
                            "op_cls": default_op
                        }
                    }

        for provider_name in self.provider_mapping.keys():
            print("\n")
            print(f"Provider: {provider_name}")
            pt = prettytable.PrettyTable()
            pt.field_names = ["op_name", "op_cls"]
            pt.align = "l"
            for op_name, op_cls in self.provider_mapping[provider_name].items():
                pt.add_row([op_name, op_cls])
            print(pt)
            print("\n")


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
        dist.init_process_group(
            backend=self.get_dist_backend(),
            init_method="env://", 
            timeout=timedelta(seconds=1800)
        )
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



    def perf(self, op_instance):        
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

            actual_profiling = self.enable_profiling and op_instance.require_profiling
            latency_us, kernel_mapping = self.core_perf(op_instance, 2, prefer_iters, tensor_list, profiling=actual_profiling)

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
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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

            arguments_str = json.dumps(task_case)
            pt = prettytable.PrettyTable()
            pt.field_names = ["key", "value"]
            pt.align = "l"
            pt.add_row(["op_name", op_name])
            pt.add_row(["op_provider", op_provider])
            pt.add_row(["rank", str(local_process_rank)])
            pt.add_row(["device_id", str(local_device_id)])
            pt.add_row(["idx", str(case_idx)])


            op_instance = None
            try:
                op_instance = op_cls(task_case, self)
                op_instance.is_concurrent = False
            except Exception as e:
                print(f"{pt}\n{arguments_str}")
                print(f"Failed to create op {op_name} with provider {op_provider} with args {task_case} with error {e}")
                traceback.print_exc()
                print("")
                output_queue.put((case_idx, result_dict))
                continue

            try:
                target_dict = self.perf(op_instance)
            except Exception as e:
                print(f"{pt}\n{arguments_str}")
                print(f"Failed to run op {op_name} with provider {op_provider} with args {task_case} with error {e}")
                traceback.print_exc()
                print("")
                output_queue.put((case_idx, result_dict))
                continue

            if target_dict:
                targets_str = json.dumps(target_dict, indent=4)
                print(f"{pt}\n{arguments_str}\n{targets_str}\n")
                result_dict = target_dict

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
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cur_process_info = process_mapping[local_process_rank]

        # set cpu affinity
        proc = psutil.Process(os.getpid())
        proc.cpu_affinity(cur_process_info["numa_cores"])


        # dist env
        local_device_id = cur_process_info["device_id"]
        local_world_size = len(process_mapping)
        local_rank = local_process_rank
        world_size = local_world_size * node_world_size
        rank = local_rank + node_rank * local_world_size

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(xccl_port)
        
        print(f"device_id: {local_device_id}, local: {local_rank}/{local_world_size}, world: {rank}/{world_size}")

        if world_size > 1:
            self.initialize_ccl(rank, world_size)
            print(f"xccl init done, rank: {rank}, world_size: {world_size}")
            self.set_device(local_device_id)
            assigned_value = 1 if rank < world_size // 2 else -1
            data = torch.ones([1], dtype=torch.float32, device=self.get_torch_device_name()) * assigned_value
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            print(data)
            cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        else:
            self.set_device(local_device_id)

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
                    {"rank": rank, "data": data}, 
                    group=cpu_group,
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
                    {"rank": rank, "result": op_instance is not None}, 
                    group=cpu_group,
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
                    target_dict = self.perf(op_instance)
                except Exception as e:
                    traceback.print_exc()
            
            exchange_area = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(
                    exchange_area, 
                    {"rank": rank, "device_id": local_device_id, "target_dict": target_dict}, 
                    group=cpu_group
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

                result_dict = new_target_dict
                output_queue.put((case_idx, result_dict))
            
        if world_size > 1:
            dist.destroy_process_group()
    
