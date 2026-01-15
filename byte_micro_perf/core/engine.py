import os
import sys
import pathlib
import traceback
import prettytable
from typing import List, Dict, Any
from datetime import timedelta
from abc import ABC, abstractmethod


FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
from core.utils import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class BaseEngine(ABC):
    def __init__(self, args_dict):
        self.args_dict = args_dict

        self.backend_instance: Backend = args_dict["backend_instance"]

        self.backend_instance.enable_profiling = args_dict.get("enable_profiling", False)


        self.device_name = self.backend_instance.backend_info["device_name"]
        self.device_count = self.backend_instance.backend_info["device_count"]

        self.numa_configs = self.backend_instance.common_info["numa_configs"]


        # 获取当前的并行方式
        self.node_world_size = args_dict.get("node_world_size", 1)
        self.node_rank = args_dict.get("node_rank", 0)

        self.master_addr = args_dict["master_addr"]
        self.host_port = args_dict["host_port"]
        self.device_port = args_dict["device_port"]


        # 根据numa配置决定要起多少个
        self.numa_num = args_dict.get("numa_num", 1)
        self.numa_order = args_dict.get("numa_order", [-1])
        self.device_mapping = args_dict.get("device_mapping", {})
        self.device_ids = args_dict.get("device_ids", [])
        
        self.device_num = len(self.device_ids)


        # 每一个numa process对应1个device, 不会创建多余的process
        if self.device_num <= self.numa_num:
            self.numa_num = self.device_num
            self.numa_order = self.numa_order[:self.numa_num]
            self.device_num_per_process = 1
        # 每一个numa process对应多个device, 且device数相等
        elif self.device_num % self.numa_num == 0:
            self.device_num_per_process = self.device_num // self.numa_num
        else:
            logger.error(f"device_num({self.device_num}) must be divisible by numa_num({self.numa_num})")



        self.process_mapping = []
        for process_id in range(self.device_num):
            device_id = self.device_ids[process_id]
            numa_id = self.numa_order[process_id // self.device_num_per_process]            
            numa_cores = self.numa_configs[numa_id] if numa_id != -1 else []
            self.process_mapping.append({
                "device_id": device_id,
                "numa_id": numa_id,
                "numa_cores": numa_cores,
            })
        
        # 进程和进程号
        self.subprocess_procs = []
        self.subprocess_pids = []

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()


    def __del__(self):
        if self.node_world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        self.stop()


    def init_host_dist_env(self):
        # 如果使用多个node参与计算，需要初始化host端通信
        if self.node_world_size > 1:
            os.environ["WORLD_SIZE"] = str(self.node_world_size)
            os.environ["RANK"] = str(self.node_rank)
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = self.host_port

            dist.init_process_group(backend="gloo")


    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    def dispatch(self, tasks: Dict[str, List[Dict[str, Any]]]):
        task_idx = 0
        for task_info, task_cases in tasks.items():
            logger.info(f"dispatch task {task_info} with {len(task_cases)} test cases")
            for test_case in task_cases:
                self.input_queue.put((task_idx, task_info, test_case))
                task_idx += 1

        all_results = {}
        for _ in range(task_idx):
            result_idx, result_dict = self.output_queue.get()
            if result_dict:
                all_results[result_idx] = result_dict

        results = {}
        sort_idx = 0
        for task_info, task_cases in tasks.items():
            cur_task_results = []
            for test_case in task_cases:
                if sort_idx not in all_results:
                    logger.error(f"missing result for task_info {task_info} test case {test_case}")
                else:
                    cur_task_results.append(all_results[sort_idx])
                sort_idx += 1

            if cur_task_results:
                results[task_info] = cur_task_results    

        return results




class ComputeEngine(BaseEngine):
    def __init__(self, args_dict):
        super().__init__(args_dict)

    def start(self):
        # 创建子进程, 子进程各自独立汇报状态
        try:
            self.subprocess_procs = mp.spawn(
                self.backend_instance.compute_infer_loop, 
                args=(
                    self.process_mapping, 
                    self.input_queue, 
                    self.output_queue,
                ), 
                nprocs=self.device_num,
                join=False, 
                daemon=False
            )
            for proc in self.subprocess_procs.processes:
                self.subprocess_pids.append(proc.pid)
            logger.info(f"spawn compute infer loop success, pids: {self.subprocess_pids}")

            for _ in range(self.device_num):
                try:
                    signal = self.output_queue.get(timeout=30)
                    if signal != "success":
                        logger.error(f"compute infer loop failed, signal: {signal}")
                        sys.exit(-1)
                except Exception as e:
                    logger.error(f"compute infer loop timeout, error: {e}")
                    sys.exit(-1)

            logger.info(f"all subprocesses are ready")

        except Exception as e:
            logger.exception(f"failed to spawn compute infer loop: {e}")
            traceback.print_exc()
            sys.exit(-1)
            


    def stop(self):
        if self.subprocess_procs:
            for _ in self.subprocess_procs.processes:
                self.input_queue.put(None)

            kill_flag = False
            for subprocess in self.subprocess_procs.processes:
                subprocess.join(timeout=10)
                if subprocess.is_alive():
                    kill_flag = True
                    break
            
            if kill_flag:
                for subprocess in self.subprocess_procs.processes:
                    subprocess.kill()

            self.subprocess_procs = []
            self.subprocess_pids = []


class XCCLEngine(BaseEngine):
    def __init__(self, args_dict):
        super().__init__(args_dict)

        os.environ["MASTER_ADDR"] = str(self.master_addr)
        os.environ["MASTER_PORT"] = str(self.device_port)

        self.node_world_size = args_dict["node_world_size"]
        self.node_rank = args_dict["node_rank"]

    def start(self):
        # 创建子进程, 由一个子进程汇报状态
        try:
            self.subprocess_procs = mp.spawn(
                self.backend_instance.xccl_infer_loop, 
                args=(
                    self.process_mapping, 
                    self.input_queue, 
                    self.output_queue,
                    self.master_addr, 
                    self.device_port, 
                    self.node_world_size, 
                    self.node_rank,
                ), 
                nprocs=self.device_num,
                join=False, 
                daemon=False
            )
            for proc in self.subprocess_procs.processes:
                self.subprocess_pids.append(proc.pid)
            logger.info(f"spawn xccl infer loop success, pids: {self.subprocess_pids}")

            try:
                signal = self.output_queue.get(timeout=30)
                if signal != "success":
                    logger.error(f"xccl infer loop failed, signal: {signal}")
                    sys.exit(-1)
            except Exception as e:
                logger.error(f"xccl infer loop timeout, error: {e}")
                sys.exit(-1)

            logger.info(f"all subprocesses are ready")

        except Exception as e:
            logger.exception(f"failed to spawn xccl infer loop: {e}")
            traceback.print_exc()
            sys.exit(-1)
            

    def stop(self):
        if self.subprocess_procs:
            if self.node_rank == 0:
                self.input_queue.put(None)

            kill_flag = False
            for subprocess in self.subprocess_procs.processes:
                subprocess.join(timeout=10)
                if subprocess.is_alive():
                    kill_flag = True
                    break
            
            if kill_flag:
                for subprocess in self.subprocess_procs.processes:
                    subprocess.kill()

            self.subprocess_procs = []
            self.subprocess_pids = []
            

class P2PEngine(BaseEngine):
    def __init__(self, args_dict):
        super().__init__(args_dict)
