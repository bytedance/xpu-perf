import os
import sys
import csv
import json
import random
import string
import shutil
import hashlib
import pathlib
import datetime
import argparse
import platform
import requests
import importlib
import subprocess
import prettytable
from functools import partial
from typing import Dict, List


import torch

from utils import load_dir_as_module, get_func_from_file
from model_zoo import BASE_MODEL_MAPPING


# rename device name
device_name_mapping = {
    "MTT S5000": {
        "formatted_name": "S5000", 
        "die_num_per_card": 1, 
        "extra_info": "S5000, x86_64, 单机8DIE"
    }
}


CWD_DIR = pathlib.Path.cwd().absolute()
FILE_DIR = pathlib.Path(__file__).parent.absolute()

DEFAULT_BYTEMLPERF_DIR = FILE_DIR.parents[1].joinpath("byte_micro_perf")
DEFAULT_LAUNCH_SCRIPT = DEFAULT_BYTEMLPERF_DIR.joinpath("launch.py")


DEFAULT_DATABASE_DIR = FILE_DIR.joinpath("database")
DEFAULT_MODELZOO_DIR = FILE_DIR.joinpath("model_zoo")
DEFAULT_WORKSPACE_DIR = FILE_DIR.joinpath("workspaces")
DEFAULT_REPORT_DIR = FILE_DIR.joinpath("reports")


def get_unique_id():
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    combined = dt_str + random_str
    hash_obj = hashlib.md5(combined.encode())  # 使用MD5哈希
    hash_str = hash_obj.hexdigest()[:8]  # 取前8位哈希值
    return f"{dt_str}_{hash_str}"



def get_info_template(url) -> Dict:
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {}

def normal_bench_template(input_dict: Dict, url) -> Dict:
    try:
        response = requests.post(url, json={
            "type": "normal", 
            "data": input_dict
        })
        return response.json()
    except Exception as e:
        return {}



def print_server_info(info_dict):
    # common info
    common_pt = prettytable.PrettyTable()
    common_pt.field_names = ["attr", "value"]
    common_pt.align = "l"
    for attr, value in info_dict["common"].items():
        if attr == "numa_configs":
            continue
        else:
            common_pt.add_row([attr, value])    
    print(common_pt)

    # provider info
    provider_pt = prettytable.PrettyTable()
    provider_pt.field_names = ["provider", "version"]
    provider_pt.align = "l"
    for provider, version in info_dict["provider"].items():
        provider_pt.add_row([provider, version])
    print(provider_pt)

    # backend info
    backend_pt = prettytable.PrettyTable()
    backend_pt.field_names = ["attr", "value"]
    backend_pt.align = "l"
    for attr, value in info_dict["backend"].items():
        backend_pt.add_row([attr, value])
    print(backend_pt)

    # runtime info
    runtime_pt = prettytable.PrettyTable()
    runtime_pt.field_names = ["attr", "value"]
    runtime_pt.align = "l"
    for attr, value in info_dict["runtime"].items():
        runtime_pt.add_row([attr, value])
    print(runtime_pt)

    print("\n\n\n")



class XpuPerfSimEngine:
    def __init__(self, ip, port, bench_config_path, **kwargs):
        # 服务信息
        self.ip = ip
        self.port = port
        self.info_url = f"http://{self.ip}:{self.port}/info"
        self.bench_url = f"http://{self.ip}:{self.port}/bench"
        self.get_info_func = partial(get_info_template, self.info_url)
        self.bench_func = partial(normal_bench_template, self.bench_url)
    
        # detect server info
        self.detect_server_info()

        # bench_config.json
        if not bench_config_path.exists():
            raise FileNotFoundError(f"bench_config.json not found in {bench_config_path}")
        self.bench_config = json.loads(bench_config_path.read_text())

        # seed_oss / seed-oss-36b
        self.base_model_name = self.bench_config["base_model_name"]
        self.model_name = self.bench_config["model_name"]


        # 获取 bench配置 和 模型deploy的template
        self.deploy_dir = bench_config_path.parent.resolve()
        self.generate_template_path = self.deploy_dir.joinpath(
            self.bench_config["template"]
        ).resolve()
        if not self.generate_template_path.exists():
            raise FileNotFoundError(f"template {self.generate_template_path} not found")
        

        self.model_config_values = BASE_MODEL_MAPPING[self.base_model_name][self.model_name]
        self.model_config = self.model_config_values["common"]
        self.src_model_config = self.model_config_values["source"]

        generate_func = get_func_from_file(self.generate_template_path, "generate")
        self.model_topo = generate_func(
            self.src_model_config, 
            self.bench_config,
        )
        
        # 解析模型
        self.parse_model(**kwargs)

        self.model_topo.print_topo_pretty()


        """
        {
            (batch_size, cache_len, q_len): {
                "bench_info": {}, 
                "perf_info": {
                    "total_latency": 端到端耗时
                    "total_node_latency": 节点总耗时
                    "breakdown": {
                        (instance_name, instance_index): {
                            provider_name: {arguments}
                        }
                    }                
                }
            }
        }
        """
        self.fix_data_dict = {}

        """
        [
            {
                bench_info": {}, 
                "perf_info": {
                    "total_latency": 端到端耗时
                    "total_node_latency": 节点总耗时
                    "breakdown": {
                        (instance_name, instance_index): {
                            provider_name: {arguments}
                        }
                    }                
                }
            }
        ]
        """
        self.var_data_dict = []

        self.create_engine()


    def detect_server_info(self):
        self.info_dict = self.get_info_func()
        print_server_info(self.info_dict)

        self.backend = self.info_dict["backend_type"]

        self.old_device_name = self.info_dict["backend"]["device_name"]
        device_info = device_name_mapping.get(
            self.old_device_name, 
            {
                "formatted_name": self.old_device_name,
                "die_num_per_card": 1,
                "extra_info": "",
            }
        )
        self.device_name = device_info["formatted_name"]
        self.extra_info = device_info["extra_info"]
        self.path_device_name = self.device_name.replace(" ", "_")
        self.actual_device_num = self.info_dict["backend"]["device_count"]
        self.card_device_ratio = device_info["die_num_per_card"]
        self.actual_card_num = self.actual_device_num // self.card_device_ratio



    def create_engine(self, **kwargs):
        self.prepare_workspace(**kwargs)


    def prepare_workspace(self, **kwargs):
        self.cur_datetime = get_unique_id()
        self.pwd_dir = pathlib.Path.cwd()

        # 放置当前运行的所有数据
        workspace_path = kwargs.get("workspace_path", str(DEFAULT_WORKSPACE_DIR))
        self.workspace_dir = pathlib.Path(workspace_path).absolute()

        # 放置每次生成的workloads的数据
        self.workload_dir = self.workspace_dir.joinpath("workloads")
        if self.workload_dir.exists():
            shutil.rmtree(self.workload_dir)
        self.workload_dir.mkdir(parents=True)

        # 放置每次workloads的实测结果
        self.result_dir = self.workspace_dir.joinpath("results")
        if self.result_dir.exists():
            shutil.rmtree(self.result_dir)
        self.result_dir.mkdir(parents=True)

    
    def dump_extra_files(self, target_dir: pathlib.Path):
        # save workloads
        shutil.copytree(self.workload_dir, target_dir.joinpath("workloads"))
        
        """
        save breakdown
        - perf.json
        - layers.json
        - head_non_layers.json
        - end_non_layers.json
        """
        breakdown_dir = target_dir.joinpath("summary")
        breakdown_dir.mkdir()

        for (batch_size, cache_len, q_len), breakdown_dict in self.breakdown_data.items():
            summary_dir = breakdown_dir.joinpath(f"bs_{batch_size}.cache_{cache_len}.q_{q_len}")
            summary_dir.mkdir()


            head_non_layers_kernel_num = 0
            end_non_layers_kernel_num = 0
            per_layer_kernel_num = len(breakdown_dict)
            layers_kernel_num = per_layer_kernel_num * self.num_layers
            total_kernel_num = layers_kernel_num
            
        
            bubble_latency = 0.
            head_non_layers_latency = 0.
            end_non_layers_latency = 0.


            bubble_ratio = 0.
            head_non_layers_ratio = 0.
            end_non_layers_ratio = 0.
            layers_ratio = 0.


            head_non_layers_formatted_data = []
            end_non_layers_formatted_data = []
            layers_formatted_data = []

            per_layer_latency = 0.
            for (instance_name, instance_index), occur_data in breakdown_dict.items():
                per_layer_latency += occur_data["targets"].get("latency(us)", 0)
            layers_latency = per_layer_latency * self.num_layers
            model_latency = layers_latency
            


            for (instance_name, instance_index), occur_data in breakdown_dict.items():
                latency_us = occur_data["targets"].get("latency(us)", 0)
                kernel_provider = occur_data.get("provider", "unknown")
                kernel_mem_bw = occur_data["targets"].get("mem_bw(GB/s)", 0)
                kernel_comm_bw = occur_data["targets"].get("bus_bw(GB/s)", 0)
                kernel_flops = occur_data["targets"].get("calc_flops_power(tflops)", 0)
                
                kernel_ratio = latency_us / per_layer_latency
                formatted_data = {
                    "kernel_type": "layers",
                    "kernel_name": instance_name,
                    "kernel_provider": kernel_provider, 
                    "kernel_index": instance_index,
                    "kernel_occurs": self.num_layers,
                    # "kernel_occur_indices": [per_layer_kernel_num * layer_id + instance_index for layer_id in range(self.num_layers)],
                    "kernel_occur_indices": [], 
                    "latency": round(latency_us, 3),
                    # "latency_list": [latency_us] * self.num_layers,
                    "latency_list": [], 
                    "kernel_ratio": kernel_ratio,
                    "kernel_ratio_str": f"{kernel_ratio * 100:.2f}%",
                    "kernel_mem_bw": kernel_mem_bw, 
                    "kernel_comm_bw": kernel_comm_bw, 
                    "kernel_flops": kernel_flops, 
                }
                layers_formatted_data.append(formatted_data)

            fixed_formatted_data = {
                "ori_num_layers": self.ori_num_layers,
                "num_layers": self.num_layers,
                "total_kernel_num": total_kernel_num, 
                "head_non_layers_kernel_num": head_non_layers_kernel_num,
                "end_non_layers_kernel_num": end_non_layers_kernel_num,
                "layers_kernel_num": layers_kernel_num,
                

                "per_layer_kernel_num": per_layer_kernel_num,
                "model_latency": round(model_latency, 3),
                "all_kernels_latency": round(model_latency, 3),
                "bubble_latency": round(bubble_latency, 3), 
                "head_non_layers_latency": round(head_non_layers_latency, 3),
                "end_non_layers_latency": round(end_non_layers_latency, 3),
                "layers_latency": round(layers_latency, 3),
                "per_layer_latency": round(per_layer_latency, 3),
                
                
                "bubble_ratio": bubble_ratio,
                "head_non_layers_ratio": head_non_layers_ratio,
                "end_non_layers_ratio": end_non_layers_ratio,
                "layers_ratio": layers_ratio,
                

                "bubble_ratio_str": f"{bubble_ratio * 100:.2f}%",
                "head_non_layers_ratio_str": f"{head_non_layers_ratio * 100:.2f}%",
                "end_non_layers_ratio_str": f"{end_non_layers_ratio * 100:.2f}%",
                "layers_ratio_str": f"{layers_ratio * 100:.2f}%",
            }

            with open(summary_dir.joinpath("perf.json"), "w") as f:
                json.dump(fixed_formatted_data, f, indent=4)
            with open(summary_dir.joinpath("head_non_layers.json"), "w") as f:
                json.dump(head_non_layers_formatted_data, f, indent=4)
            with open(summary_dir.joinpath("end_non_layers.json"), "w") as f:
                json.dump(end_non_layers_formatted_data, f, indent=4)
            with open(summary_dir.joinpath("layers.json"), "w") as f:
                json.dump(layers_formatted_data, f, indent=4)

            with open(summary_dir.joinpath("head_non_layers.csv"), "w") as f:
                csv_writer = csv.DictWriter(f, fieldnames=["kernel_name", "latency", "kernel_ratio", "kernel_ratio_str"])
                for data in head_non_layers_formatted_data:
                    csv_writer.writerow(
                        {
                            "kernel_name": data["kernel_name"], 
                            "latency": data["latency"], 
                            "kernel_ratio": data["kernel_ratio"],
                            "kernel_ratio_str": data["kernel_ratio_str"],
                        }
                    )
            with open(summary_dir.joinpath("end_non_layers.csv"), "w") as f:
                csv_writer = csv.DictWriter(f, fieldnames=["kernel_name", "latency", "kernel_ratio", "kernel_ratio_str"])
                for data in end_non_layers_formatted_data:
                    csv_writer.writerow(
                        {
                            "kernel_name": data["kernel_name"], 
                            "latency": data["latency"], 
                            "kernel_ratio": data["kernel_ratio"],
                            "kernel_ratio_str": data["kernel_ratio_str"],
                        }
                    )
            with open(summary_dir.joinpath("layers.csv"), "w") as f:
                csv_writer = csv.DictWriter(f, fieldnames=["kernel_name", "latency", "kernel_ratio", "kernel_ratio_str"])
                for data in layers_formatted_data:
                    csv_writer.writerow(
                        {
                            "kernel_name": data["kernel_name"], 
                            "latency": data["latency"], 
                            "kernel_ratio": data["kernel_ratio"],
                            "kernel_ratio_str": data["kernel_ratio_str"],
                        }
                    )

            pt = prettytable.PrettyTable()
            pt.align = "l"
            pt.field_names = ["kernel_name", "kernel_provider", "num_occurs", "aver_latency (us)", "kernel_flops", "kernel_mem_bw", "kernel_comm_bw", "layer kernel (us)"]
            for data in layers_formatted_data:
                pt.add_row([data["kernel_name"], data["kernel_provider"], self.num_layers, data["latency"], data["kernel_flops"], data["kernel_mem_bw"], data["kernel_comm_bw"], per_layer_latency])
            print(pt)


    def parse_model(self, **kwargs):
        # 解析当前 bench 配置
        self.model_name = self.bench_config.get("model_name", "")
        self.infer_dtype = self.bench_config.get("infer_dtype", "")
        if self.model_name == "" or self.infer_dtype == "":
            raise ValueError("model_name or infer_dtype is empty")
        
        # 解析并行方式
        parallel_config = self.bench_config.get("parallel_config", {})
        self.deploy_node_num = 1    # 暂时只支持单node
        self.deploy_device_num = parallel_config.get("device_num", 1)
        if self.actual_device_num < self.deploy_device_num:
            raise ValueError(f"actual_device_num: {self.actual_device_num} < device_num: {self.deploy_device_num}")
        self.deploy_card_num = self.deploy_device_num // self.card_device_ratio


        """
        pp: 不同卡串行执行, 切分num_layers
        dp: 不同卡并行执行, 切分batch_size
        sp: 不同卡并行执行, 切分num_tokens
        tp: 不同卡并行执行, 切分K/N/head
        ep: 不同卡并行执行, 切分experts

        目前常用的并行方式:
        - PP8
        - PP16
        - PP8-TP2-EP2
        - TP8-EP8
        - SP8-TP8-EP8
        - DP16-EP16
        """
        self.dp_size = parallel_config.get("dp_size", 1)
        self.pp_size = parallel_config.get("pp_size", 1)
        self.sp_size = parallel_config.get("sp_size", 1)
        self.tp_size = parallel_config.get("tp_size", 1)
        self.ep_size = parallel_config.get("ep_size", 1)
        
        if self.deploy_node_num > 1:
            parallel_config_str = f"{self.deploy_node_num}N{self.deploy_card_num}C{self.deploy_device_num}D"
        else:
            parallel_config_str = f"{self.deploy_card_num}C{self.deploy_device_num}D"

        if self.dp_size > 1:
            parallel_config_str += f"_DP{self.dp_size}"
        if self.pp_size > 1:
            parallel_config_str += f"_PP{self.pp_size}"
        if self.sp_size > 1:
            parallel_config_str += f"_SP{self.sp_size}"
        if self.tp_size > 1:
            parallel_config_str += f"_TP{self.tp_size}"
        if self.ep_size > 1:
            parallel_config_str += f"_EP{self.ep_size}"
        self.parallel_config_str = parallel_config_str


        # 获取 bench_mode
        self.run_mode = kwargs.get("run_mode", "prefill")
        self.ori_num_layers = self.model_config.num_layers[0]
        self.test_num_layers = self.ori_num_layers
        num_layers = self.ori_num_layers
        num_mirror_layers = self.model_config.num_mirror_layers
        if self.run_mode == "prefill":
            num_layers -= num_mirror_layers
            self.test_num_layers -= num_mirror_layers

        # 根据 pp_size 调整层数和 xpu_config
        if self.pp_size > 1:
            num_layers = (num_layers + self.pp_size - 1) // self.pp_size

        self.num_layers = num_layers


        # 打印当前配置项
        pt = prettytable.PrettyTable(["attr", "value"])
        pt.align = "l"

        # 基础模型配置
        pt.add_row(["model_name", self.model_name])
        pt.add_row(["infer_dtype", self.infer_dtype])
        pt.add_row(["parallel_config", self.parallel_config_str])
        pt.add_row(["run_mode", self.run_mode])
        pt.add_row(["-" * 25, "-" * 50])

        # 并行策略
        pt.add_row(["deploy_node_num", self.deploy_node_num])
        pt.add_row(["deploy_card_num", self.deploy_card_num])
        pt.add_row(["deploy_device_num", self.deploy_device_num])
        pt.add_row(["dp_size", self.dp_size])
        pt.add_row(["pp_size", self.pp_size])
        pt.add_row(["sp_size", self.sp_size])
        pt.add_row(["tp_size", self.tp_size])
        pt.add_row(["ep_size", self.ep_size])

        # 模型配置
        pt.add_row(["ori_num_layers", self.ori_num_layers])
        pt.add_row(["test_num_layers", self.test_num_layers])
        pt.add_row(["num_layers", num_layers])
        print(pt)
        print("")


    def execute(self, execute_info):
        bench_mode = execute_info.get("bench_mode", "fix")
        bench_info = {}

        if bench_mode == "fix":
            # fix input
            batch_size = execute_info["batch_size"]
            cache_len = execute_info["cache_len"]
            q_len = execute_info["q_len"]

            # optional common input
            block_size = execute_info.get("block_size", 512)
            slot_mapping = execute_info.get("slot_mapping", list(range(batch_size)))
            if block_size > 0:
                kv_len = cache_len + q_len
                max_num_blocks_per_seq = (kv_len + block_size - 1) // block_size
                default_block_table = []
                for seq_id in range(batch_size):
                    start_block_id = seq_id * max_num_blocks_per_seq
                    
                    default_block_table.append(
                        [start_block_id + block_idx for block_idx in range(max_num_blocks_per_seq)]
                    )
            else:
                default_block_table = []
            block_table = execute_info.get("block_table", default_block_table)


            bench_info = {
                # basic info
                "batch_size": batch_size, 
                "cache_len": cache_len, 
                "q_len": q_len, 

                # attn info
                "run_mode": self.run_mode, 
                "block_size": block_size, 
                "slot_mapping": slot_mapping, 
                "block_table": block_table
            }
        elif bench_mode == "var":
            # batch input
            batch_size = execute_info["batch_size"]
            cache_lens = execute_info["cache_lens"]
            q_lens = execute_info["q_lens"]

            # optional common input
            block_size = execute_info.get("block_size", 512)
            slot_mapping = execute_info.get("slot_mapping", list(range(batch_size)))
            if block_size > 0:
                kv_lens = [cache_len + q_len for cache_len, q_len in zip(cache_lens, q_lens)]
                num_blocks_per_seq = [
                    (kv_len + block_size - 1) // block_size for kv_len in kv_lens
                ]
                max_num_blocks_per_seq = max(num_blocks_per_seq)
                default_block_table = [[-1] * max_num_blocks_per_seq for _ in range(batch_size)]
                for seq_id in range(batch_size):
                    start_block_id = seq_id * max_num_blocks_per_seq
                    for block_idx in range(num_blocks_per_seq[seq_id]):
                        default_block_table[seq_id][block_idx] = start_block_id + block_idx
            else:
                default_block_table = []
            block_table = execute_info.get("block_table", default_block_table)
                    
            bench_info = {
                # basic info
                "batch_size": batch_size, 
                "cache_lens": cache_lens, 
                "q_lens": q_lens, 

                # attn info
                "run_mode": self.run_mode, 
                "block_size": block_size, 
                "slot_mapping": slot_mapping, 
                "block_table": block_table
            }

        # set num_tokens
        self.model_topo.set_bench_info(bench_info)
        bench_results = self.send_bench_request(self.model_topo.op_dict)
        parsed_results = self.model_topo.parse_results(bench_results)

        # get e2e latency
        node_times, critical_path, total_latency, node_cost, node_provider, total_node_cost, critical_total = self.model_topo.calculate_timeline(parsed_results)
        self.model_topo.print_schedule(parsed_results)

        temp_pt = prettytable.PrettyTable(["instance_name", "instance_index", "provider", "latency(us)", "tflops", "mem_bw", "comm_bw"])
        temp_pt.align = "l"
        for (instance_name, instance_index), result_info in parsed_results.items():
            provider = list(result_info.keys())[0]
            provider_result = result_info[provider]

            tflops_value = provider_result.get("calc_flops_power(tflops)", 0)
            mem_bw_value = provider_result.get("mem_bw(GB/s)", 0)
            bus_bw_value = provider_result.get("bus_bw(GB/s)", 0)

            temp_pt.add_row([
                instance_name, instance_index, 
                provider, provider_result["latency(us)"], 
                tflops_value if tflops_value > 0 else "",
                mem_bw_value if mem_bw_value > 0 else "",
                bus_bw_value if bus_bw_value > 0 else ""])
        print(temp_pt)

        
        if bench_mode == "fix":
            item_key = (batch_size, cache_len, q_len)
            self.fix_data_dict[item_key] = {
                "bench_info": bench_info, 
                "perf_info": {
                    "e2e_latency": round(total_latency * self.num_layers / 1000, 3), 
                    "total_latency": total_latency,
                    "total_node_latency": total_node_cost,
                    "breakdown": parsed_results
                }
            }
            return item_key
        elif bench_mode == "var":
            item_key = len(self.var_data_dict)

            self.var_data_dict.append({
                "bench_info": bench_info, 
                "perf_info": {
                    "e2e_latency": round(total_latency * self.num_layers / 1000, 3), 
                    "total_latency": total_latency,
                    "total_node_latency": total_node_cost,
                    "breakdown": parsed_results
                }
            })
            return item_key
        else:
            raise ValueError(f"bench_mode {bench_mode} is not supported")
            


    def get_export_info(self, bench_mode):
        target_report_dir = DEFAULT_REPORT_DIR.joinpath(
            self.path_device_name, 
            self.model_name, 
            self.parallel_config_str, 
            self.infer_dtype, 
            bench_mode
        )
        config_dict = {
            "device_name": self.device_name,
            "model_name": self.model_name,
            "impl_framework": "bytemlperf_sim", 
            "infer_dtype": self.infer_dtype,
            "deploy_config_str": self.parallel_config_str,
            "deploy_config": {
                "node_num": self.deploy_node_num, 
                "card_num": self.deploy_card_num, 
                "device_num": self.deploy_device_num, 
                "pp_size": self.pp_size,
                "sp_size": self.sp_size,
                "tp_size": self.tp_size,
                "ep_size": self.ep_size
            }, 
            "bench_mode": bench_mode, 
            "envs": {}, 
            "extra_info": {
                "test_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        return target_report_dir, config_dict


    def send_bench_request(self, workloads):
        try:
            response = requests.post(self.bench_url, json={
                "type": "normal", 
                "data": workloads, 
            })
            return response.json()
        except Exception as e:
            empty_results = {}
            for op_name in workloads:
                empty_results[op_name] = []
                for _ in workloads[op_name]:
                    empty_results[op_name].append({})
            return empty_results





    def fix_bench_func(self, batch_size=1, cache_len=0, q_len=4096, block_size=512):
        """
        fix模式支持的参数:
        - batch_size: 当前推理的独立seq数
        - cache_len: 当前推理所有seq的cache_len, cache_len + q_len = kv_len
        - q_len: 当前推理的所有seq的q_len
        """
        pt = prettytable.PrettyTable()
        pt.align = "l"
        pt.field_names = ["key", "value"]
        pt.add_row(["batch_size", batch_size])
        pt.add_row(["cache_len", cache_len])
        pt.add_row(["q_len", q_len])
        print(pt)

        execute_info = {
            "bench_mode": "fix", 
            "batch_size": batch_size, 
            "cache_len": cache_len, 
            "q_len": q_len, 
            "block_size": block_size
        }
        print(f"sending fix input: {execute_info}")
        self.execute(execute_info)


    def var_bench_func(self, cache_lens, q_lens, block_size=512):
        """
        var模式支持的参数:
        - cache_lens: 所有seq的cache_len列表
        - q_lens: 所有seq的q_len列表
        """
        batch_size = min(len(cache_lens), len(q_lens))
        cache_lens = cache_lens[:batch_size]
        q_lens = q_lens[:batch_size]

        pt = prettytable.PrettyTable()
        pt.align = "l"
        pt.field_names = ["key", "value"]
        pt.add_row(["batch_size", batch_size])
        pt.add_row(["cache_lens", cache_lens])
        pt.add_row(["q_lens", q_lens])
        print(pt)

        execute_info = {
            "bench_mode": "var", 
            "batch_size": batch_size, 
            "cache_lens": cache_lens, 
            "q_lens": q_lens, 
            "block_size": block_size
        }
        print(f"sending var input: {execute_info}")
        self.execute(execute_info)



    def bench(
        self, 
        batch_size, cache_len, q_len, 
        test_cases, 
        block_size=512
    ):
        if test_cases:
            for case_idx, test_case in enumerate(test_cases):
                cache_len = test_case.get("cache_len", "")
                q_len = test_case.get("q_len", "")

                if not cache_len or not q_len:
                    print("[warning] cache_len or q_len is empty, skip")

                try:
                    if ";" in cache_len or ";" in q_len:
                        cache_lens = [int(x) for x in cache_len.split(";")]
                        q_lens = [int(x) for x in q_len.split(";")]
                        self.var_bench_func(cache_lens, q_lens, block_size=block_size)
                    else:
                        batch_size = int(test_case.get("batch_size", 1))
                        cache_len = int(cache_len)
                        q_len = int(q_len)
                        self.fix_bench_func(batch_size, cache_len, q_len, block_size=block_size)
                except Exception as e:
                    print(f"[error] fix_bench_func failed, case_idx: {case_idx}, test_case: {test_case}, error: {e}")
        else:
            self.fix_bench_func(batch_size, cache_len, q_len, block_size)

        self.dump_info()



    def summary_results(self, data_dict):
        pt = prettytable.PrettyTable()
        pt.field_names = ["key", "value"]
        pt.align = "l"
        for key, value in data_dict["bench_info"].items():
            if key in ["slot_mapping", "block_table"]:
                continue
            pt.add_row([key, value])
        pt.add_row(["-" * 25, "-" * 50])
        pt.add_row(["total_latency", f"{data_dict['perf_info']['e2e_latency']} ms"])
        print(pt)
        print("")



    def dump_info(self):
        # reports/raw
        if self.fix_data_dict:
            for value in self.fix_data_dict.values():
                self.summary_results(value)

        # reports/batch
        if self.var_data_dict:
            for value in self.var_data_dict:
                self.summary_results(value)



        # target_report_dir, config_dict = endpoint.get_export_info(bench_mode)
        # if target_report_dir.exists():
        #     shutil.rmtree(target_report_dir)
        # target_report_dir.mkdir(parents=True)

        # self.dump_extra_files(target_report_dir, bench_mode)







def dump_info(endpoint, data_list, bench_mode):
    target_report_dir, config_dict = endpoint.get_export_info(bench_mode)
    if target_report_dir.exists():
        shutil.rmtree(target_report_dir)
    target_report_dir.mkdir(parents=True)

    endpoint.dump_extra_files(target_report_dir)


    # config.json
    config_json_file = target_report_dir.joinpath("config.json")
    with open(config_json_file, "w") as f:
        json.dump(config_dict, f, indent=4)

    # latency.csv
    latency_csv_file = target_report_dir.joinpath("latency.csv")
    with open(latency_csv_file, "w") as f:
        dict_writer = csv.DictWriter(f, ["batch_size", "cache_len", "q_len", "stage_latency", "e2e_latency"])
        dict_writer.writeheader()
        for data_dict in data_list:
            batch_size = data_dict["original_inputs"]["batch_size"]
            cache_len = data_dict["original_inputs"]["cache_len"]
            q_len = data_dict["original_inputs"]["q_len"]
            latency = data_dict["results"]["latency"]
            dict_writer.writerow(
                {
                    "batch_size": batch_size,
                    "cache_len": cache_len,
                    "q_len": q_len,
                    "stage_latency": latency,
                    "e2e_latency": latency,
                }
            )



def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=49371)

    """
    run_mode: 指定当前的运行模式, 目前主要是应对kv_mirror的处理场景
    - prefill: 不需要运行mirror层的计算, 出字交给decode处理。
    - decode: 需要完整运行所有层的计算。
    """
    run_mode_choices = ["prefill", "decode"]
    parser.add_argument("--run_mode", type=str, choices=run_mode_choices, default="prefill")

    parser.add_argument("--block_size", type=int, default=512)

    # mode 1
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_len", type=int, default=0)
    parser.add_argument("--q_len", type=int, default=4096)
    
    # mode 2
    parser.add_argument("--csv", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()

    # check bench_config
    bench_config_path = pathlib.Path(args.model).absolute()
    if not bench_config_path.exists():
        raise FileNotFoundError(f"Model directory {bench_config_path} not found!")
    
    # check test_cases, parsed from csv file
    test_cases = []
    if args.csv is not None:
        csv_file = pathlib.Path(args.csv).absolute()
        if csv_file and csv_file.exists():
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                test_cases = list(reader)


    endpoint = XpuPerfSimEngine(
        args.ip, 
        args.port, 
        bench_config_path, 
        run_mode=args.run_mode
    )

    endpoint.bench(
        # used for raw mode
        batch_size=args.batch_size, 
        cache_len=args.cache_len, 
        q_len=args.q_len, 

        # used for batch mode
        test_cases=test_cases,

        block_size=args.block_size,
    )

    del endpoint

    
    








# def raw_bench_model(
#     script_path, backend, model_dir, run_mode, 
#     batch_size=1, cache_len=0, q_len=4096, 
#     **kwargs
# ):
#     bench_mode = "raw"
#     print("*"*100)
#     print(f"bench_mode={bench_mode}, run_mode={run_mode}")
#     print(f"batch_size={batch_size}, cache_len={cache_len}, q_len={q_len}")
#     print("*"*100)

#     bench_mode = "raw"
#     endpoint = XpuPerfSimEngine(script_path, backend, model_dir, run_mode=run_mode)
#     endpoint.create_engine()

#     bench_dict = {
#         "bench_mode": bench_mode, 
#         "batch_size": batch_size, 
#         "cache_len": cache_len, 
#         "q_len": q_len
#     }
#     print(f"sending {bench_mode} sample: {bench_dict}")

#     data_dict = {
#         "original_inputs": bench_dict, 
#         "results": endpoint.execute([bench_dict])[0]
#     }

#     print("*"*100)
#     print(f"{bench_mode}")
#     print("*"*100)
#     summary_results(data_dict)
#     print("")

#     dump_info(endpoint, [data_dict], bench_mode)




# def csv_bench_model(
#     script_path, 
#     backend, 
#     model_dir,
#     run_mode, 
#     test_cases, 
#     **kwargs
# ):
#     bench_mode = "csv"
#     endpoint = XpuPerfSimEngine(script_path, backend, model_dir, run_mode=run_mode)
#     endpoint.create_engine()

#     bench_dict_list = []
#     for case in test_cases:
#         bench_dict = {
#             "bench_mode": bench_mode, 
#             "batch_size": int(case["batch_size"]), 
#             "q_len": int(case["q_len"]), 
#             "cache_len": int(case["cache_len"])
#         }
#         bench_dict_list.append(bench_dict)
    
#     result_dict_list = endpoint.execute(bench_dict_list)
    
#     data_list = []
#     for bench_dict, result_dict in zip(bench_dict_list, result_dict_list):
#         data_dict = {
#             "original_inputs": bench_dict, 
#             "results": result_dict
#         }
#         data_list.append(data_dict)
    
#     # summary to stdout
#     print("*"*100)
#     print(f"{bench_mode}")
#     print("*"*100)
#     for data_dict in data_list:
#         summary_results(data_dict)
#     print("")

#     dump_info(endpoint, data_list, bench_mode)



# def prefill_bench_model(
#     script_path, backend, model_dir, run_mode="prefill", 
#     **kwargs
# ):
#     bench_mode = "prefill"
#     print("*"*100)
#     print(f"bench_mode={bench_mode}, run_mode={run_mode}")
#     print("*"*100)

#     endpoint = XpuPerfSimEngine(script_path, backend, model_dir, 
#         bench_mode=bench_mode
#     )
#     endpoint.create_engine()

#     # test cases
#     # __full_q_len_list = [4096]
#     __full_q_len_list = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
#     __full_q_len_list.extend([2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096])
#     __full_q_len_list.extend([6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768])

#     batch_size = 1
#     cache_len = 0

    
#     bench_list = []
#     for q_len in __full_q_len_list:
#         bench_list.append({
#             "bench_mode": bench_mode, 
#             "batch_size": batch_size, 
#             "q_len": q_len, 
#             "cache_len": cache_len
#         })
    
#     result_list = endpoint.execute(bench_list)
    
#     data_list = []
#     for bench_dict, result_dict in zip(bench_list, result_list):
#         data_dict = {
#             "original_inputs": bench_dict, 
#             "results": result_dict
#         }
#         data_list.append(data_dict)

#     data_list = sorted(data_list, key=lambda x: x["original_inputs"]["q_len"])

#     # summary to stdout
#     print("*"*100)
#     print(f"{bench_mode}")
#     print("*"*100)
#     for data_dict in data_list:
#         summary_results(data_dict)
#     print("")

#     dump_info(endpoint, data_list, bench_mode)
    
        
    


