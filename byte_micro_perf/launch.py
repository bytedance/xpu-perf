import os
import sys
import csv
import json
import copy
import shutil
import pathlib
import argparse
import jsonlines
import traceback
import importlib
import prettytable
from typing import Any, Dict, List

import torch
import torch.multiprocessing as mp


FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR
BACKENDS_DIR = BYTE_MLPERF_ROOT.joinpath("backends")
sys.path.insert(0, str(BYTE_MLPERF_ROOT))


from core.utils import logger, setup_logger, parse_json_file, parse_csv_file
from core.engine import ComputeEngine, XCCLEngine


def parse_tasks(task_dir, task):
    """
    默认使用task_dir下递归遍历得到的 **task.json** 以及 **task.csv** 文件。
    """
    task_dict : Dict[str, List[Dict[str, Any]]] = {}

    if task_dir is not None:
        task_dir = pathlib.Path(task_dir).absolute()
        if not task_dir.exists():
            logger.error(f"Task dir {task_dir} not exists")
            sys.exit(1)

        # csv_task_list = [task_csv.stem for task_csv in task_dir.rglob("*.csv")]
        # json_task_list = [task_json.stem for task_json in task_dir.rglob("*.json")]
        # all_task_list = list(set(json_task_list) | set(csv_task_list))

        json_file_list = list(task_dir.rglob("*.json"))
        
        
        all_test_cases = {}
        for json_file in json_file_list:
            cur_task_cases = parse_json_file(json_file)

            for kernel, test_cases in cur_task_cases.items():
                if kernel not in all_test_cases:
                    all_test_cases[kernel] = []
                all_test_cases[kernel].extend(test_cases)

        target_op_set = set()
        if task == "all":
            pass
        else:
            for required_task in task.split(","):
                required_task = required_task.strip()
                target_op_set.add(required_task)

        target_test_cases = {k: v for k, v in all_test_cases.items() if k in target_op_set}
       
    return target_test_cases


def parse_workload(workload):
    """
    解析指定的 json or csv 文件
    其中 json文件是列表, 通过笛卡尔积的方式生成所有参数组合, 方便快速生成所有测试用例。
    csv文件是逗号分隔的文件, 第一行是表头, 后面是所有参数组合。
    """
    task_dict : Dict[str, List[Dict[str, Any]]] = {}
    if workload is not None:
        workload_path = pathlib.Path(workload).absolute()
        if not workload_path.exists():
            logger.error(f"Workload file {workload_path} not exists")
            sys.exit(1)

        if workload_path.suffix == ".json":
            task_dict.update(parse_json_file(workload_path))
        elif workload_path.suffix == ".csv":
            task_dict.update(parse_csv_file(workload_path))
        else:
            logger.error(f"Workload file {workload_path} not support, only support json or csv format")
            sys.exit(1)

    return task_dict


def parse_replay_tasks(replay_dir):
    """
    解析replay_dir下的所有 ** rank_*.json **文件, 对齐batch解析所有rank的test cases
    """
    replay_dir = pathlib.Path(replay_dir).absolute()
    if not replay_dir.exists() or not replay_dir.is_dir():
        return {}
    
    replay_task_dict: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for rank_file in replay_dir.glob("rank_*.json"):
        rank_id = int(rank_file.stem.split("_")[-1])
        replay_task_dict[rank_id] = parse_workload(rank_file)

    sorted_items_by_key = sorted(replay_task_dict.items(), key=lambda x: x[0])
    replay_task_dict = dict(sorted_items_by_key)

    return replay_task_dict



def parse_args():
    if not BACKENDS_DIR.exists():
        logger.error(f"Backends directory {BACKENDS_DIR} not found")
        return 1

    backend_list = []
    for backend_dir in BACKENDS_DIR.iterdir():
        if backend_dir.is_dir():
            backend_list.append(backend_dir.name)

    parser = argparse.ArgumentParser()
    
    # backend
    parser.add_argument(
        "--backend", type=str, default="GPU", choices=backend_list, 
        help="Backend to use, default is GPU"
    )
    parser.add_argument(
        "--show_backends", action="store_true", 
        help="Show supported backends"
    )

    # task
    parser.add_argument(
        "--task_dir", type=str, 
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads", "basic")), 
        help="Task directory, default is workloads/basic"
    )
    parser.add_argument(
        "--task", type=str, default="all", 
        help="Task to bench, default is all"
    )
    parser.add_argument(
        "--workload", type=str, default=None, 
        help="Workload to bench, suppor jsonl or csv format. "
             "default is None which use **task.json** or **task.csv** in **task_dir**"
    )   
    parser.add_argument(
        "--replay_dir", type=str, default=None, 
        help="replay directory, default is None"
    )
    parser.add_argument(
        "--report_dir", type=str, 
        default=str(BYTE_MLPERF_ROOT.joinpath("reports")), 
        help="Report directory, default is reports"
    )

    # numa
    parser.add_argument(
        "--numa", type=str, default=None, 
        help="Numa config. "
            "Default is None which create **num_numa_nodes** processes to bench, "
             "each of them run on one numa node and schedule some devices. "
             "Values '-1' or '0' or '1' mean creating one process and specifing all numa nodes or node 0 or node 1. "
             "Value '0,1' means creating 2 processes and assign node 0 and node 1 to them respectively. "
    )

    # devices
    parser.add_argument(
        "--device", type=str, default=None, 
        help="Device config."
             "Default is None which use all devices on current machine."
             "Value '0,1' means using device 0 and device 1 on current machine."
    )

    # serving config
    parser.add_argument(
        "--node_world_size", type=int, default=1, 
        help="Node world size, default is 1"
    )
    parser.add_argument(
        "--node_rank", type=int, default=0,
        help="Node rank, default is 0"
    )


    # 通信相关
    parser.add_argument(
        "--server_port", type=int, default=49372, 
        help="Server port, default is 49372"
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost", 
        help="Master address, default is localhost"
    )
    parser.add_argument(
        "--host_port", type=int, default=49373, 
        help="Host port, default is 49373"
    )
    parser.add_argument(
        "--device_port", type=int, default=49374, 
        help="Device port, default is 49374"
    )

    # utils
    parser.add_argument("--enable_profiling", action="store_true", 
                        help="Enable profiling, default is False")
    parser.add_argument("--log_level", type=str, default="INFO")


    args = parser.parse_args()


    # setup logger
    setup_logger(args.log_level)


    # backend相关
    if args.show_backends:
        logger.info(f"Supported backends: {backend_list}")
        return 0
    if args.backend not in backend_list:
        logger.error(f"Backend {args.backend} not found in {backend_list}")
        return 1
    else:
        logger.info(f"Using backend {args.backend}")

    try:
        backend_module = importlib.import_module(
            "backends." + args.backend + ".backend_" + args.backend.lower())
        backend_class = getattr(backend_module, "Backend" + args.backend)
        backend_instance = backend_class()
        backend_instance.backend_type = args.backend
        backend_instance.load_all_ops()
    except Exception as e:
        logger.error(f"Failed to import backend {args.backend}: {e}")
        return 1

    # 获取系统基本信息
    common_pt = prettytable.PrettyTable()
    common_pt.field_names = ["attr", "value"]
    common_pt.align = "l"
    for attr, value in backend_instance.common_info.items():
        if attr == "numa_configs":
            continue
        else:
            common_pt.add_row([attr, value])    

    # 获取provider相关信息
    provider_pt = prettytable.PrettyTable()
    provider_pt.field_names = ["provider", "version"]
    provider_pt.align = "l"
    for provider, version in backend_instance.provider_info.items():
        provider_pt.add_row([provider, version])


    # 获取backend相关信息
    info_pt = prettytable.PrettyTable()
    info_pt.field_names = ["attr", "value"]
    info_pt.align = "l"
    for attr, value in backend_instance.backend_info.items():
        info_pt.add_row([attr, value])

    # 获取env相关信息
    env_pt = prettytable.PrettyTable()
    env_pt.field_names = ["env", "is_preset", "default_val", "final_val"]
    env_pt.align = "l"
    for attr in backend_instance.default_envs:
        if attr in backend_instance.override_envs:
            env_pt.add_row([attr, "True", backend_instance.default_envs[attr], os.environ[attr]])
        else:
            env_pt.add_row([attr, "False", backend_instance.default_envs[attr], os.environ[attr]])
    logger.info(f"Backend {args.backend} instance created.")

    logger.info(f"common info: \n{common_pt}")
    logger.info(f"backend info: \n{info_pt}")
    logger.info(f"provider info: \n{provider_pt}")
    logger.info(f"env info: \n{env_pt}")



    # 解析 numa_config
    numa_config = args.numa
    if numa_config is None:
        numa_num = len(backend_instance.common_info["numa_configs"])
        numa_order = list(range(numa_num))
        
    else:
        numa_num = len(numa_config.split(","))
        numa_order = [int(x) for x in numa_config.split(",")]
    device_mapping = list(range(backend_instance.backend_info["device_count"]))

    logger.info(f"use {numa_num} numa nodes, numa_order: {numa_order}, mapping to {device_mapping}")


    # 解析 node dist config
    node_world_size = args.node_world_size
    node_rank = args.node_rank
    all_numa_num = node_world_size * numa_num
    logger.info(f"node_world_size: {node_world_size}, node_rank: {node_rank}, all_numa_num: {all_numa_num}")

    # devices
    device = args.device
    if device is None:
        device_ids = device_mapping
    else:
        try:
            device_ids = [int(x) for x in device.split(",")]
        except ValueError:
            logger.error(f"Invalid device format: {device}, should be comma-separated integers")
            return 1
        for id in device_ids:
            if id not in device_mapping:
                logger.error(f"Invalid device id: {id}, not in {device_mapping}")
                return 1
    logger.info(f"using devices: {device_ids}")


    # 解析 serving config
    master_addr = args.master_addr
    server_port = args.server_port
    host_port = args.host_port
    device_port = args.device_port

    server_pt = prettytable.PrettyTable()
    server_pt.field_names = ["attr", "value", "note"]
    server_pt.align = "l"
    server_pt.add_row(["master_addr", master_addr, "ip address for serving and dist communication for gloo and xccl."])
    server_pt.add_row(["server_port", server_port, "port for serving requests."])
    server_pt.add_row(["host_port", host_port, "port for host communication."])
    server_pt.add_row(["device_port", device_port, "port for device communication."])
    logger.info(f"serving config: \n{server_pt}")



    """
    只有node0才会解析task, 并派发权重
    """
    replay_test_cases = {}
    test_cases = {}
    if node_rank == 0:
        # replay process
        if args.replay_dir is not None:
            replay_test_cases = parse_replay_tasks(args.replay_dir)
        elif args.workload is not None:
            test_cases = parse_workload(args.workload)
        else:
            test_cases = parse_tasks(args.task_dir, args.task)
    
        if replay_test_cases:
            print("*" * 100)
            logger.info(f"replay test cases: ")
            for rank, all_op_cases in replay_test_cases.items():
                for op_name, op_cases in all_op_cases.items():
                    logger.info(f"rank {rank} {op_name} has {len(op_cases)} test cases")
            print("*" * 100)

        if test_cases:
            print("*" * 100)
            logger.info(f"test cases: ")
            for op_name, op_cases in test_cases.items():
                logger.info(f"{op_name} has {len(op_cases)} test cases")
            print("*" * 100)

    report_dir = pathlib.Path(args.report_dir).absolute()
    report_dir.mkdir(parents=True, exist_ok=True)

    return {
        # backend instance to use
        "backend_instance": backend_instance,

        # task related
        "replay_test_cases": replay_test_cases,
        "test_cases": test_cases,

        # device config
        "device_mapping": device_mapping,
        "device_ids": device_ids,

        # numa config
        "numa_num": numa_num,
        "numa_order": numa_order,
        
        # dist config
        "node_world_size": node_world_size,
        "node_rank": node_rank,
        "all_numa_num": all_numa_num, 
        "master_addr": master_addr,
        "server_port": server_port,
        "host_port": host_port,
        "device_port": device_port,

        # report dir
        "report_dir": report_dir,

        # utils
        "enable_profiling": args.enable_profiling,
    }


def norm_bench_process(args_dict):
    backend_instance = args_dict["backend_instance"]
    device_ids = args_dict["device_ids"]


    # 默认保存到 reports 目录下
    report_dir = args_dict["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    # 导出环境信息
    common_info = copy.deepcopy(backend_instance.common_info)
    common_info.pop("numa_configs", None)
    common_info["device_ids"] = str(device_ids)
    export_dict = {
        "common": common_info,
        "provider": backend_instance.provider_info, 
        "backend": backend_instance.backend_info,
        "default_envs": backend_instance.default_envs,
        "override_envs": backend_instance.override_envs
    }
    target_info_file = report_dir.joinpath(
        backend_instance.backend_type, 
        backend_instance.backend_info["device_name"], 
        "info.json"
    )
    target_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_info_file, "w") as f:
        json.dump(export_dict, f, indent=4)





    """
    {
        "flash_attention": [], 
        "gemm": []
    }
    """
    test_cases = args_dict["test_cases"]
    
    # 分类当前任务同时创建对应的engine进行任务派发。
    engine_tasks = {}
    for op_name in test_cases:
        if op_name not in backend_instance.op_mapping:
            logger.error(f"op_name: {op_name} not in backend_instance.op_mapping")
            continue

        for op_provider, op_config in backend_instance.op_mapping[op_name].items():
            op_cls = op_config["op_cls"]
            engine_name = op_config["engine_name"]

            if engine_name not in engine_tasks:
                engine_tasks[engine_name] = {}
            
            key_tuple = (op_name, op_provider, op_cls)
            engine_tasks[engine_name][key_tuple] = test_cases[op_name]

    """
    Engine的作用:
    1. 负责拉起当前node需要参与计算的所有卡, 每张卡对应一个子进程, 并指定对应的numa亲和性, 维护engine和backend的通信。
    2. 维护engine和主进程的通信, 接收主进程下发的task和test_case, 根据各自engine的属性下发给对应的子进程。
    3. 对于每次请求, 在测试完成后, 接收子进程返回的结果, 进行整理后返回给主进程。

    目前支持三类engine: 
    1. ComputeEngine: 目前只有node rank0创建, 使用本地所有卡独立参与所有测试用例计算。 
    2. XCCLEngine: 所有node创建, 每个node使用所有卡参与计算。
    3. P2PEngine: 目前只有node rank0创建, 使用本地所有卡按照通信矩阵完成所有测试用例计算。  
    """

    engines = {}
    if "ComputeEngine" in engine_tasks:
        engines["ComputeEngine"] = ComputeEngine(args_dict)
    if "XCCLEngine" in engine_tasks:
        engines["XCCLEngine"] = XCCLEngine(args_dict)
    for engine_name, engine in engines.items():
        engine.start()


    total_results = {}

    for engine_name, engine in engines.items():
        print("\n")
        logger.info(f"{'*'*20} {engine_name} {'*'*20}")
        cur_results = engine.dispatch(engine_tasks[engine_name])
        total_results.update(cur_results)
        print("\n")

    for engine_name, engine in engines.items():
        engine.stop()



    for task_info, results in total_results.items():
        op_name, op_provider, *_ = task_info
        
        """
        target_dir: 
        - backend
        - device_name
        - op_name
        - op_provider
        """
        target_dir = report_dir.joinpath(
            backend_instance.backend_type, 
            backend_instance.backend_info["device_name"], 
            op_name, 
            op_provider
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        # 按照 arg_type, dtype, dst_dtype进行分类
        data_classify = {}

        for result in results:
            key_str = ""

            result_idx = result[0]
            result_args = result[1]["arguments"]
            result_targets = result[1]["targets"]

            if "arg_type" in result_args:
                key_str += f"{result_args['arg_type']}_"
            if "dtype" in result_args:
                key_str += f"{result_args['dtype']}_"
            if "dst_dtype" in result_args:
                key_str += f"{result_args['dst_dtype']}_"
            if "world_size" in result_args:
                key_str += f"group{result_args['world_size']}_"

            if key_str.endswith("_"):
                key_str = key_str[:-1]

            if key_str not in data_classify:
                data_classify[key_str] = []

            template_dict = {
                "sku_name": backend_instance.backend_info["device_name"], 
                "op_name": op_name, 
                "provider": op_provider, 
                "arguments": result_args,
                "targets": result_targets
            }

            data_classify[key_str].append(template_dict)


        for key in data_classify:
            with jsonlines.open(target_dir.joinpath(f"{key}.jsonl"), "w") as f:
                f.write_all(data_classify[key])

            keys = ["sku_name", "op_name", "provider"]
            keys.extend(data_classify[key][0]["arguments"].keys())
            keys.extend(data_classify[key][0]["targets"].keys())

            with open(target_dir.joinpath(f"{key}.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for item in data_classify[key]:
                    row = []
                    row.append(item["sku_name"])
                    row.append(item["op_name"])
                    row.append(item["provider"])
                    row.extend(item["arguments"].values())
                    row.extend(item["targets"].values())
                    writer.writerow(row)




def replay_bench_process(args_dict):
    backend_instance = args_dict["backend_instance"]
    device_ids = args_dict["device_ids"]
    """
    0: {"flash_attention": []}
    1: {"flash_attention": []}
    ...
    15: {"flash_attentin": []}
    """
    replay_test_cases = args_dict["replay_test_cases"]
    
    # 目前只启动ComputeEngine用于复现算子性能
    engine = ComputeEngine(args_dict)
    engine.start()

    """
    将replay_test_cases合成一次下发加快速度, 
    前提是每个rank的每个算子的所有测试用例都是对应上的, 如果是线上抓取的数据一般是可以保证的. 
    """
    rank_keys = list(replay_test_cases.keys())
    num_ranks = len(rank_keys)

    all_dispatch_tasks = {}
    for op_name in replay_test_cases[rank_keys[0]]:
        if op_name not in backend_instance.op_mapping:
            logger.error(f"op_name: {op_name} not in backend_instance.op_mapping")
            continue

        for op_provider, op_config in backend_instance.op_mapping[op_name].items():
            op_cls = op_config["op_cls"]
            engine_name = op_config["engine_name"]
            key_tuple = (op_name, op_provider, op_cls)

            # 收集所有rank的同配置测试用例
            all_dispatch_tasks[key_tuple] = []
            for target_rank in rank_keys:
                all_dispatch_tasks[key_tuple].extend(replay_test_cases[target_rank][op_name])
        
    all_dispatch_results = engine.dispatch(all_dispatch_tasks)


    replay_results = {}
    for rank_idx in rank_keys:
        replay_results[rank_idx] = copy.deepcopy(all_dispatch_results)
        for key_tuple in replay_results[rank_idx]:
            num_results = len(replay_results[rank_idx][key_tuple])
            num_results_per_rank = num_results // num_ranks
            replay_results[rank_idx][key_tuple] = replay_results[rank_idx][key_tuple][rank_idx * num_results_per_rank : (rank_idx + 1) * num_results_per_rank]
            


    # 默认保存到 reports_replay 目录下
    report_dir = args_dict["report_dir"]
    replay_report_dir = report_dir.parent.joinpath(f"{report_dir.name}_replay")
    replay_report_dir.mkdir(parents=True, exist_ok=True)

    # 导出环境信息
    common_info = copy.deepcopy(backend_instance.common_info)
    common_info.pop("numa_configs", None)
    common_info["device_ids"] = str(device_ids)
    export_dict = {
        "common": common_info,
        "provider": backend_instance.provider_info, 
        "backend": backend_instance.backend_info,
        "default_envs": backend_instance.default_envs,
        "override_envs": backend_instance.override_envs
    }
    target_info_file = replay_report_dir.joinpath(
        backend_instance.backend_type, 
        backend_instance.backend_info["device_name"], 
        "info.json"
    )
    target_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_info_file, "w") as f:
        json.dump(export_dict, f, indent=4)



    # 逐op导出每个batch的耗时
    for key_tuple, results in replay_results[rank_keys[0]].items():
        op_name, op_provider, *_ = key_tuple

        """
        target_dir
        - backend
        - device_name
        - op_name
        """
        target_dir = replay_report_dir.joinpath(
            backend_instance.backend_type, 
            backend_instance.backend_info["device_name"], 
            op_name
        )
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        

        target_file = target_dir.joinpath(f"{op_provider}.json")
        all_target_data = []

        for result_idx, result in enumerate(results):
            peer_results = []
            for rank_idx in rank_keys:
                peer_results.append(replay_results[rank_idx][key_tuple][result_idx][1])

            template_dict = {
                "sku_name": backend_instance.backend_info["device_name"], 
                "op_name": op_name, 
                "provider": op_provider, 
                "arguments": peer_results[0]["arguments"],
                "targets": peer_results[0]["targets"],
            }

            if "q_lens" in template_dict["arguments"]:
                q_lens_list = [
                    peer_result["arguments"]["q_lens"] for peer_result in peer_results
                ]
                sum_q_list = [
                    sum(q_lens) for q_lens in q_lens_list
                ]
                aver_q_list = [
                    int(sum_q / len(q_lens)) for sum_q, q_lens in zip(sum_q_list, q_lens_list)
                ]
                template_dict["arguments"]["q_lens"] = [
                    str(q_lens) for q_lens in q_lens_list
                ]
                template_dict["arguments"]["sum_q_lens"] = str([sum_q for sum_q in sum_q_list])
                template_dict["arguments"]["aver_q_lens"] = str([aver_q for aver_q in aver_q_list])
            if "cache_lens" in template_dict["arguments"]:
                cache_lens_list = [
                    peer_result["arguments"]["cache_lens"] for peer_result in peer_results
                ]
                sum_cache_list = [
                    sum(cache_lens) for cache_lens in cache_lens_list
                ]
                aver_cache_list = [
                    int(sum_cache / len(cache_lens)) for sum_cache, cache_lens in zip(sum_cache_list, cache_lens_list)
                ]
                template_dict["arguments"]["cache_lens"] = [
                    str(cache_lens) for cache_lens in cache_lens_list
                ]
                template_dict["arguments"]["sum_cache_lens"] = str([sum_cache for sum_cache in sum_cache_list])
                template_dict["arguments"]["aver_cache_lens"] = str([aver_cache for aver_cache in aver_cache_list])
            if "block_table" in template_dict["arguments"]:
                template_dict["arguments"]["block_table"] = [
                    str(peer_result["arguments"]["block_table"]) for peer_result in peer_results
                ]

            latency_list = [
                peer_result["targets"]["latency(us)"] for peer_result in peer_results
            ]

            for key in template_dict["targets"]:
                template_dict["targets"][key] = str([
                    peer_result["targets"][key] for peer_result in peer_results
                ])
            
            all_target_data.append(template_dict)

            print("*" * 100)
            print("batch:", result_idx)
            print("max latency(us):", max(latency_list))
            print("min latency(us):", min(latency_list))
            print("latency:", template_dict["targets"]["latency(us)"])
            print("mem_bw(GB/s):", template_dict["targets"]["mem_bw(GB/s)"])
            print("calc_flops_power(tflops):", template_dict["targets"]["calc_flops_power(tflops)"])
            
        with open(target_file, "w") as f:
            json.dump(all_target_data, f, indent=4)

    engine.stop()


if __name__ == "__main__":
    args_dict = parse_args()
    backend_instance = args_dict["backend_instance"]
    node_world_size = args_dict.get("node_world_size", 1)
    node_rank = args_dict.get("node_rank", 0)

    # init mp spawn context
    try:
        mp.set_start_method("spawn", force=True)
    except Exception as e:
        logger.exception(f"failed to set spawn context: {e}")
        traceback.print_exc()
        sys.exit(-1)   

    if node_rank == 0:
        if args_dict.get("test_cases", {}):
            norm_bench_process(args_dict)
        elif args_dict.get("replay_test_cases", {}):
            replay_bench_process(args_dict)
    else:
        engine = XCCLEngine(args_dict)
        engine.start()
        if engine.subprocess_procs:
            for subprocess in engine.subprocess_procs.processes:
                subprocess.join()

    # clean redundant files
    backend_instance.clean_extra_files()
    