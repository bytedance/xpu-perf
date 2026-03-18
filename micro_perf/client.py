import os
import csv
import sys
import json
import copy
import pathlib
import requests
import argparse
import jsonlines
import prettytable
from functools import partial
from typing import Any, Dict, List


FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR
BACKENDS_DIR = BYTE_MLPERF_ROOT.joinpath("backends")
sys.path.insert(0, str(BYTE_MLPERF_ROOT))

from core.utils import logger, setup_logger, parse_json_file, parse_csv_file


def parse_tasks(task_dir, task):
    """
    默认使用task_dir下递归遍历得到的 **task.json** 文件。
    """
    task_dict : Dict[str, List[Dict[str, Any]]] = {}


    if task_dir is not None:
        task_dir = pathlib.Path(task_dir).absolute()
        if not task_dir.exists():
            logger.error(f"Task dir {task_dir} not exists")
            sys.exit(1)

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
            task_dict = all_test_cases
        else:
            for required_task in task.split(","):
                required_task = required_task.strip()
                target_op_set.add(required_task)
            task_dict = {k: v for k, v in all_test_cases.items() if k in target_op_set}
    return task_dict


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


def parse_args():
    setup_logger("INFO")
    parser = argparse.ArgumentParser()

    # task input
    parser.add_argument("--task_dir", type=str, 
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads", "basic")))
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--workload", type=str)

    # report output
    parser.add_argument("--report_dir", type=str, 
        default=str(BYTE_MLPERF_ROOT.joinpath("reports")))

    parser.add_argument("--server_ip", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=49371)
    
    args = parser.parse_args()

    return args




def export_reports(
    given_report_dir, 
    info_dict, 
    test_cases={}, 
    bench_results={}
):
    print("*"*100)
    print(f"reports")
    print("*"*100)
    report_dir = pathlib.Path(given_report_dir).absolute()
    report_dir.mkdir(parents=True, exist_ok=True)

    backend_type = info_dict["backend_type"]
    device_name = info_dict["backend"]["device_name"]
    
    target_info_file = report_dir.joinpath(backend_type, device_name, "info.json")
    target_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_info_file, "w") as f:
        export_dict = copy.deepcopy(info_dict)
        export_dict["common"].pop("numa_configs")
        export_dict["runtime"]["device_ids"] = str(export_dict["runtime"]["device_ids"])
        export_dict["runtime"]["device_mapping"] = str(export_dict["runtime"]["device_mapping"])
        export_dict["runtime"]["numa_order"] = str(export_dict["runtime"]["numa_order"])
        json.dump(export_dict, f, indent=4)

    for op_name in bench_results:
        provider_results = {}
        for argument_dict, all_target_dict in zip(test_cases[op_name], bench_results[op_name]):
            for provider, target_dict in all_target_dict.items():
                if target_dict:
                    if provider not in provider_results:
                        provider_results[provider] = []
                    template_dict = {
                        "sku_name": info_dict["backend"]["device_name"], 
                        "op_name": op_name, 
                        "provider": provider, 
                        "arguments": argument_dict, 
                        "targets": target_dict
                    }
                    provider_results[provider].append(template_dict)
        
        for op_provider, data_list in provider_results.items():
            target_dir = report_dir.joinpath(
                backend_type, 
                device_name, 
                op_name, 
                op_provider
            )
            target_dir.mkdir(parents=True, exist_ok=True)


            target_jsonl_file = target_dir.joinpath(f"{op_name}-{op_provider}.jsonl")
            with jsonlines.open(target_jsonl_file, "w") as f:
                f.write_all(data_list)

            temp_pt = prettytable.PrettyTable(["attr", "value"])
            temp_pt.align = "l"
            temp_pt.add_row(["op_name", op_name])
            temp_pt.add_row(["op_provider", op_provider])
            
            arguments_set = set()
            target_set = set()
            for data in data_list:
                arguments_set.update(data["arguments"].keys())
                target_set.update(data["targets"].keys())  

                print(temp_pt)
                print(data["arguments"])
                print(json.dumps(data["targets"], indent=4))
                print("")

            
            target_csv_file = target_dir.joinpath(f"{op_name}-{op_provider}.csv")
            with open(target_csv_file, "w") as f:
                first_occur_arguments = data_list[0]["arguments"].keys()
                first_occur_targets = data_list[0]["targets"].keys()

                keys = ["sku_name", "op_name", "provider"]
                keys.extend(first_occur_arguments)
                for key in arguments_set - set(first_occur_arguments):
                    keys.append(key)
                keys.extend(first_occur_targets)
                for key in target_set - set(first_occur_targets):
                    keys.append(key)

                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()

                for item in data_list:
                    data_dict = {}
                    data_dict["sku_name"] = item["sku_name"]
                    data_dict["op_name"] = item["op_name"]
                    data_dict["provider"] = item["provider"]
                    data_dict.update(item["arguments"])
                    data_dict.update(item["targets"])
                    writer.writerow(data_dict)




if __name__ == '__main__':
    args = parse_args()

    # client apis
    info_url = f"http://{args.server_ip}:{args.server_port}/info"
    bench_url = f"http://{args.server_ip}:{args.server_port}/bench"

    get_info_func = partial(get_info_template, url=info_url)
    normal_bench_func = partial(normal_bench_template, url=bench_url)


    info_dict = get_info_func()
    if not info_dict or "status" in info_dict:
        logger.error("get server info failed.")
        sys.exit(-1)
    print_server_info(info_dict)




    test_cases = {}
    if args.workload is not None:
        test_cases = parse_workload(args.workload)
    else:
        test_cases = parse_tasks(args.task_dir, args.task)
    if not test_cases:
        logger.error("No valid test cases found. Exiting.")
        sys.exit(1)
    print("*" * 100)
    logger.info(f"test cases: ")
    for op_name, op_cases in test_cases.items():
        logger.info(f"{op_name} has {len(op_cases)} test cases")
    print("*" * 100)


    bench_results = normal_bench_func(test_cases)
    if not bench_results or "status" in info_dict:
        logger.error("get no results.")
        sys.exit(-1)

    export_reports(
        args.report_dir, 
        info_dict, 
        test_cases, 
        bench_results
    )


        
