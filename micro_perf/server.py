import os
import sys
import signal
import pathlib
import argparse
import threading
import traceback
import importlib
import prettytable

import torch
import torch.multiprocessing as mp

from flask import Flask, request, jsonify

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR
BACKENDS_DIR = BYTE_MLPERF_ROOT.joinpath("backends")
sys.path.insert(0, str(BYTE_MLPERF_ROOT))

from core.utils import logger, setup_logger
from perf_engine import XpuPerfServer


mp.set_start_method('spawn', force=True)
g_server_instance = None
g_app_instance = None



def parse_args():
    setup_logger("INFO")
    if not BACKENDS_DIR.exists():
        logger.error(f"Backends directory {BACKENDS_DIR} not found")
        return 1
    
    backend_list = []
    for backend_dir in BACKENDS_DIR.iterdir():
        if backend_dir.is_dir():
            backend_list.append(backend_dir.name)

    parser = argparse.ArgumentParser()
    
    # backend config
    parser.add_argument(
        "--backend", type=str, default="GPU", choices=backend_list, 
        help="Backend to use, default is GPU"
    )

    # numa config
    parser.add_argument(
        "--numa", type=str, default=None, 
        help="Numa config. "
            "Default is None which create **num_numa_nodes** processes to bench, "
             "each of them run on one numa node and schedule some devices. "
             "Values '-1' or '0' or '1' mean creating one process and specifing all numa nodes or node 0 or node 1. "
             "Value '0,1' means creating 2 processes and assign node 0 and node 1 to them respectively. "
    )
    
    # device config
    parser.add_argument(
        "--device", type=str, default=None, 
        help="Device config."
             "Default is None which use all devices on current machine."
             "Value '0,1' means using device 0 and device 1 on current machine."
    )

    # node config
    parser.add_argument(
        "--node_world_size", type=int, default=1, 
        help="Node world size, default is 1"
    )
    parser.add_argument(
        "--node_rank", type=int, default=0,
        help="Node rank, default is 0"
    )

    # server config
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=49371)
    parser.add_argument("--host_port", type=int, default=49372)
    parser.add_argument("--device_port", type=int, default=49373)
    
    args = parser.parse_args()


    # backend
    if args.backend not in backend_list:
        logger.error(f"Backend {args.backend} not found in {backend_list}")
        return 1
    else:
        logger.info(f"Using backend {args.backend}")

    # create backend instance and load all avail ops
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
    
    return {
        "backend_instance": backend_instance,

        "device_mapping": device_mapping,
        "device_ids": device_ids,

        "numa_num": numa_num,
        "numa_order": numa_order,

        "node_world_size": node_world_size,
        "node_rank": node_rank,
        "all_numa_num": all_numa_num, 

        "master_addr": master_addr,
        "server_port": server_port,
        "host_port": host_port,
        "device_port": device_port
    }



if __name__ == '__main__':
    args_dict = parse_args()
    server_instance = XpuPerfServer(args_dict)
    server_instance.create()
    g_server_instance = server_instance


    app_instance = Flask("XpuPerfServer")
    g_app_instance = app_instance

    @app_instance.route("/info", methods=["GET"])
    def info():
        try:
            info_dict = server_instance.get_info()
            return jsonify(info_dict)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app_instance.route("/bench", methods=["POST"])
    def bench():
        try:
            input_dict = request.json
            result_dict = server_instance.bench(input_dict)
            return jsonify(result_dict)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    app_instance.run(
        host=args_dict["master_addr"],
        port=args_dict["server_port"],
        debug=False, 
        use_reloader=False,
        threaded=True
    )

    server_instance.destroy()
