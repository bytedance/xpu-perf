import os
import sys
import time
import pathlib
import argparse
from tqdm import tqdm
import numpy as np
import prettytable as pt
import csv
import importlib
import json
import random

FILE_DIR = pathlib.Path(__file__).parent.absolute()
# ${prj_root}/byte_infer_perf
INFER_ROOT_DIR = FILE_DIR.parent
CUR_DIR = pathlib.Path.cwd().absolute()
os.chdir(INFER_ROOT_DIR)


sys.path.insert(0, str(INFER_ROOT_DIR))

from llm_perf.utils.logger import logger, setup_logger


def parse_args():
    default_model_config = FILE_DIR.joinpath(
        "model_zoo", 
        "mixtral-torch-bf16-8x22b.json"
    ).absolute()

    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware_type", type=str, default="GPU")

    # model_info
    parser.add_argument("--model_config", type=str, default=default_model_config)   

    # deploy_config
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--max_batch_size", type=int, default=8)

    # realization_version, mapping to:
    # 1. engine_type: tp_engine, pp_engine, ...
    # 2. model realization info,
    parser.add_argument("--model_version", type=str, default="default")
    parser.add_argument("--infer_dtype", type=str, default="bfloat16")

    # perf config, using default config
    parser.add_argument("--perf_config", type=str, default=None)

    # feature control
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--log_model", action="store_true")


    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="info")

    return parser.parse_args()


def perf_engine(xpu_config, workspace):
    hardware_type = xpu_config["hardware_type"]
    setup = importlib.import_module(
        ".setup", package=f"llm_perf.backends.{hardware_type}"
    )
    logger.info(f"import setup: {setup}")

    xpu_config["sku_name"] = setup.get_device_name()
    with open(workspace.joinpath("config.json"), "w") as f:
        json.dump(xpu_config, f, indent=4)



    random_seed = xpu_config["feature_config"]["random_seed"]
    log_model = xpu_config["feature_config"]["log_model"]

    max_batch_size = xpu_config["deploy_config"]["max_batch_size"]


    # create mp_engine (currently only support tp)
    #   1. tp_size
    #   2. max_batch_size
    engine = setup.get_engine(xpu_config)




    def update_template(mode, batch_size, seq_len):
        if mode == "context":
            input_template = {
                "is_context": True, 

                "input_ids": [[random.randint(1000, 5000) for _ in range(seq_len)] for _ in range(batch_size)], 
                "position_ids": [[i for i in range(seq_len)] for _ in range(batch_size)], 
                "attention_mask": [[1 for _ in range(seq_len)] for _ in range(batch_size)], 

                "valid_slot_ids": [i for i in range(batch_size)], 
                "all_q_len": [seq_len for _ in range(batch_size)], 
                "all_kv_len": [seq_len for _ in range(batch_size)], 

                "get_input_logits": False, 
                "workspace": workspace,
                "log": log_model, 
                "override_hidden_states": True,
                "random_seed": random_seed
            }
        elif mode == "decode":
            input_template = {
                "is_context": False, 

                "input_ids": [[random.randint(1000, 5000)] for _ in range(batch_size)], 
                "position_ids": [[seq_len] for _ in range(batch_size)], 
                "attention_mask": [[1] for _ in range(batch_size)], 

                "valid_slot_ids": [i for i in range(batch_size)], 
                "all_q_len": [1 for _ in range(batch_size)], 
                "all_kv_len": [seq_len for _ in range(batch_size)], 
                
                "get_input_logits": False, 
                "workspace": workspace,
                "log": log_model, 
                "override_hidden_states": True,
                "random_seed": random_seed
            }
        return input_template


    # warmup
    num_warm_iter = 10
    input_template = update_template("context", 1, 1024)
    is_graph = int(os.environ.get("ENABLE_GRAPH", "0"))

    if is_graph:
        #ROCM_HIPGRAPH modify
        input_template['capture'] = 1
        engine.mp_forward(input_template)
        input_template.pop('capture')

    start_time = time.perf_counter_ns()
    for _ in range(num_warm_iter):
        engine.mp_forward(input_template)
    duration_s = round((time.perf_counter_ns() - start_time) / 1e9, 3)
    logger.info(f"warmup cost: {duration_s}s")


    def results_to_csv(file_path, results):
        batch_size_set = sorted(results.keys())
        seq_len_set = set()
        for batch_size in batch_size_set:
            for seq_len in results[batch_size].keys():
                seq_len_set.add(seq_len)
        seq_len_set = sorted(list(seq_len_set))

        with open(file_path, "w") as f:
            csv_writer = csv.writer(f)
            header = ["batch_size\n----------\nseq_len"]
            for batch_size in batch_size_set:
                header.append(f"{batch_size}")
            csv_writer.writerow(header)

            for seq_len in seq_len_set:
                row = [f"{seq_len}"]
                for batch_size in batch_size_set:
                    row.append(f"{results[batch_size][seq_len]}")
                csv_writer.writerow(row)

    log_results = []
    if xpu_config["perf_config"]["perf_context"]:
        batch_size_list = [1]
        seq_len_list = xpu_config["perf_config"]["seq_len_list"]

        context_results = {}
        for batch_size in batch_size_list:
            context_results[batch_size] = {}
            for seq_len in seq_len_list:
                input_template = update_template("context", 1, seq_len)
                if is_graph:
                    #ROCM_HIPGRAPH modify
                    input_template['capture'] = 1
                    engine.mp_forward(input_template)
                    input_template.pop('capture')


                total_test_iter = 20
                start_iters = 2
                test_iter = 0
                duration_ms = 0.
                while test_iter < total_test_iter:
                    result = engine.mp_forward(input_template)
                    if start_iters > 0:
                        start_iters -= 1
                        continue
                    test_iter += 1
                    duration_ms += result["duration_ms"]

                avg_duration_ms = round(duration_ms / test_iter, 3)
                context_results[batch_size][seq_len] = avg_duration_ms
                log_results.append(f"context, batch_size={batch_size}, seq_len={seq_len}, avg_duration_ms={avg_duration_ms}")
                if log_model:
                    log_path = workspace.joinpath("rank_0", "run.log")
                    if log_path.exists():
                        lines = workspace.joinpath("rank_0", "run.log").read_text().splitlines()
                        log_results[-1] += f", {lines[0]}"
                print(log_results[-1])
        results_to_csv(workspace.joinpath("context_perf.csv"), context_results)


    if xpu_config["perf_config"]["perf_decode"]:
        batch_size_list = xpu_config["perf_config"]["batch_size_list"]
        seq_len_list = xpu_config["perf_config"]["seq_len_list"]

        decode_results = {}

        for batch_size in batch_size_list:
            if batch_size > max_batch_size:
                continue
            decode_results[batch_size] = {}
            for seq_len in seq_len_list:
                input_template = update_template("decode", batch_size, seq_len)
                if is_graph:
                    #ROCM_HIPGRAPH modify
                    input_template['capture'] = 1
                    engine.mp_forward(input_template)
                    input_template.pop('capture')

                total_test_iter = 20
                start_iters = 2
                test_iter = 0

                duration_ms = 0.
                while test_iter < total_test_iter:
                    result = engine.mp_forward(input_template)
                    if start_iters > 0:
                        start_iters -= 1
                        continue
                    test_iter += 1
                    duration_ms += result["duration_ms"]

                avg_duration_ms = round(duration_ms / test_iter, 3)

                decode_results[batch_size][seq_len] = avg_duration_ms

                log_results.append(f"decode, batch_size={batch_size}, seq_len={seq_len}, avg_duration_ms={avg_duration_ms}")
                if log_model:
                    log_path = workspace.joinpath("rank_0", "run.log")
                    if log_path.exists():
                        lines = workspace.joinpath("rank_0", "run.log").read_text().splitlines()
                        log_results[-1] += f", {lines[0]}"
                print(log_results[-1])

        results_to_csv(workspace.joinpath("decode_perf.csv"), decode_results)

    # release subprocess resources manually to enable auto garbage collection
    engine.clean_subprocess()

    with open(workspace.joinpath("output.txt"), "w") as f:
        for log in log_results:
            f.write(log + "\n")
    print("\n".join(log_results))


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args.log_level)

    hardware_type = args.hardware_type
    model_config_path = pathlib.Path(args.model_config)
    if not model_config_path.is_absolute():
        model_config_path = CUR_DIR / model_config_path
    if not model_config_path.exists():
        logger.error(f"model_config_path not exist: {model_config_path}")
        sys.exit(-1)

    model_name = model_config_path.stem.split("-")
    model_name = "-".join([model_name[0], model_name[-1]])

    tp_size = args.tp_size
    max_batch_size = args.max_batch_size
    model_version = args.model_version

    if args.workspace is not None:
        workspace = pathlib.Path(args.workspace)
        if not workspace.is_absolute():
            workspace = CUR_DIR / workspace
    else:
        workspace = INFER_ROOT_DIR.joinpath(
            "llm_perf", "reports", 
            hardware_type, model_config_path.stem, 
            "bench_model"
        )
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)


    logger.info(f"hardware_type: {hardware_type}")
    logger.info(f"model_config: {model_config_path}")

    logger.info(f"infer_dtype: {args.infer_dtype}")
    logger.info(f"tp_size: {tp_size}")
    logger.info(f"max_batch_size: {max_batch_size}")
    logger.info(f"model_version: {model_version}")
    logger.info(f"workspace: {workspace}")
    print("\n\n")


    xpu_config = {
        "hardware_type": hardware_type,
        "sku_name": "", 
        "model_name": model_name, 
        "deploy_config": {
            "impl_framework": "bytemlperf", 
            "impl_version": model_version, 
            "infer_dtype": args.infer_dtype,
            "tp_size": tp_size,
            "max_batch_size": max_batch_size,
        }, 
        "model_config": json.loads(model_config_path.read_text()), 
        "perf_config": {
            "perf_context": True, 
            "perf_decode": True,
            "batch_size_list": [1, 4, 8, 16, 24, 32],
            "seq_len_list": [1024, 2048, 4096, 8192]
        }, 
        "feature_config": {
            "random_seed": args.random_seed,
            "log_model": args.log_model,
        }, 
        "tp_size": tp_size,
        "max_batch_size": max_batch_size,
    }

    perf_engine(xpu_config, workspace)
