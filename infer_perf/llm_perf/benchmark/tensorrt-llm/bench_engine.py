import os
import sys
import pathlib
import argparse
import logging
import json
import subprocess

CUR_DIR = pathlib.Path.cwd()
FILE_DIR = pathlib.Path(__file__).parent.absolute()

logger = logging.getLogger("bench_trtllm")

def setup_logger(loglevel: str):
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel.upper())
    logger.propagate = False




def parse_args():
    parser = argparse.ArgumentParser()

    # tensorrt-llm project path
    parser.add_argument("--trtllm_dir", type=str)

    # model engine
    parser.add_argument("--engine_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    # perf config
    parser.add_argument("--batch_size_list", type=str, help="batch_size list, split by comma, \"1,2,4,8,16,32\"")
    parser.add_argument("--seq_len_list", type=str, help="seq_len list, split by comma, \"1024,2048,4096,8192\"")

    # workspace
    parser.add_argument("--workspace", type=str, default=str(CUR_DIR.joinpath("workspace")))

    # logging
    parser.add_argument("--loglevel", type=str, default="INFO")

    args = parser.parse_args()

    setup_logger(args.loglevel)


    # check trtllm
    if args.trtllm_dir is None and os.getenv("TRTLLM_PATH") is None:
        logger.error("trtllm_dir is None, please set trtllm_dir or set TRTLLM_PATH in env")
        sys.exit(-1)
    trtllm_dir = pathlib.Path(args.trtllm_dir) if args.trtllm_dir is not None else pathlib.Path(os.getenv("TRTLLM_PATH")).absolute()
    benchmark_build_dir = trtllm_dir.joinpath("cpp", "build", "benchmarks")
    session_benchmark = benchmark_build_dir.joinpath("gptSessionBenchmark")
    manager_benchmark = benchmark_build_dir.joinpath("gptManagerBenchmark")
    if not benchmark_build_dir.exists() or not session_benchmark.exists() or not manager_benchmark.exists():
        logger.error(f"benchmark_build_dir: {benchmark_build_dir} not exists, please build benckmark first, cd cpp/build/benchmarks && make")
        sys.exit(-1)

    benchmark_dir = trtllm_dir.joinpath("benchmarks", "cpp")
    prepare_dataset_script = benchmark_dir.joinpath("prepare_dataset.py")
    if not benchmark_dir.exists() or not prepare_dataset_script.exists():
        logger.error(f"{prepare_dataset_script} not exists")
        sys.exit(-1)

    # check engine
    engine_dir = pathlib.Path(args.engine_dir).absolute()
    if not engine_dir.exists():
        logger.error(f"engine_dir: {engine_dir} not exists")
        sys.exit(-1)

    # check model
    model_dir = pathlib.Path(args.model_dir).absolute()
    if not model_dir.exists():
        logger.error(f"model_dir: {model_dir} not exists")
        sys.exit(-1)

    # check batch_size_list
    if args.batch_size_list is None:
        logger.error("batch_size_list is None")
        sys.exit(-1)    
    batch_size_list = [int(batch_size) for batch_size in args.batch_size_list.split(",")]

    # check seq_len_list
    if args.seq_len_list is None:
        logger.error("seq_len_list is None")
        sys.exit(-1)
    seq_len_list = [int(seq_len) for seq_len in args.seq_len_list.split(",")]

    # workspace
    workspace = pathlib.Path(args.workspace).absolute()
    if not workspace.exists():
        workspace.mkdir(parents=True)

    return (
        workspace, 
        session_benchmark, manager_benchmark, prepare_dataset_script, 
        engine_dir, model_dir, 
        batch_size_list, seq_len_list
    )



def context_perf(session_benchmark, engine_dir, seq_len_list):
    print("")
    engine_config = engine_dir.joinpath("config.json")

    config_data = json.loads(engine_config.read_text())

    max_batch_size = config_data["build_config"]["max_batch_size"]
    max_input_len = config_data["build_config"]["max_input_len"]
    max_seq_len = config_data["build_config"]["max_seq_len"]
    max_num_tokens = config_data["build_config"]["max_num_tokens"]

    tp_size = config_data["build_config"]["auto_parallel_config"]["gpus_per_node"]
    device_name = config_data["build_config"]["auto_parallel_config"]["cluster_key"]
    device_info = config_data["build_config"]["auto_parallel_config"]["cluster_info"]


    for seq_len in seq_len_list:
        if seq_len > max_num_tokens:
            logger.warning(f"seq_len: {seq_len} > max_num_tokens: {max_num_tokens}, skip")
            continue
        
        run_cmd = f"mpirun --allow-run-as-root -n {tp_size} {session_benchmark}"
        run_cmd += f" --engine_dir {engine_dir}"
        run_cmd += f" --batch_size 1"
        run_cmd += f" --warm_up 2 --num_runs 20"
        run_cmd += f" --input_output_len \"{seq_len},1\""

        results = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        if results.returncode != 0:
            logger.error(f"run cmd: {run_cmd} failed, returncode: {results.returncode}, stderr: {results.stderr}")
            sys.exit(-1)
        for line in results.stdout.splitlines():
            if line.startswith("[BENCHMARK]"):
                try:
                    data_items = line.split()
                    batch_size = int(data_items[2])
                    seq_len = int(data_items[4])
                    output_len = int(data_items[6])
                    latency = float(data_items[8])
                except Exception as e:
                    logger.error(f"parse line: {line} failed, error: {e}")
                    sys.exit(-1)
        logger.info(f"prefill, batch_size: {batch_size}, seq_len: {seq_len}, latency: {latency} ms")



def decode_perf(workspace, manager_benchmark, prepare_dataset_script, engine_dir, model_path, batch_size_list, seq_len_list):
    print("")
    engine_config = engine_dir.joinpath("config.json")

    config_data = json.loads(engine_config.read_text())

    max_batch_size = config_data["build_config"]["max_batch_size"]
    max_input_len = config_data["build_config"]["max_input_len"]
    max_seq_len = config_data["build_config"]["max_seq_len"]
    max_num_tokens = config_data["build_config"]["max_num_tokens"]

    tp_size = config_data["build_config"]["auto_parallel_config"]["gpus_per_node"]
    device_name = config_data["build_config"]["auto_parallel_config"]["cluster_key"]
    device_info = config_data["build_config"]["auto_parallel_config"]["cluster_info"]

    for seq_len in seq_len_list:
        if seq_len > max_num_tokens:
            logger.warning(f"seq_len: {seq_len} > max_num_tokens: {max_num_tokens}, skip")
            continue
        
        seq_workspace = workspace.joinpath(f"seq_{seq_len}")
        seq_workspace.mkdir(parents=True, exist_ok=True)

        context_generate_tokens = 1
        decode_generate_tokens = 101

        context_dataset = seq_workspace.joinpath(f"context_{seq_len}_{context_generate_tokens}.json")
        decode_dataset = seq_workspace.joinpath(f"decode_{seq_len}_{decode_generate_tokens}.json")

        prepare_dataset_cmd = f"python3 {prepare_dataset_script}"
        prepare_dataset_cmd += f" --output {context_dataset}"
        prepare_dataset_cmd += f" --tokenizer {model_path}"
        prepare_dataset_cmd += f" token-norm-dist --num-requests {max_batch_size}"
        prepare_dataset_cmd += f" --input-mean {seq_len} --input-stdev 0"
        prepare_dataset_cmd += f" --output-mean {context_generate_tokens} --output-stdev 0"
        subprocess.run(prepare_dataset_cmd, shell=True, capture_output=True, text=True)

        prepare_dataset_cmd = f"python3 {prepare_dataset_script}"
        prepare_dataset_cmd += f" --output {decode_dataset}"
        prepare_dataset_cmd += f" --tokenizer {model_path}"
        prepare_dataset_cmd += f" token-norm-dist --num-requests {max_batch_size}"
        prepare_dataset_cmd += f" --input-mean {seq_len} --input-stdev 0"
        prepare_dataset_cmd += f" --output-mean {decode_generate_tokens} --output-stdev 0"
        subprocess.run(prepare_dataset_cmd, shell=True, capture_output=True, text=True)

        for batch_size in batch_size_list:
            if batch_size > max_batch_size:
                logger.warning(f"batch_size: {batch_size} > max_batch_size: {max_batch_size}, skip")
                continue

            context_csv = seq_workspace.joinpath(f"context_batch{batch_size}.csv")
            decode_csv = seq_workspace.joinpath(f"decode_batch{batch_size}.csv")

            # context
            run_cmd = f"mpirun --allow-run-as-root -n 8 {manager_benchmark}"
            run_cmd += f" --engine_dir {engine_dir}"
            run_cmd += f" --type IFB"
            run_cmd += f" --max_num_tokens {min(int(seq_len * 1.5), int(max_num_tokens))}"
            run_cmd += f" --max_num_samples {batch_size}"
            run_cmd += f" --static_emulated_batch_size {batch_size}"
            run_cmd += f" --enable_kv_cache_reuse false"
            run_cmd += f" --dataset {context_dataset}"
            run_cmd += f" --output_csv {context_csv}"
            subprocess.run(run_cmd, shell=True, capture_output=True, text=True)

            # decode
            run_cmd = f"mpirun --allow-run-as-root -n 8 {manager_benchmark}"
            run_cmd += f" --engine_dir {engine_dir}"
            run_cmd += f" --type IFB"
            run_cmd += f" --max_num_tokens {min(int(seq_len * 1.5), int(max_num_tokens))}"
            run_cmd += f" --max_num_samples {batch_size}"
            run_cmd += f" --static_emulated_batch_size {batch_size}"
            run_cmd += f" --enable_kv_cache_reuse false"
            run_cmd += f" --dataset {decode_dataset}"
            run_cmd += f" --output_csv {decode_csv}"
            subprocess.run(run_cmd, shell=True, capture_output=True, text=True)

            if context_csv.exists() and decode_csv.exists():
                try:
                    context_latency = float(context_csv.read_text().splitlines()[1].split(",")[2])
                    decode_latency = float(decode_csv.read_text().splitlines()[1].split(",")[2])
                except Exception as e:
                    logger.error(f"parse context_csv: {context_csv} and decode_csv: {decode_csv} failed, error: {e}")
                    continue

            per_token_latency = round((decode_latency - context_latency) / (decode_generate_tokens - context_generate_tokens), 3)
            logger.info(f"decode, batch_size: {batch_size}, seq_len: {seq_len}, latency: {per_token_latency} ms")

        break  


if __name__ == "__main__":
    workspace, session_benchmark, manager_benchmark, prepare_dataset_script, engine_dir, model_dir, batch_size_list, seq_len_list = parse_args()

    logger.info(f"session_benchmark: {session_benchmark}")
    logger.info(f"manager_benchmark: {manager_benchmark}")
    logger.info(f"engine_dir: {engine_dir}")
    logger.info(f"batch_size_list: {batch_size_list}")
    logger.info(f"seq_len_list: {seq_len_list}")

    context_perf(session_benchmark, engine_dir, seq_len_list)
    decode_perf(workspace, manager_benchmark, prepare_dataset_script, engine_dir, model_dir, batch_size_list, seq_len_list)
