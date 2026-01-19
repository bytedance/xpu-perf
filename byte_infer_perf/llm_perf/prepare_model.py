import os
import sys
import pathlib
import argparse
import subprocess

# ${prj_root}/
BYTE_MLPERF_ROOT = pathlib.Path(__file__).parents[1].absolute()
LLM_PERF_ROOT = BYTE_MLPERF_ROOT.joinpath("llm_perf")

task_map = {
    "chatglm2-torch-fp16-6b": ("chatglm2-6b", "THUDM/chatglm2-6b"), 
    "llama3-torch-bf16-70b": ("llama3-70b", "shenzhi-wang/Llama3-70B-Chinese-Chat"), 
    "falcon-torch-bf16-180b": ("falcon-180b", "tiiuae/falcon-180B"), 
    "mixtral-torch-bf16-8x22b": ("mixtral-8x22b-instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1"),
    "llama3.1-8b-torch-bf16": ("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct")
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="chatglm2-torch-fp16-6b")
    parser.add_argument("--download_model", action="store_true")
    parser.add_argument("--download_baseline", action="store_true")
    args = parser.parse_args()

    os.chdir(LLM_PERF_ROOT)

    task_name = args.task
    if task_name not in task_map:
        print(f"task {task_name} not found, please check your task name")
        sys.exit(-1)

    model_name = task_map[task_name][0]
    model_repo_name = task_map[task_name][1]

    download_path = LLM_PERF_ROOT.joinpath("download")
    download_path.mkdir(parents=True, exist_ok=True)

    if args.download_model:
        sota_model_path = LLM_PERF_ROOT.joinpath("model_zoo", "sota")
        sota_model_path.mkdir(parents=True, exist_ok=True)

        model_path = sota_model_path.joinpath(model_name)
        if model_path.exists():
            print(f"model {model_name} already exists, skip downloading model.")
        else:
            print(f"downloading model {model_name}")
            subprocess.run(
                f"huggingface-cli download --local-dir {model_path} {model_repo_name}", 
                shell=True, check=True
            )

    if args.download_baseline:
        gpu_baseline_path = LLM_PERF_ROOT.joinpath("reports", "base")
        gpu_baseline_path.mkdir(parents=True, exist_ok=True)

        tar_file_name = f"reports_gpu_{task_name}.tar.gz"
        src_path = f"https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/llm/{tar_file_name}"
        dst_path = download_path.joinpath(tar_file_name)

        if dst_path.exists():
            print(f"baseline {model_name} already exists, skip downloading baseline.")
        else:
            print(f"downloading baseline {model_name}")
            subprocess.run(
                f"wget -O {dst_path} {src_path}",
                shell=True, check=True
            )

        base_path = gpu_baseline_path.joinpath(task_name)
        if base_path.exists():
            print(f"baseline {model_name} already exists, skip extracting baseline.")
        else:
            print(f"extracting baseline {model_name}")
            subprocess.run(
                f"tar -xzvf {dst_path} -C {gpu_baseline_path}", 
                shell=True, check=True
            )





