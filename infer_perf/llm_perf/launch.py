# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import random
import argparse
import subprocess
import json
import pathlib
import multiprocessing as mp
import signal
from typing import Any, Dict, Iterable, List
import traceback

# ${prj_root}/
BYTE_MLPERF_ROOT = pathlib.Path(__file__).parents[1]
LLM_PERF_ROOT = BYTE_MLPERF_ROOT.joinpath("llm_perf")
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT.__str__())

from llm_perf.benchmark.bench import benchmark
from llm_perf.utils.logger import logger, setup_logger
from llm_perf.utils.reporter import Reporter, ReportType


class PerfEngine:
    def __init__(self, hardware, task, host, port) -> None:
        super().__init__()

        self.backend_type = hardware
        self.task = task
        self.host = host
        self.port = port

        self.result_queue = mp.Queue()
        self.jobs: List[mp.Process] = []
        self.server_process = None
        self.version = self.get_version()


    def __del__(self):
        self.stop_server()


    def get_version(self):
        version = ""
        try:
            version_file = os.path.join(str(BYTE_MLPERF_ROOT), "../VERSION")
            with open(version_file) as f:
                _version = f.read().splitlines()
            version = '.'.join(v.split('=')[1] for v in _version)
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"get bytemlperf version failed, error msg: {e}")
        return version


    def start_engine(self) -> None:
        # load workload
        workload = load_workload(self.task)

        model_name = workload["model"]

        min_tp_size = int(workload["min_tp_size"])
        test_accuracy = bool(workload["test_accuracy"]) if "test_accuracy" in workload else False
        test_perf = bool(workload["test_perf"]) if "test_perf" in workload else False
        if not any([test_perf, test_accuracy]):
            logger.info(f"End of the llm_perf, enable at least one test item")
            return


        # download model parameter and golden outputs
        download_cmd = f"python3 llm_perf/prepare_model.py --task {self.task} --download_model"
        if test_accuracy:
            download_cmd += " --download_baseline"
        subprocess.run(download_cmd, shell=True)


        # create and start reporter
        self.reporter = Reporter(
            task=self.task,
            backend=self.backend_type,

            tp_size=min_tp_size,
            batch_size=1,
            input_tokens=1024,
            min_new_tokens=1,
            max_new_tokens=512,

            test_perf=test_perf,
            test_accuracy=test_accuracy,

            version=self.version,
        )
        self.reporter.start()

        if test_accuracy:
            accuracy_config = workload["accuracy_config"]

            logger.info("start test accuracy.")
            logger.info(f"using tp_size={min_tp_size}")
            logger.info(f"using batch_size=1")

            self.run_perf(
                accuracy_config, 
                min_tp_size, 1, 1024, 
                ReportType.ACCURACY
            )

        if test_perf:
            perf_config = workload["perf_config"]

            test_tp_sizes = []
            for tp_size in perf_config["tp_sizes"]:
                if tp_size >= min_tp_size:
                    test_tp_sizes.append(tp_size)
            test_batch_sizes = perf_config["batch_sizes"]
            test_input_tokens = perf_config["input_tokens"]
            
            logger.info("start test performance.")
            logger.info(f"tp_sizes list: {test_tp_sizes}")
            logger.info(f"batch_sizes list: {test_batch_sizes}")
            logger.info(f"input_tokens list: {test_input_tokens}")

            for tp_size in test_tp_sizes:
                for batch_size in test_batch_sizes:
                    for input_tokens in test_input_tokens:
                        print("*"*150)
                        print(f"using tp_size={tp_size}, batch_size={batch_size}, input_tokens={input_tokens}")                        
                        print("*"*150)
                        self.run_perf(
                            perf_config,
                            tp_size, batch_size, input_tokens,
                            ReportType.PERFORMANCE,
                        )
                        print("\n\n\n")

        self.reporter.stop()
        self.reporter.summary()


    def run_perf(
        self,
        workload: Dict[str, Any],
        tp_size: int,
        batch_size: int,
        input_tokens: int,
        report_type: ReportType,
    ) -> None:
        # 1. Start server
        self.start_server(tp_size, batch_size)

        # 2. Benchmark clients
        self.start_benchmark(workload, batch_size, input_tokens, report_type)

        # 3. Get result
        alive_clients = batch_size if report_type == ReportType.PERFORMANCE else 1
        started: bool = False
        while alive_clients:
            result = self.result_queue.get()
            if isinstance(result, str) and result == "@start":
                if not started:
                    # Reset reporter mate information
                    self.reporter.update_meta(tp_size, batch_size, input_tokens)
                started = True
                continue
            elif result is None:
                alive_clients = alive_clients - 1
                continue
            self.reporter.submit(result, report_type)

        # 4. Join benchmark client process
        for p in self.jobs:
            p.join()

        # 5. Kill server process
        self.stop_server()

    def start_server(self, tp_size: int, batch_size: int):
        fifo_name = "./server_fifo"
        try:
            os.mkfifo(fifo_name)
        except FileExistsError:
            logger.debug(f"{fifo_name} already exist")

        # create server
        command = [
            "python3", 
            "llm_perf/server/launch_server.py", 
            "--model_config", "llm_perf/model_zoo/" + self.task + ".json", 
            "--hardware_type", self.backend_type, 
            "--tp_size", str(tp_size), 
            "--max_batch_size", str(batch_size), 
            "--port", str(self.port)
        ]
        logger.info(f"Start Server: {' '.join(command)}")
        self.server_process = subprocess.Popen(command, start_new_session=True)

        # wait until server is ready
        with open(fifo_name, "r") as fifo_fd:
            while True:
                data = fifo_fd.readline().strip()
                if data == "Server Ready":
                    break
        os.remove(fifo_name)
        logger.info("Server Ready")

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            logger.info("stopping server process")
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            try:
                self.server_process.wait(timeout=5)
                logger.info("server process has stopped")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                logger.info("server process force killing")
        else:
            # logger already exit
            print(f"server process already exit with {self.server_process.poll()}")


    # launch clients threads
    def start_benchmark(
        self,
        workload: Dict[str, Any],
        batch_size: int,
        input_tokens: int,
        report_type: ReportType,
    ): 
        clients = 1 if report_type == ReportType.ACCURACY else batch_size

        sleep_units = [i for i in range(batch_size)]
        random.shuffle(sleep_units)

        for i in range(clients):
            p = mp.Process(
                target=benchmark,
                args=(
                    i,
                    sleep_units[i], 
                    workload,
                    report_type,
                    input_tokens,
                    self.result_queue,
                    self.host, self.port
                ),
            )
            self.jobs.append(p)
            p.start()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hardware_type", type=str, 
        default="GPU",
        help="The backend going to be evaluted, refs to backends/",
    )
    parser.add_argument(
        "--task", type=str, 
        default="chatglm2-torch-fp16-6b",
        help="The task going to be evaluted, refs to workloads/",
    )

    parser.add_argument(
        "--host", type=str, 
        default="127.0.0.1", 
        help="Host for the gRPC server"
    )
    parser.add_argument(
        "--port", type=int, 
        default=51000, 
        help="port of the server")

    parser.add_argument(
        "--log_level", type=str,
        default=os.environ.get("LOG_LEVEL", "info"),
        help="log level"
    )

    args = parser.parse_args()
    return args



def load_workload(task: str) -> Dict[str, Any]:
    """
    Return a list of dictionary with model Configuration

    Args: List[str]

    Returns: List[dic]
    """
    modules_dir = LLM_PERF_ROOT.joinpath("workloads")

    workload_dict = None
    for filepath in modules_dir.iterdir():
        if filepath.suffix == ".json" and filepath.stem == task:
            with open(filepath) as file:
                workload_dict = json.load(file)
                break
    if workload_dict is None:
        logger.error(f"Task name: {task} was not found, please check your task name")
        exit(-1)
    return workload_dict




if __name__ == "__main__":
    args = parse_args()

    hardware = args.hardware_type
    task = args.task    
    host = args.host
    port = args.port

    setup_logger(args.log_level)

    logger.info(f"hardware: {hardware}")
    logger.info(f"task: {task}")
    logger.info(f"host: {host}")
    logger.info(f"port: {port}")

    instance = PerfEngine(hardware, task, host, port)
    instance.start_engine()
