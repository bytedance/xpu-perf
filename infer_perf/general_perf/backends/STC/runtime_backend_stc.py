# Copyright 2023 Stream Computing Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
runtime backend of stc
"""

import os
import logging
import numpy as np

from tqdm import tqdm

from stc_ddk import tools
from general_perf.backends import runtime_backend
from run_engine import engine

log = logging.getLogger("RuntimeBackendSTC")
log.setLevel(logging.INFO)

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
}


class RuntimeBackendSTC(runtime_backend.RuntimeBackend):
    """
    STC runtime backend.
    """
    def __init__(self):
        super().__init__()
        self.hardware_type = "STC"
        self.tmpdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "mix_tmp")
        self.best_batch = 1
        self.thread_num = 8
        self.exec = []

    def benchmark(self, dataloader):
        latencies = []
        iterations = self.workload["iterations"]
        print("start benchmark.....")
        self.unload()

        qpses = []
        latencies = []
        print(
            f"stc_benchmark run_params : model_path [{self.local_file}], \
                thread_num [{self.thread_num}], batchs [{str(self.best_batch)}]"
        )
        for _ in tqdm(range(iterations)):
            qps = tools.stc_benchmark(
                self.local_file,
                thread_num=self.thread_num,
                batchs=str(self.best_batch),
                loop_count=100,
                warm_up=100,
            )
            latency = self.thread_num * 1000 / qps
            qpses.append(qps * self.best_batch)
            latencies.append(latency)

        avg_latency = np.mean(latencies)

        report = {}
        report["BS"] = self.best_batch
        report["AVG_Latency"] = round(avg_latency, 2)
        latencies.sort()
        p99_latency = round(latencies[int(len(latencies) * 0.99)], 2)
        report["P99_Latency"] = round(np.mean(p99_latency), 2)
        report["QPS"] = round(np.mean(qpses), 2)
        return report

    def get_loaded_batch_size(self):
        if self.best_batch is None:
            log.error(
                "Not found the best batch_size. Please call pre_optimize to infer it."
            )
        return self.best_batch

    def unload(self):
        """
        Unload model
        """
        if self.exec:
            del self.exec
            self.exec = None

    def load(self, batch_size):
        self.best_batch = self.configs["max_batch_size"]
        model_name = self.model_info["model"]
        local_file = os.path.join(self.tmpdir, model_name)
        self.local_file = local_file
        self.exec = engine(local_file, self.thread_num)
        self.output_names = self.exec.get_output_names()
        self.input_names = self.exec.get_inputs()

        self.suffix = ":0" if list(self.input_names.keys())[0][-2:] == ":0" else ""

    def predict(self, feeds):
        input_types = self.model_info["input_type"].split(",")
        self.suffix = "" if list(feeds.keys())[0][-2:] == self.suffix else self.suffix
        keys = list(feeds.keys())
        new_feeds = {}
        for i, key in enumerate(keys):
            name = key + self.suffix
            new_feeds[name] = np.array(feeds[key], dtype=INPUT_TYPE[input_types[i]])

        stc_out = self.exec.run(new_feeds)
        if isinstance(stc_out, list):
            result = {}
            for idx, out_node_name in enumerate(self.model_info["outputs"].split(",")):
                result[out_node_name] = np.array(stc_out[idx])
            return result
        return stc_out

    def __del__(self):
        self.unload()

    def version(self):
        return "2.3"
