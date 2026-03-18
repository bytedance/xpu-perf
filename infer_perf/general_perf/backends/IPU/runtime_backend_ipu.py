# Copyright 2023 Graphcore Ltd.
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

import logging
import os

import numpy as np
from poprt.runtime import PackAlgorithm, PackRunnerConfig, RuntimeConfig

from general_perf.backends import runtime_backend

from . import engine_poprt

log = logging.getLogger("RuntimeBackendIPU")


class RuntimeBackendIPU(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(RuntimeBackendIPU, self).__init__()
        self.hardware_type = "IPU"
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.pack_config = None
        self.batch_size = -1
        self.pack_bs = -1
        self.packrunner = False
        self.engine = None
        self.runner_name = "POPRT"
        self.compiled_dir = (
            os.path.split(os.path.abspath(__file__))[0] + "/compiled_models"
        )
        self.precision = "fp32"

    def predict(self, feeds, test_benchmark=False):
        # apply samll adjustments to ipu results to align with cpu's
        self._input_adjustment(feeds)
        results = self.engine.predict(feeds)

        if "videobert" in self.workload["model"]:
            # open_cifar required outputs as: logits_per_image, logits_per_text
            return results["3034"], results["3035"]
        return results

    def _get_engine(self, batch_size):
        if not self.batch_size == batch_size:
            self.update_packrunner_info()
            self.batch_size = batch_size if not self.packrunner else 1

            interact_info = self.configs.get("interact_info", {})
            interact_info.get("runtime_options", {})

            is_pack = interact_info.get("pack_config", False)
            if not is_pack:
                config = RuntimeConfig()
            else:
                config = PackRunnerConfig()
                # set the time out to 0 since the test_accuracy.py does not support async: let the packing timeout in poprt asap
                assert interact_info.get(
                    "pack_config"
                ), "pack mode requires 'pack_config'"
                self.pack_config = interact_info["pack_config"]
                assert (
                    "dynamic_input_name" in self.pack_config
                ), "you must specify the name of the input who has dynamic length."
                assert (
                    "mask_name" in self.pack_config
                ), "you must specify the name of 'mask' input for pack runner."
                assert (
                    "input_names" in self.pack_config
                ), "you must specify all input names for pack runner for auto padding removal."

                mask_name = self.pack_config["mask_name"]

                if self.configs["model"] == "deberta-torch-fp32":
                    config.enable_input_single_row_mode(mask_name, "unpack_info", 1)
                else:
                    config.enable_input_single_row_mode(mask_name)
                config.dynamic_input_name = self.pack_config["dynamic_input_name"]
                config.timeout_microseconds = self.pack_config.get(
                    "timeout_microseconds", 15000
                )
                # best performance mode
                config.algorithm = PackAlgorithm.first_fit
                config.max_valid_num = self.pack_config.get("max_pack_num", 40)
                # remove user provided padded zeros in pack runner
                config.enable_padding_remove_mode(
                    self.pack_config["mask_name"],
                    [n for n in self.pack_config["input_names"] if n != mask_name],
                )

            if self.runner_name == "POPRT":
                if self.engine:
                    del self.engine
                self.engine = engine_poprt.PopRT(self.popef_path, config)
            else:
                raise ValueError("engine_name must be POPRT")
        return self.engine

    def benchmark(self, dataloader):
        report = {}
        report["BS"] = self.batch_size
        interact_info = self.configs.get("interact_info", {})
        if self.packrunner:
            report["BS"] = self.pack_bs
            iterations = self.workload["iterations"]
            qps, avg_latency, tail_latency = self.engine.benchmark_pack(
                interact_info["pack_config"], iterations
            )

        else:
            iterations = self.workload["iterations"]
            clients = interact_info.get("clients", 1)

            qps, avg_latency, tail_latency = self.engine.benchmark(
                clients, self.batch_size, iterations
            )

        report["QPS"] = int(qps)
        report["AVG Latency"] = avg_latency
        report["P99 Latency"] = tail_latency

        return report

    def get_loaded_batch_size(self):
        # return self.workload['batch_sizes'][0]
        return self.batch_size

    def load(self, batch_size) -> None:
        # synchronize configuration updates from compile backend
        self.update_packrunner_info()
        self.precision = (
            self.configs.get("interact_info", {}).get("converter_options", {})
            .get("precision", "FP32")
            .upper()
        )
        if self.packrunner:
            batch_size = self.pack_bs
        self.popef_path = os.path.join(
            self.compiled_dir,
            self.configs["model"],
            str(batch_size),
            "executable_{}.popef".format(self.precision),
        )
        self._get_engine(batch_size)

    def update_packrunner_info(self):
        interact_info = self.configs.get("interact_info", {})
        is_pack = interact_info.get("pack_config", False)
        if not is_pack:
            return
        pack_config = interact_info["pack_config"]
        if is_pack:
            self.packrunner = True
            self.pack_bs = pack_config["batch_size"]

    def _input_adjustment(self, inputs):
        # packing mode require "position_ids" for bert-like models
        if self.packrunner:
            seq_len = np.count_nonzero(inputs[self.pack_config["mask_name"]])
            if self.configs["model"] == "roberta-torch-fp32":
                inputs["position_ids"] = np.arange(seq_len, dtype=np.int32) + 1
            elif self.configs["model"] in ("albert-torch-fp32", "bert-torch-fp32"):
                inputs["position_ids"] = np.arange(seq_len, dtype=np.int32)
            elif self.configs["model"] == "deberta-torch-fp32":
                inputs["unpack_info"] = np.zeros(1, dtype=np.int32)
                inputs.pop("token_type_ids")
