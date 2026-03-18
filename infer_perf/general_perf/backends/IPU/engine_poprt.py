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
import random
import threading as th
import time
from queue import Queue

import numpy as np
import torch
from poprt import runtime

from . import engine

log = logging.getLogger("engine_poprt")


class PopRT(engine.Engine):
    def __init__(self, popef_path, config):
        self.runner = runtime.Runner(popef_path, config)
        self.packrunner = True if type(config) == runtime.PackRunnerConfig else False

    def predict(self, feeds):
        input_descriptions = self.runner.get_model_inputs()
        for desc in input_descriptions:
            if isinstance(feeds[desc.name], list):
                feeds[desc.name] = np.array(
                    feeds[desc.name], dtype=desc.numpy_data_type()
                )
            elif isinstance(feeds[desc.name], np.ndarray):
                feeds[desc.name] = feeds[desc.name].astype(desc.numpy_data_type())
            elif isinstance(feeds[desc.name], torch.Tensor):
                feeds[desc.name] = (
                    feeds[desc.name].numpy().astype(desc.numpy_data_type())
                )
            else:
                raise TypeError(
                    "The feeds[value] must be list, np.ndarray or torch.Tensor"
                )

        # create the output numpy arrays
        output_descriptions = self.runner.get_model_outputs()
        results = {}
        for output_desc in output_descriptions:
            output_shape = output_desc.shape
            results[output_desc.name] = np.zeros(
                output_shape, dtype=output_desc.numpy_data_type()
            )

        if self.packrunner:
            fut = self.runner.executeAsync(dict(feeds), dict(results))
            fut.wait()
        else:
            self.runner.execute(feeds, results)
        return results

    def benchmark(self, clients, batch_size, iterations):
        input_view = runtime.InputMemoryView()
        input_descriptions = self.runner.get_model_inputs()
        output_descriptions = self.runner.get_model_outputs()
        inputs = {}
        outputs = {}
        for input_desc in input_descriptions:
            inputs[input_desc.name] = np.random.randn(*input_desc.shape).astype(
                input_desc.numpy_data_type()
            )
        for output_desc in output_descriptions:
            outputs[output_desc.name] = np.zeros(
                output_desc.shape, dtype=output_desc.numpy_data_type()
            )

        log.info("Warm up")
        for _ in range(5):
            self.runner.execute(inputs, outputs)
        log.info("Warm up completed, start the time counting")

        q = Queue()

        def perf_count(model_runner, iteration, input_view):
            durations = []
            for _ in range(iteration):
                start_time = time.time()
                self.runner.execute(inputs, outputs)
                end_time = time.time()
                durations.append((start_time, end_time))
            # remove the first and last 20
            if iteration > 40:
                durations = durations[20:-20]
            q.put(durations, timeout=10)

        thp = [
            th.Thread(target=perf_count, args=(self.runner, iterations, input_view))
            for _ in range(clients)
        ]
        for t in thp:
            t.start()
        for t in thp:
            t.join()

        durations_from_th = []
        while not q.empty():
            durations_from_th += q.get()
        max_timestamp = max(y for _, y in durations_from_th)
        min_timestamp = min(x for x, _ in durations_from_th)
        if iterations > 40:
            iterations -= 40  # iterations -40 as line 260
        qps = clients * batch_size * iterations / (max_timestamp - min_timestamp)
        times_range = [y - x for x, y in durations_from_th]

        times_range.sort()
        tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000, 2)
        avg_latency = round(sum(times_range) / len(times_range) * 1000, 2)

        log.info(
            "Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}".format(
                batch_size, int(qps), avg_latency, tail_latency
            )
        )

        np_latency = np.array(times_range) * 1000.0
        log.info(
            f"====== Latency P50: {np.percentile(np_latency, 50)}, P90: {np.percentile(np_latency, 90)}, P99: {np.percentile(np_latency, 99)} ======"
        )

        return qps, avg_latency, tail_latency

    def benchmark_pack(self, pack_config, iterations):
        output_descriptions = self.runner.get_model_outputs()

        outputs = {}
        for output_desc in output_descriptions:
            shape = output_desc.shape
            shape[0] = 1
            outputs[output_desc.name] = np.zeros(
                shape, dtype=output_desc.numpy_data_type()
            )

        # average sequence length in squad is ~172
        avg_len = 172
        max_valid_seq = 384

        bs = pack_config.get("batch_size", 20)
        sample_num = iterations * bs
        input_len = np.random.normal(avg_len, avg_len, size=sample_num).astype(np.int32)
        input_len = np.clip(input_len, 1, max_valid_seq)

        datasets = []
        for s_len in input_len:
            sample = {}
            # set value to 1 does not affect the performance, where attention_mask in pack mode required to be set to 1
            for input_name in pack_config["input_names"]:
                sample[input_name] = np.ones(s_len).astype(np.int32)

            datasets.append(sample)

        # each client sent a single data, one pack batch can pack more than 2*bs of data
        clients = int(bs * 3.5)
        count_percent = 0.6

        q = Queue()

        def perf_count(model_runner, iteration):
            durations = []
            for i in range(sample_num):
                start_time = time.time()
                sample_idx = random.randint(0, sample_num - 1)
                self.runner.execute(datasets[sample_idx], outputs)
                end_time = time.time()
                durations.append((start_time, end_time))
            # remove first and last example's time counter
            ignored_samples = int(sample_num * (1 - count_percent) / 2)
            durations = durations[ignored_samples:-ignored_samples]
            q.put(durations, timeout=10)

        thp = [
            th.Thread(target=perf_count, args=(self.runner, iterations))
            for _ in range(clients)
        ]
        for t in thp:
            t.start()
        for t in thp:
            t.join()

        durations_from_th = []
        while not q.empty():
            durations_from_th += q.get()
        max_timestamp = max(y for _, y in durations_from_th)
        min_timestamp = min(x for x, _ in durations_from_th)
        # iterations -40 as line 260
        qps = clients * (sample_num * count_percent) / (max_timestamp - min_timestamp)
        times_range = [y - x for x, y in durations_from_th]

        times_range.sort()
        tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000, 2)
        avg_latency = round(sum(times_range) / len(times_range) * 1000, 2)

        log.info(
            "Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}".format(
                bs, int(qps), avg_latency, tail_latency
            )
        )

        np_latency = np.array(times_range) * 1000.0
        log.info(
            f"====== Latency P50: {np.percentile(np_latency, 50)}, P90: {np.percentile(np_latency, 90)}, P99: {np.percentile(np_latency, 99)} ======"
        )
        return qps, avg_latency, tail_latency
