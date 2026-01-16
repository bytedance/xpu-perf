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

import logging
import os
import time

import numpy as np
import sophon.sail as sail
from general_perf.backends import runtime_backend
import multiprocessing
log = logging.getLogger("RuntimeBackendTPU")

class RuntimeBackendTPU(runtime_backend.RuntimeBackend):
    def __init__(self):
        super().__init__()
        self.hardware_type = "TPU"
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.pack_config = None
        self.batch_size = -1
        self.pack_bs = -1
        self.packrunner = False
        self.engine = None
        self.runner_name = "SAIL"
        self.compiled_dir = (
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/compiled_models/"
        )
        self.precision = "fp32"
        self.max_time=multiprocessing.Value('d', -float('inf'))
        self.min_time=multiprocessing.Value('d', float('inf'))
        self.lock = multiprocessing.Lock()
   
    def version(self) -> str:
        return sail.__version__
    
    def load(self, batch_size) -> None:
        log.warning("TPU Backend only support static batch_size now.")
        self.bmodel_path = self.compiled_dir + self.configs["model"] + ".bmodel"
        # self.input_key = self.configs["input_shape"][self.configs["inputs"]]
        self.dev_id = 1
        self.net = sail.nn.Engine(self.bmodel_path, self.dev_id)
        self.stream = sail.nn.Engine(self.bmodel_path, self.dev_id)
        self.net_name = self.net.get_net_names()[0]
        self.input_name = self.net.get_input_names(self.net_name)[0]
        self.output_names = self.net.get_output_names(self.net_name)
        self.input_shape = self.net.get_input_shapes(self.net_name, 0)[0]
        self.output_shapes = self.net.get_output_shapes(self.net_name, 0)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
    
    def get_loaded_batch_size(self) -> int:
        return self.batch_size
    
    def predict(self, data):
        if isinstance(data, dict):
            input_data = {0: next(iter(data.values()))}
        else:
            input_data = {0: data}

        output_arrays = [np.ndarray(shape=(self.output_shapes[i]), dtype=np.float32) for i in range(len(self.output_shapes))]
        outputs = {i:array for i, array in enumerate(output_arrays)}
        ret = self.net.process(input_data, outputs, self.stream, self.net_name)
        return outputs

    def single_chip_test(self, dev_id, iter, thread_id):
        net = sail.nn.Engine(self.bmodel_path, dev_id)
        stream = sail.nn.Stream(dev_id)
        net_name = net.get_net_names()[0]
        input_shape = net.get_input_shapes(net_name, 0)[0]
        output_shapes = net.get_output_shapes(net_name, 0)
        
        input = np.random.rand(*input_shape).astype(np.float32)
        input_tensor = sail.nn.Tensor(input, sail.DataType.TPU_FLOAT32, dev_id)
        input_data = {0: input_tensor}

        output_arrays = [sail.nn.Tensor(output_shapes[i], sail.DataType.TPU_FLOAT32, dev_id) for i in range(len(output_shapes))]
        outputs = {i:array for i, array in enumerate(output_arrays)}

        start_time=time.time()
        for i in range(iter):
            net.process_async(input_data, outputs, stream, net_name)
        stream.sync()
        end_time=time.time()

        with self.lock:
            self.min_time.value=min(self.min_time.value,start_time)
            self.max_time.value=max(self.max_time.value,end_time)

 
    def _run_benchmark(self, bs, iter):
        chip_num, core_num, start_chip =2, 1, 0
        thread_list = []
        for chip_id in range(chip_num):
            for core_id in range(core_num):
                thread_list.append(multiprocessing.Process(target=self.single_chip_test, args=(chip_id+start_chip, iter, chip_id*core_num+core_id)))

        logging.info("Predict running...")
        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()
        logging.info("Predict finished")

        total_time = self.max_time.value - self.min_time.value

        frame_num = chip_num * core_num * iter
        qps = frame_num / total_time
        avg_latency = total_time / frame_num
        tail_latency = -1
        print(f'chip_num = {chip_num}, core_num = {core_num}, frame_num = {frame_num}, qps = {qps}')  
        
        return qps, avg_latency, tail_latency
    
    def benchmark(self, dataloader):
        report = {}
        report["BS"] = self.batch_size
        interact_info = self.configs.get("interact_info", {})
        iterations = self.workload["iterations"]

        qps, avg_latency, tail_latency = self._run_benchmark(
            self.batch_size, iterations*100
        )

        report["QPS"] = int(qps)
        report["AVG Latency"] = avg_latency
        report["P99 Latency"] = tail_latency

        return report