import os
import json
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import time
import numpy as np
from threading import Thread

from general_perf.backends import runtime_backend

log = logging.getLogger("BackendHPU")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "INT8": torch.int8,
    "LONG": torch.long
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool
}

class RuntimeBackendHPU(runtime_backend.RuntimeBackend):
    def __init__(self):
        super(RuntimeBackendHPU, self).__init__()
        self.hardware_type = 'HPU'
        self.need_reload = False
        self.model_runtimes = []
        self.configs = None
        self.batch_size = -1

    def predict(self, feeds):
        results = {}
        if self.framework == "Pytorch":
            input_tensors = []
            i = 0
            for key, _ in feeds.items():
                if self.input_type[i] == "FLOAT32":
                    datatype = torch.bfloat16
                else:
                    datatype = pt_dtype_map[self.input_type[i]]
                input_tensors.append(
                    torch.tensor(feeds[key],
                                 dtype=datatype).to(
                                     self.device,non_blocking=True))
                i += 1

            import habana_frameworks.torch.core as htcore
            with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
                for model_runtime in self.model_runtimes:
                    results = model_runtime(*input_tensors)
                    htcore.mark_step()
                    htcore.hpu.default_stream().synchronize()
            if isinstance(results, dict):
                for key, val in results.items():
                    results[key] = val.float().cpu().detach().numpy() if val.dtype==torch.bfloat16 else val.cpu().detach().numpy()
            elif isinstance(results, tuple):
                dic = {}
                for i, key in enumerate(self.outputs):
                    dic[key] = list(results)[i]
            else:
                results = {self.outputs[0]: results.float().cpu().numpy() if results.dtype==torch.bfloat16 else results.cpu().numpy()}
        else:
            print("Just test pytorch for now.")
        return results

    def benchmark(self, dataloader):
        iterations = self.workload['iterations']
        batch_size = self.get_loaded_batch_size()
        times_range = []
        report = {}
        report['BS'] = batch_size
        test_data = self._get_fake_samples(
            batch_size, self.configs['segments'][0]['input_tensor_map'],
            self.configs['input_type'])
        enable_profile = False
        if enable_profile:
            warmup_steps = 2
            active_steps = 5
            import habana_frameworks.torch.core as htcore
            prof = torch.profiler.profile(
                   activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU),
                       schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1),
                       on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile/'),
                       record_shapes=False,
                       with_stack=True)


        for _ in range(30):
            self.predict(test_data)
        if enable_profile:
            prof.start()
        for _ in range(iterations):
            start_time = time.time()
            self.predict(test_data)
            end_time = time.time()
            times_range.append(end_time - start_time)
            if enable_profile:
                prof.step()
        if enable_profile:
            prof.stop()

        times_range.sort()
        tail_latency = round(
            times_range[int(len(times_range) * 0.99)] * 1000, 2)
        avg_latency = round(sum(times_range) / iterations * 1000, 2)
        qps = int(1000.0 * batch_size / avg_latency)

        # start_time = time.time()
        # threads = []
        # for i in range(iterations):
            # with torch.hpu.stream(torch.hpu.Stream()):
                # threads.append(Thread(target=self.predict, args=(test_data,)))
                # threads[i].start()
        # for t in threads:
            # t.join()
        # end_time = time.time()

        # qps = int(1000.0 * batch_size * iterations / (end_time-start_time))

        log.info(
            'Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}'.
            format(batch_size, qps, avg_latency, tail_latency))

        report['QPS'] = qps
        report['AVG Latency'] = avg_latency
        report['P99 Latency'] = tail_latency

        return report

    def get_loaded_batch_size(self):
        return self.batch_size

    def load(self, batch_size) -> None:
        self.batch_size = batch_size
        self.model_runtimes = []
        self.input_type = self.configs['input_type']
        self.framework = self.configs['framework']

        self.model_name = self.configs['model']

        import habana_frameworks.torch.core as htcore
        for i, segment in enumerate(self.configs['segments']):
            # there is no input/output meta data i the graph so it need to come from config.
            if not segment['input_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs inputs")
            if not segment['output_tensor_map']:
                raise ValueError("Segment " + str(i) + " needs outputs")

            self.input_shapes = segment['input_tensor_map']
            self.outputs = segment['output_tensor_map'].split(",")

            if self.framework == "Pytorch":
                self.device = torch.device('hpu')
                model = torch.jit.load(
                    segment['compiled_model'][0]['compiled_obj']).to(self.device)
                model.to(torch.bfloat16)
                model.eval()
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph
                model = wrap_in_hpu_graph(model)

            self.model_runtimes.append(model)

    def _get_fake_samples(self, batch_size, shape, input_type):
        data = {}
        if input_type:
            i = 0
            for key, val in shape.items():
                if key != "text":
                    val = [val[0] * batch_size] + val[1:]
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                else:
                    data[key] = np.random.random(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")
