import os
import json
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import torch
import onnxruntime
import time
import numpy as np

from general_perf.backends import compile_backend

log = logging.getLogger("CompileBackendCPU")

pt_dtype_map = {
    "FLOAT32": torch.float32,
    "FLOAT16": torch.float16,
    "INT8": torch.int8,
    "LONG": torch.long
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64
}


class CompileBackendCPU(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendCPU, self).__init__()
        self.hardware_type = 'CPU'
        self.need_reload = False
        self.model_runtimes = []

    def compile(self, config, dataloader=None):
        result = {
            "model":
            config['model_info']['model'],
            "framework":
            config['model_info']['framework'],
            "compile_precision":
            config['model_info']['model_precision'],
            "optimizations":{},
            "instance_count": 1,
            "device_count": 128,
            "input_type":
            config['model_info']['input_type'].split(","),
            "max_batch_size":
            config['model_info']['max_batch_size'],
            "compile_status":
            "success",
            "sg_percent":
            100,
            "segments": [
                {
                    "sg_idx":
                    0,
                    "is_fallback":
                    False,
                    "input_tensor_map":
                    config['model_info']['input_shape'],
                    "output_tensor_map":
                    config['model_info']['outputs'],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": config['model_info']['model_path'],
                        },
                    ],
                },
            ]
        }
        self.configs = result
        self.workload = config['workload']
        self.model_info = config['model_info']
        return result

    def get_interact_profile(self, config):
        model_profile = []
        file_path = "general_perf/backends/CPU/" + self.hardware_type + '.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                model_profile = json.load(f)
        else:
            log.info(
                'File path: {} does not exist, please check'.format(file_path))

        return model_profile

    def get_best_batch_size(self):
        """
        Get Best Batch Size for the model
        """
        return None