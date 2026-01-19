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


import copy
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict
import tpu_mlir
import shutil

from general_perf.backends import compile_backend

log = logging.getLogger("CompileBackendTPU")

class CompileBackendTPU(compile_backend.CompileBackend):
    def __init__(self):
        super().__init__()
        self.hardware_type = "TPU"
        self.need_reload = False
        self.need_quant = False
        self.current_dir = os.path.split(os.path.abspath(__file__))[0]
        self.model_config = None
        self.precision = "fp32"
        self.model_precision = "F32"
        self.mean = "0.0,0.0,0.0"
        self.scale = "1.0,1.0,1.0"
        self.pixel_format = "rgb"
        self.input_num = 200
        
    def version(self) -> str:
        """
        Return compile backend version details
        """
        return tpu_mlir.distribution
    
    def pre_optimize(self, configs: Dict[str, Any]):
        """Model pre-optimization interface.

        Requirements: Model pre-optimization
        cannot change the model format. Torch model export to ONNX is allowed.
        """
        
        return configs
    
    def compile(self, configs: Dict[str, Any], dataloader=None) -> Dict[str, Any]:
        if not self.model_config:
            self.model_config = configs
        
        self.model_info = configs["model_info"]
        self.interact_info = configs["interact_info"]
        self.model_path = self.model_info["model_path"]
        self.input_shapes = self.model_info["input_shape"][self.model_info["inputs"]]
        self.input_shapes_str = ','.join(str(num) for num in self.input_shapes)
        self.model_name = self.model_info["model"]
        if("model_precision" in self.interact_info.keys()):
            self.model_precision = self.interact_info["model_precision"]
            self.mean = self.interact_info["mean"]
            self.scale = self.interact_info["scale"]
            self.pixel_format = self.interact_info["pixel_format"]
            self.input_num = self.interact_info["input_num"]

        self.precision=self.model_precision.upper()
        gen_mlir_commands = f'model_transform \
            --model_name {self.model_name} \
            --model_def ../../{self.model_path} \
            --mean {self.mean} \
            --scale {self.scale} \
            --pixel_format {self.pixel_format}  \
            --input_shapes [[{self.input_shapes_str}]] \
            --mlir {self.model_name}.mlir'
        gen_mlir_logs = './model_transform.log'

        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        origin_dir = os.getcwd()
        self.compile_dir_name = current_dir + '/compiled_models/'
        if os.path.exists(self.compile_dir_name):
            shutil.rmtree(self.compile_dir_name)
        os.mkdir(self.compile_dir_name)
        os.chdir(self.compile_dir_name)
        with open(gen_mlir_logs, 'w') as logfile:
            subprocess.call(gen_mlir_commands, stdout=logfile, stderr=subprocess.STDOUT, shell=True)
        if(self.precision == "INT8"):
            self.dataset_path = current_dir+"/datasets/"+self.model_info["dataset_name"]+"/"+self.interact_info["dataset_path"]

            run_calibration_commands = f'run_calibration {self.model_name}.mlir \
                --dataset {self.dataset_path} \
                --input_num {self.input_num}  \
                -o {self.model_name}_cali_table'
                
            run_calibration_logs = './run_calibration.log'

        
            with open(run_calibration_logs , 'w') as logfile:
                subprocess.call(run_calibration_commands, stdout=logfile, stderr=subprocess.STDOUT, shell=True)
        
            deploy_commands = f'model_deploy \
                --mlir {self.model_name}.mlir \
                --quantize {self.model_precision} \
                --chip bm1690 \
                --calibration_table {self.model_name}_cali_table \
                --model {self.model_name}.bmodel'
        else:
            deploy_commands = f'model_deploy \
                --mlir {self.model_name}.mlir \
                --quantize {self.model_precision} \
                --chip bm1690 \
                --model {self.model_name}.bmodel'
        deploy_commands_logs = './model_deploy.log'
        
        with open(deploy_commands_logs, 'w') as logfile:
            subprocess.call(deploy_commands, stdout=logfile, stderr=subprocess.STDOUT, shell=True)
        
        os.chdir(origin_dir)
        
        result = {
            "model": self.model_name,
            "framework": configs["model_info"]["framework"],
            "compile_precision": self.precision,
            "input_type": configs["model_info"]["input_type"].split(","),
            "max_batch_size": configs["model_info"]["max_batch_size"],
            "compile_status": "success",
            "optimizations": {},
            "instance_count": 1,
            "device_count": 1,
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": configs["model_info"]["input_shape"],
                    "output_tensor_map": configs["model_info"]["outputs"],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": configs["model_info"]["model_path"],
                        },
                    ],
                },
            ],
            "interact_info": self.model_config,
        }
        
        return result
    
    def get_interact_profile(self, config: Dict[str, Any]):
        """Collect information for core engine to let user interactively fill in configurations."""
        # load the interact_info by model name
        model_profile = []

        interact_info_file = os.path.join(
            self.current_dir, "interact_infos", config["model_info"]["model"] + ".json"
        )
        file_path = os.path.join(self.current_dir, self.hardware_type + ".json")

        with open(interact_info_file, "r") as f:
            interact_info = json.load(f)

       

        return interact_info
    
    def get_best_batch_size(self) -> compile_backend.List[int] | None:
        return None