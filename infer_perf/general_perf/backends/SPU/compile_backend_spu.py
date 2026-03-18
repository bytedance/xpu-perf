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
import json
import logging
from typing import Any, Dict, List, Optional
from general_perf.backends import compile_backend
from base_compile import Resnet50Builder, BertBaseBuilder, AlbertBuilder, RobertaBuilder, ConformerBuilder, GeneralBuilder

log = logging.getLogger("CompileBackendSPU")


class CompileBackendSPU(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendSPU, self).__init__()
        self.hardware_type = "SPU"
        self.batch_size = None
        self.model_name = ""
        self.configs = None
        self.workload = None
        self.model_info = None
        self.model = None
        self.interact_info = None

    def compile(self,
                configs: Dict[str, Any],
                dataloader=None) -> Dict[str, Any]:
        """
        Model compilation interface. Model conversion and compilation 
        can be performed here. The model format can be changed here.
        """
        model_info = configs["model_info"]
        name = model_info ['model']
        builder_dict = {
            "resnet50-torch-fp32": Resnet50Builder,
            "bert-torch-fp32": BertBaseBuilder,
            "albert-torch-fp32": AlbertBuilder,
            "roberta-torch-fp32": RobertaBuilder,
            "conformer-encoder-onnx-fp32": ConformerBuilder
        }

        if name in builder_dict:
            SparserBuilder = builder_dict[name]
        else:
            SparserBuilder = GeneralBuilder
        interact_info = self.get_interact_profile(configs)
        onnx_path = interact_info["onnx_path"]
        dump_dir=os.path.dirname(os.path.abspath(interact_info["model_path"]))
        dataset_dir = interact_info["calibration_dir"]
        dataset_cfg = interact_info["transform_file"]
        model_precision = interact_info["model_precision"]
        batch_size = interact_info["batch_size"]
        verify = interact_info["verify"]

        builder = SparserBuilder(onnx_path, dump_dir, dataset_dir, dataset_cfg, model_precision, batch_size, verify)
        compile_info = builder()

        result = {
            "model": configs["model_info"]["model"],
            "framework": configs["model_info"]["framework"],
            "compile_precision": model_precision,
            "input_type": configs["model_info"]["input_type"].split(","),
            "max_batch_size": configs["workload"]["batch_sizes"][-1],
            "compile_status":"success",
            "sg_percent": 100,
            "sparsity_ratio":compile_info["sparsity_ratio"],
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": configs["model_info"]["input_shape"],
                    "output_tensor_map": configs["model_info"]["outputs"],
                    "compiled_model": [
                        {
                            "compiled_bs": batch_size,
                            "compiled_obj": dump_dir,
                        },
                    ],
                },
            ],
            "interact_info": interact_info,
        }

 
        return result

    def get_interact_profile(self, configs: Dict[str, Any]):

        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"interact_info/{configs['model_info']['model']}.json")
        if os.path.exists(file_path):
            with  open(file_path, 'r') as f:
                model_profile = json.load(f)
            return model_profile
        else:
            log.info('File path: {} does not exist, please check'.format(file_path))
            raise NotImplementedError("CompileBackend:get_interact_profile")

    def get_best_batch_size(self) -> Optional[List[int]]:
        """
        Get Best Batch Size for the model
        """
        return [1]
