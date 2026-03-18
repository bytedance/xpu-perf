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
compile backend of stc
"""

import os
import ast
import sys
import json
import shutil
import atexit
import logging
import subprocess

from pathlib import Path
from typing import Any, Dict

from general_perf.backends import compile_backend

sys.path.append(os.path.dirname(__file__))
log = logging.getLogger("CompileBackendSTC")
log.setLevel(logging.INFO)


class CompileBackendSTC(compile_backend.CompileBackend):
    """
    STC compile backend.
    """

    TARGET = "stc_tc"

    def __init__(self):
        super().__init__()
        self.frontend_util = None
        self.need_quant = False
        self.hardware_type = "STC"
        self.stc_dtype = "float16"
        self.object_suffix = ".stcobj"
        self.best_batch = 1
        self.tmpdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "mix_tmp")
        self.tmpfiles = set()
        atexit.register(self.__del)

    def __del(self):
        if self.frontend_util is not None:
            self.tmpfiles.update(self.frontend_util.gc())
        for tmpfile in self.tmpfiles:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

    def get_interact_profile(self, config):
        return []

    def version(self):
        return "2.3"

    def pre_optimize(self, configs: Dict[str, Any]):
        logging.root.level = logging.WARNING
        log.info("Running Backend Pre Compilation...")
        self.model_name = configs["model_info"]["model"]
        profile_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "STC.json")
        with open(profile_path) as file_reader:
            model_profile = json.loads(file_reader.read())
        self.best_batch = model_profile[self.model_name]["best_batch"]
        return configs

    def get_best_batch_size(self):
        if self.best_batch is None:
            log.error(
                "Not found the best batch_size. Please call pre_optimize to infer it."
            )
        return self.best_batch

    def __split_expression(self, expression):
        table = {"(", ")", "+", "-", "*", "/", " "}

        def check(words):
            for word in words:
                if word in table or word.isdigit():
                    return False
            return True

        def recursion(words):
            if not words:
                return []
            if check(words):
                return [
                    words,
                ]

            for i, word in enumerate(words):
                if word in table:
                    return recursion(words[:i]) + recursion(words[i + 1 :])
            return []

        return list(set(recursion(expression)))

    def compile(self,
                configs: Dict[str, Any],
                dataloader=None) -> Dict[str, Any]:
        logging.root.level = logging.WARNING
        log.info("Running Backend Compilation...")
        model_name = configs["model_info"]["model"]

        def gen_mix_cmd():
            input_name = configs["model_info"]["inputs"]
            input_shapes = []
            outputs = ""

            for input_name in input_name.split(","):
                shapes = configs["model_info"]["input_shape"][input_name]
                new_shape = []
                for shape in shapes:
                    if isinstance(type, str):
                        for name in self.__split_expression(shape):
                            if name not in configs["model_info"]:
                                log.error("Not found %s in configs", name)
                            shape = shape.replace(name, f'config["model_info"]["{name}"]')
                        shape = ast.literal_eval(shape)
                    new_shape.append(shape)
                new_shape[0] *= self.best_batch
                input_shapes.append("[" + ",".join(str(val) for val in new_shape) + "]")

            input_shapes = ",".join(val for val in input_shapes)
            outputs = configs["model_info"]["outputs"]
            output_num = len(configs["model_info"]["outputs"].split(","))
            input_dtypes = configs["model_info"]["input_type"]
            output_dtypes = ",".join(
                configs["model_info"]["model_precision"] for _ in range(output_num)
            )

            res_path = os.path.join(self.tmpdir, model_name)

            input_names = configs["model_info"]["inputs"]

            out_cmd = [
                "stc_ddk.stc_aic",
                "--model",
                configs["model_info"]["model_path"],
                "--input_names",
                input_names,
                "--input_shapes",
                input_shapes,
                "--input_dtypes",
                input_dtypes,
                "--output_names",
                outputs,
                "--output_dtypes",
                output_dtypes,
                "--outdir",
                res_path,
            ]

            return out_cmd, res_path

        out_cmd, res_path = gen_mix_cmd()

        if os.path.exists(os.path.join(res_path, "model.json")):
            log.info("Stcobj has exists, skip compile.")
        else:
            if os.path.exists(res_path):
                shutil.rmtree(res_path)
            try:
                log.info(" ".join(str(val) for val in out_cmd))
                subprocess.call(out_cmd)
            except Exception:
                pass

        with open(os.path.join(res_path, "model.json")) as file_reader:
            compiled_model_info = json.loads(file_reader.read())

        compile_info = {
            "model": configs["model_info"]["model"],
            "framework": configs["model_info"]["framework"],
            "compile_precision": "fp16",
            "input_type": configs["model_info"]["input_type"],
            "max_batch_size": self.best_batch,
            "sg_percent": compiled_model_info["stcop_rate"],
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": configs["model_info"]["input_shape"],
                    "output_tensor_map": configs["model_info"]["outputs"],
                    "compiled_model": [
                        {
                            "compiled_bs": self.best_batch,
                            "compiled_obj": configs["model_info"]["model_path"],
                        },
                    ],
                },
            ],
        }
        self.workload = configs["workload"]
        self.model_info = configs["model_info"]

        if not os.path.exists(res_path) or self.__check_aic(res_path):
            run_cmd = " ".join(str(val) for val in out_cmd)
            log.error("model convert error. run_cmd is : %s", run_cmd)
            compile_info["compile_status"] = "failed"
        else:
            compile_info["compile_status"] = "success"
        return compile_info

    def __check_aic(self, res_path):
        """ Check whether the compilation was successful. """
        aic_fail_flag = False
        res_path = Path(res_path)
        json_file = res_path / "model.json"
        if json_file.exists():
            with open(str(json_file), "r") as file_reader:
                model_info = json.load(file_reader)
            if len(model_info["nodes"]) > 0:
                for node in model_info["nodes"]:
                    file = res_path / node["source"]
                    if not file.exists():
                        aic_fail_flag = True
                        break
            else:
                aic_fail_flag = True
        else:
            aic_fail_flag = True
        return aic_fail_flag
