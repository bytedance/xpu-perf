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

import argparse
import json

import numpy as np
import torch


def torch_to_onnx(model_path, output_path):
    model_name = output_path.split("/")[-1][:-4]
    with open("general_perf/model_zoo/" + model_name + "json", "r") as f:
        model_info = json.load(f)
    model_inputs = model_info["inputs"].split(",")
    input_shapes = model_info["input_shape"]
    input_type = model_info["input_type"].split(",")
    example_inputs = _get_fake_samples(input_shapes, input_type)

    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    model.eval()

    names = model_inputs
    dynamic_inputs = {}
    for i in range(len(names)):
        dynamic_inputs[names[i]] = {0: "batch_size"}
    outputs = model_info["outputs"].split(",")
    for output in outputs:
        dynamic_inputs[output] = {0: "batch_size"}
    torch.onnx.export(
        model,
        example_inputs,
        output_path,
        opset_version=11,
        input_names=names,
        output_names=outputs,
        dynamic_axes=dynamic_inputs,
    )


def _get_fake_samples(shape, type):
    data = []
    idx = 0
    for key, val in shape.items():
        val = [val[0] * 1] + val[1:]
        data.append(torch.from_numpy(np.random.random(val).astype(type[idx].lower())))
        idx += 1
    return data


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    torch_to_onnx(args.model_path, args.output_path)
