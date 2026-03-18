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

import mxnet as mx

import numpy as np
import onnx


def get_mod(prefix, epoch, ctx, data_shape):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

    mod.bind(for_training=False,
             data_shapes=[("data", data_shape)],
             label_shapes=mod._label_shapes)

    mod.set_params(arg_params, aux_params, allow_missing=True)

    return mod


def load_mxnet():
    prefix = "image_level_space"
    epoch = 0
    ctx = mx.cpu()
    data_shape = (1, 3, 736, 416)

    mod = get_mod(prefix, epoch, ctx, data_shape)

    return mod


'''
require mxnet >= 19.0
'''


def do_mxnet2onnx(sym, params, onnx_file, in_shapes, in_types,
                  dynamic_input_shapes):
    '''
    example:

    sym = 'byte_mlperf/byte_mlperf/download/manysplit/image_level_space-symbol.json'
    params = 'byte_mlperf/byte_mlperf/download/manysplit/image_level_space-0000.params'
    onnx_file = 'manysplit.onnx'

    in_shapes = [(1,3,736,416)]
    in_types = [np.float32]
    dynamic_input_shapes = [(None,3,736,416)]
    '''

    converted_model_path = mx.onnx.export_model(
        sym,
        params,
        in_shapes,
        in_types,
        onnx_file,
        dynamic=True,
        dynamic_input_shapes=dynamic_input_shapes,
        verbose=True)

    # Load the ONNX model
    model_proto = onnx.load_model(converted_model_path)

    # Check if the converted ONNX protobuf is valid
    onnx.checker.check_graph(model_proto.graph)


if __name__ == "__main__":
    # load_mxnet()
    do_mxnet2onnx()
