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

import tf2onnx
from tf2onnx import tf_loader
import argparse
ONNX_OPSET = 11


def _convert_graphdef_to_onnx(graph_def,
                              inputs=None,
                              outputs=None,
                              output_path='',
                              **kwargs):
    inputs_as_nchw = kwargs.get('inputs_as_nchw', None)
    custom_ops = kwargs.get('custom_ops', None)
    custom_op_handlers = kwargs.get('custom_op_handlers', None)
    custom_rewriter = kwargs.get('custom_rewriter', None)
    extra_opset = kwargs.get('extra_opset', None)
    large_model = kwargs.get('large_model', False)
    name = kwargs.get('name', 'habana_convert')
    target = kwargs.get('target', None)
    shape_override = kwargs.get('shape_override', {})

    tf2onnx.convert.from_graph_def(graph_def,
                                   name=name,
                                   input_names=inputs,
                                   output_names=outputs,
                                   opset=ONNX_OPSET,
                                   custom_ops=custom_ops,
                                   custom_op_handlers=custom_op_handlers,
                                   custom_rewriter=custom_rewriter,
                                   inputs_as_nchw=inputs_as_nchw,
                                   extra_opset=extra_opset,
                                   shape_override=shape_override,
                                   target=target,
                                   large_model=large_model,
                                   output_path=output_path)
    return output_path


def savedmodel_to_onnx(model_path, output_path='', **kwargs):
    inputs = kwargs.get('inputs', None)
    outputs = kwargs.get('outputs', None)
    graph_def, inputs, outputs = tf_loader.from_saved_model(
        model_path, inputs, outputs)
    return _convert_graphdef_to_onnx(graph_def, inputs, outputs, output_path,
                                     **kwargs)


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    savedmodel_to_onnx(args.model_path, args.output_path)
