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

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import logging
import argparse


def frozen_graph(h5_file_path, workdir, pb_name):
    model = tf.keras.models.load_model(h5_file_path,
                                       custom_objects={
                                           "backend": backend,
                                       })
    model.summary()

    full_model = tf.function(lambda input_1: model(input_1))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=workdir,
                      name=pb_name,
                      as_text=False)
    print('model has been saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='VC model h5->freezedpb script')
    parser.add_argument("--h5_model_path", type=str, required=True)
    parser.add_argument("--freezed_pb_name", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    args = parser.parse_args()
    frozen_graph(args.h5_model_path, args.workdir, args.freezed_pb_name)
