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
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def convert_pb_to_server_model(pb_model_path, export_dir, input_names,
                               output_names):
    if not input_names:
        raise ValueError("Converter needs inputs")
    if not output_names:
        raise ValueError("Converter needs outputs")
    input_names = input_names.split(",")
    output_names = output_names.split(",")
    graph_def = read_pb_model(pb_model_path)
    convert_pb_saved_model(graph_def, export_dir, input_names, output_names)


def read_pb_model(pb_model_path):
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def convert_pb_saved_model(graph_def, export_dir, input_names, output_names):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    sigs = {}
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        input_infos = {}
        output_infos = {}
        for input_name in input_names:
            input_infos[input_name] = g.get_tensor_by_name(input_name)
        for output_name in output_names:
            output_infos[output_name] = g.get_tensor_by_name(output_name)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                input_infos, output_infos)

        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()


path = "densenet121.pb"
convert_pb_to_server_model(path,
                           os.path.abspath('.') + "/densenet_saved_model",
                           "input_1", "fc1000")
