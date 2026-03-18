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
'''
An Interface to export saved_models to frozen models.

Please notice, this API makes 2 assumptions

    1. saved_model like below:
        |--save-model.pb
        |--variable
        |-- |--variables.data-00000-of-00001
        |-- |--variables.index

    2. saved_tags is tag_constants.SERVING by default if not specific
    3. signature is signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY by default if not specific
Copyright Reserve: Habana Labs
'''

import sys
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import saved_model_cli
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse
from six import StringIO
import contextlib


def freeze_saved_model(saved_model_dir,
                       output_nodes,
                       pb_name,
                       saved_tags=tag_constants.SERVING):
    input_saved_model_dir = saved_model_dir
    output_node_names = output_nodes
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = True
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = saved_tags
    output_graph_filename = pb_name

    freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                              saved_model_tags)


@contextlib.contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr

    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def get_output_node(saved_model_dir, saved_tags, sign):

    parser = saved_model_cli.create_parser()
    args = parser.parse_args([
        'show', '--dir', saved_model_dir, '--tag_set', saved_tags,
        '--signature_def', sign
    ])

    with captured_output() as (out, err):
        saved_model_cli.show(args)

    result = out.getvalue().strip()

    print(result)

    output_num = 0
    output_nodes = None
    lines = result.split('\n')
    for idx, line in enumerate(result.split('\n')):
        if "outputs[" in line:
            line = lines[idx + 3]
            output = line.split(":")[1]
            if output_num > 0:
                output_nodes = output_nodes + "," + output
            else:
                output_nodes = output
            output_num = output_num + 1

    if output_nodes == None:
        raise RuntimeError("No Output Nodes found in saved_model.")

    return output_nodes, output_num


def saved_to_frozen(
    saved_model_dir,
    frozen_path,
    saved_tags=tag_constants.SERVING,
    sign=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):

    output_nodes, output_num = get_output_node(saved_model_dir, saved_tags,
                                               sign)

    output_nodes = output_nodes

    print("[INFO]: Save Model has [", output_num, "] outputs.")
    print("[INFO]: Outputs Nodes: [", output_nodes, "].")

    # cwd = os.getcwd()
    # frozen_path = os.path.join(cwd, "converted_frozen.pb")

    freeze_saved_model(saved_model_dir, output_nodes, frozen_path, saved_tags)

    print("[INFO]: Saved Model convert to Frozen Model done.")
    print("[INFO]: Frozen Model saved here: ", frozen_path)

    return frozen_path


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    saved_to_frozen(args.model_path, args.output_path)
