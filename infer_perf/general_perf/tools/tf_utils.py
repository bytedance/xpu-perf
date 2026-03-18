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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.core import framework
from tensorflow.core.framework import types_pb2, graph_pb2, attr_value_pb2
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf import text_format
import numpy as np


def isTextProtobuf(filename):
    """ Returns whether a filename is a text protobuf based on the file extension.

    Args:
        filename: string - file name to process.

    Returns:
        true if `filename`'s extension is .pbtxt, false otherwise.
    """

    retval = False

    _, filename_ext = os.path.splitext(filename)
    if filename_ext and filename_ext.lower() == ".pbtxt":
        retval = True

    return retval


def saveGraphProtobufToFile(file_name, graph_d):
    """ Saves a `GraphDef` protocol buffer graph to a file.

    Args:
        file_name: string - name of the file where to write the graph.
        graph_d: The `GraphDef` protocol buffer to save.
    """
    output_file_name_no_dir = os.path.basename(file_name)
    output_file_dir = os.path.dirname(file_name)
    tf.io.write_graph(graph_d,
                      output_file_dir,
                      output_file_name_no_dir,
                      as_text=isTextProtobuf(file_name))


def loadGraphProtobufFromFile(file_name):
    """ Loads a `GraphDef` protocol buffer graph from a file.

    Args:
        file_name: string - name of the file to load.

    Returns:
        A `GraphDef` protocol buffer loaded from the file.
    """
    graph_d = framework.graph_pb2.GraphDef()
    with open(file_name, "rb") as f:
        if isTextProtobuf(file_name):
            # for text file:
            text_format.Merge(f.read(), graph_d)
        else:
            # for binary file:
            graph_d.ParseFromString(f.read())
    return graph_d


def duplicateGraph(graph_d):
    """ Creates a deep copy of a tf GraphDef.

    Args:
        graph_d: A `GraphDef` protocol buffer to duplicate.

    Returns:
        A deep copy of the specified tf GraphDef.
    """

    with tf.Graph().as_default() as tmp_graph:
        _ = tf.import_graph_def(graph_d, name="")
        return tmp_graph.as_graph_def()


def getNodeNames(nodes_d):
    """ Compiles a list of strings representing all the name of
    the nodes in the specified list of nodes.

    Args:
        nodes_d: List of `NodeDef` objects to process.

    Returns:
        A list of strings representing all the name of the nodes in `nodes_d`.
    """
    return [node_d.name for node_d in nodes_d]


def getNodeIndexByName(nodes_d, node_name):
    """ Finds the NodeDef node in list of NodeDef corresponding to
    the specified name.

    Args:
        nodes_d: List of `NodeDef` objects to process.
        node_name: node to find.

    Returns:
        And integer index representing the index of the node in the list
        passed or -1 if not found.
    """

    retval = -1
    for i, node_d in enumerate(nodes_d):
        if node_d.name == node_name:
            retval = i
            break
    return retval


def getNodeInputNamesClean(node_input_names):
    retval = []
    for input_name in node_input_names:
        tensor_idx = input_name.rfind(":")
        if tensor_idx < 0:
            retval.append(input_name)
        else:
            retval.append(input_name[:tensor_idx])
    return retval


def getNodeByName(nodes_d, node_name):
    """ Finds the NodeDef node in list of NodeDef corresponding to
    the specified name.

    Args:
        nodes_d: List of `NodeDef` objects to process.
        node_name: node to find.

    Returns:
        The `NodeDef` node in `nodes_d` corresponding to the specified name,
        or None if name is not found in `nodes_d`.
    """

    retval = getNodeIndexByName(nodes_d, node_name)
    if (retval < 0):
        retval = None
    else:
        retval = nodes_d[retval]
    return retval


def getInputNodeNames(graph_d):
    """ Finds the placeholder nodes (or inputs) in the graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.

    Returns:
        A list of node names corresponding to all nodes that are
        inputs to the graph.
    """

    retval = []
    for node_d in graph_d.node:
        if node_d.op == "Placeholder":
            retval.append(node_d.name)
    return retval


def getOutputNodeNames(graph_d):
    """ Finds the nodes that are leaf nodes (or outputs) in the graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.

    Returns:
        A list of node names corresponding to all nodes that are
        leaf nodes (or outputs) in the graph.
    """

    non_output_node_names = set()
    for node_d in graph_d.node:
        non_output_node_names = non_output_node_names | set(
            getNodeInputNamesClean(node_d.input))
    graph_node_names = set(getNodeNames(graph_d.node))
    return list(graph_node_names - non_output_node_names)


def getNodesInOutput(graph_d, node_name):
    """ Finds all nodes that use the output of specified node as
    their input in the specified graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.
        node_name: String name of node to check.

    Returns:
        A list of node names corresponding to all nodes that use the
        output of specified node as their input.
    """
    retval = []

    for node_d in graph_d.node:
        node_input_names = getNodeInputNamesClean(node_d.input)
        for id, input_name in enumerate(node_input_names):
            if input_name == node_name:
                retval.append([id, node_d.name])
                break

    return retval


def getNodesInSubGraph(graph_d, start_nodes, end_nodes):
    subgraph = []
    for node in start_nodes:
        subgraph.append(node)

    successor = start_nodes
    while len(successor) != 0:
        for node in successor:
            tmp_suc = getNodesInOutput(graph_d, node)
            for suc in tmp_suc:
                if suc in subgraph:
                    continue
                else:
                    subgraph.append(suc)
        successor = tmp_suc

    return subgraph


def convertTensorflow2NumpyShape(shape_tf):
    """ Converts a tensorflow `TensorShape` to a numpy shape.
    All unknown values for partial shapes will be converted to -1.

    Args:
        shape_tf: A `TensorShape` object to convert.

    Returns:
        A list of values representing a valid numpy style shape.
    """
    retval = [
        shape_val if shape_val is not None else -1
        for shape_val in shape_tf.as_list()
    ]
    return retval


def convertNumpy2TensorflowShape(shape_np):
    """ Converts a numpy shape to a tensorflow shape.
    All unknown (-1) values for partial shapes will be converted to None.

    Args:
        shape_np: A list of values representing a valid numpy shape.

    Returns:
        A list of values representing a valid tensorflow style shape.
    """
    retval = [shape_val if shape_val >= 0 else None for shape_val in shape_np]
    return retval


def getInputShape(graph_d, numpy_format=False):
    """ Retrieves the shape of all inputs to specified `GraphDef` object.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.
        numpy_format: boolean - if False (default), shape is given in tensorflow format,
            otherwise, numpy format.

    Returns:
        A mapping string => list: from input tensor name to shape.
    """

    retval = {}

    input_node_names = getInputNodeNames(graph_d)

    tf.import_graph_def(graph_d, name="")
    for input_node_name in input_node_names:
        # find all output tensors for this placeholder, i.e. input:0, input:1, etc.
        try:
            i = 0
            while True:
                input_tensor_name = input_node_name + ":" + str(i)
                next_input_tensor = tf.get_default_graph().get_tensor_by_name(
                    input_tensor_name)
                tensor_shape = next_input_tensor.shape
                if numpy_format:
                    tensor_shape = convertTensorflow2NumpyShape(tensor_shape)
                retval[input_tensor_name] = tensor_shape
                i += 1
        except:
            pass  # reached the end of the placeholder outputs

    return retval


def getInputOutputNodes(frozen_graph):
    """ Finds all input and output nodes in the specified graph.

    Args:
        frozen_graph: TensorFlow frozen graph

    Returns:
        A list of input and output node names.
    """
    predefined_inputs = ['segment', 'mask', 'input_ids']
    graph_d = loadGraphProtobufFromFile(frozen_graph)
    inputs = getInputNodeNames(graph_d)
    outputs = getOutputNodeNames(graph_d)
    nodes = [
        str for str in inputs if any(sub in str for sub in predefined_inputs)
    ]
    if len(nodes) == len(predefined_inputs):
        return [inputs, outputs]
    else:
        status, inputs = findNodeByName(graph_d, predefined_inputs)
        if status:
            return [inputs, outputs]
        else:
            raise RuntimeError(
                "Cannot find suitable inputs for this tool, please indicate the names of inputs after preprocessing"
            )


def findNodeByName(graph_d, node_name):
    """ Finds nodes specified by name in the specified graph.

    Args:
        graph_d: A `GraphDef` protocol buffer to process.
        node_name: String name of node to check.

    Returns:
        status - True if all nodes are found, False otherwise
        A list of node names.
    """
    status = False
    all_nodes = list(getNodeNames(graph_d.node))
    retval = [str for str in all_nodes if any(sub in str for sub in node_name)]
    if len(node_name) == len(retval):
        status = True

    return status, retval


def load_graph(model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        if model_path.endswith("pb"):
            with open(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
        else:
            with open(model_path, "r") as pf:
                text_format.Parse(pf.read(), graph_def)
        return graph_def


from opt_tf import *
import os
import tensorflow as tf
import sys
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import saved_model_cli
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.tools.graph_transforms import TransformGraph
from six import StringIO, iteritems
import contextlib

from tensorflow.core.framework import types_pb2, tensor_shape_pb2, graph_pb2, attr_value_pb2

import numpy as np
from load_runstep import load_runstep


def load_graph(model):
    graph_def = tf.GraphDef()

    print("load model: ", model)
    with open(model, 'rb') as f:
        graph_def.ParseFromString(f.read())

    return graph_def


def find_node(graph_def, name):
    node = None
    for n in graph_def.node:
        if n.name == name:
            node = n
            break
    # if node == None:
    #     print('Node {} not found'.format(name))

    return node


def find_node_by_type(graph_def, type):
    node = []
    for n in graph_def.node:
        if n.op == type:
            node.append(n)
    return node


def get_node_successor(graph_def, node_name):
    outputs = []
    for n in graph_def.node:
        for input in n.input:
            if node_name == input.split(':')[0]:
                outputs.append(n)

    # if len(outputs) == 0:
    #     print("[INFO] {} has no successor".format(node_name))

    return outputs


def get_node_output(graph_def, node_name):
    outputs = []
    for n in graph_def.node:
        for input in n.input:
            if node_name == input.split(':')[0]:
                if len(input.split(':')) == 1:
                    if not input + ":0" in outputs:
                        outputs.append(input + ":0")
                else:
                    if not input in outputs:
                        outputs.append(input)

    # if len(outputs) == 0:
    #     print("[INFO] {} has no output".format(node_name))

    return outputs


# single in & singel out


def remove_nodes(graph_d, nodes):
    for node in nodes:
        # assert len(node.input) == 1
        pre_node = node.input[0]

        succ_nodes = get_node_successor(graph_d, node.name)
        for succ in succ_nodes:
            for idx, name in enumerate(succ.input):
                if name == node.name:
                    succ.input[idx] = pre_node

        graph_d.node.remove(node)

    return graph_d


def create_shape_proto(shape):
    shape_proto = tensor_shape_pb2.TensorShapeProto()
    for dim in shape:
        shape_proto.dim.add().size = dim
    return attr_value_pb2.AttrValue(shape=shape_proto)


def set_shape(node, shape):
    node.attr["shape"].CopyFrom(create_shape_proto(shape))


def remove_control_dep(graph_def):
    # reset & import
    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name="")

    for node in graph_def.node:
        op = tf.get_default_graph().get_operation_by_name(node.name)
        if len(op.control_inputs) != 0:
            tf.contrib.graph_editor.remove_control_inputs(
                op, op.control_inputs)

    graph_def = tf.get_default_graph().as_graph_def()
    return graph_def


def is_leaf_node(graph_d, name):
    for n in graph_d.node:
        for in_n in n.input:
            if name == in_n or name == in_n.split(":0")[0]:
                return False
    return True


def get_node_shape(node):
    return [d.size for d in node.attr["shape"].shape.dim]


def get_graph_input(graph_d):
    in_node = []
    for n in graph_d.node:
        if n.op == "Placeholder":
            in_node.append(n.name)

    to_remove = []
    for in_n in in_node:
        if is_leaf_node(graph_d, in_n):
            to_remove.append(in_n)

    for name in to_remove:
        node = find_node(graph_d, name)
        graph_d.node.remove(node)

    real_in = set(in_node) - set(to_remove)

    return list(real_in)


def get_graph_output(graph_d):
    out_node = []
    for n in graph_d.node:
        if len(get_node_successor(graph_d, n.name)) == 0:
            out_node.append(n.name)

    # if len(out_node) == 0:
    #     print("[INFO] Graph No Outputs??")

    return out_node


def get_constant_val(node):
    val = tf.make_ndarray(node.attr["value"].tensor)
    return val


def get_dtype_from_np(val):
    if val.dtype == np.int32:
        return types_pb2.DT_INT32

    if val.dtype == np.float32:
        return types_pb2.DT_FLOAT

    if val.dtype == np.int64:
        return types_pb2.DT_INT64

    if val.dtype == np.float16:
        return types_pb2.DT_HALF

    raise ValueError("DTYPE {} NOT SUPPORTEED!".format(val.dtype))


def set_constant_val(node, val):
    tf_dtype = get_dtype_from_np(val)
    node.attr["value"].tensor.CopyFrom(
        tf.make_tensor_proto(val, dtype=tf_dtype))


@contextlib.contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr

    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def get_saved_input_node(saved_model_dir, saved_tags, sign):

    parser = saved_model_cli.create_parser()
    args = parser.parse_args([
        'show', '--dir', saved_model_dir, '--tag_set', saved_tags,
        '--signature_def', sign
    ])

    with captured_output() as (out, err):
        saved_model_cli.show(args)

    result = out.getvalue().strip()

    input_tensors = []

    lines = result.split('\n')
    for idx, line in enumerate(result.split('\n')):
        if "inputs[" in line:
            line = lines[idx + 3]
            input = line.split(":")[1]
            input_tensors.append(input.strip() + ":0")
    return input_tensors


def get_saved_output_node(saved_model_dir, saved_tags, sign):

    parser = saved_model_cli.create_parser()
    args = parser.parse_args([
        'show', '--dir', saved_model_dir, '--tag_set', saved_tags,
        '--signature_def', sign
    ])

    with captured_output() as (out, err):
        saved_model_cli.show(args)

    result = out.getvalue().strip()

    # print(result)

    output_nodes = []
    lines = result.split('\n')
    for idx, line in enumerate(result.split('\n')):
        if "outputs[" in line:
            line = lines[idx + 3]
            output = line.split(":")[1]
            output_nodes.append(output.strip() + ":0")

    return output_nodes


def duplicate_const(graph_d):
    all_consts = find_node_by_type(graph_d, "Const")

    need_duplicate = []
    for node in all_consts:
        if len(get_node_successor(graph_d, node.name)) > 1:
            need_duplicate.append(node.name)

    for node in need_duplicate:
        succ_nodes = get_node_successor(graph_d, node)

        for idx, succ in enumerate(succ_nodes):
            ori_node = find_node(graph_d, node)

            new_node = graph_d.node.add()
            new_node.op = ori_node.op
            new_node.name = ori_node.name + "new_{}".format(idx)
            new_node.input.extend(ori_node.input)
            attrs = list(ori_node.attr.keys())
            for attr in attrs:
                new_node.attr[attr].CopyFrom(ori_node.attr[attr])

            for i, input in enumerate(succ.input):
                if input == ori_node.name:
                    succ.input[i] = new_node.name

    return graph_d


def rewrite_batch_norm_node_v2(node, graph_def, target_type):
    """
    Rewrite FusedBatchNorm with FusedBatchNormV2 for reserve_space_1 and reserve_space_2 in FusedBatchNorm require float32 for 
    gradient calculation (See here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fused-batch-norm)
    """
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp32':
        dtype = types_pb2.DT_FLOAT

    new_node = graph_def.node.add()
    new_node.op = "FusedBatchNormV2"
    new_node.name = node.name
    new_node.input.extend(node.input)
    new_node.attr["U"].CopyFrom(
        attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
    for attr in list(node.attr.keys()):
        if attr == "T":
            node.attr[attr].type = dtype
        new_node.attr[attr].CopyFrom(node.attr[attr])

    print("rewrite fused_batch_norm done!")


def convert_graph_to_fp16(model_path,
                          save_path,
                          name,
                          as_text=False,
                          target_type='fp16',
                          input_name=None,
                          output_names=None,
                          keep_fp32_node_name=[]):
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp32':
        dtype = types_pb2.DT_FLOAT

    source_sess = load_graph(model_path)
    source_graph_def = source_sess.graph.as_graph_def()
    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(source_graph_def.versions)

    for node in source_graph_def.node:
        # fused batch norm node
        if node.op == "FusedBatchNorm":
            rewrite_batch_norm_node_v2(node,
                                       target_graph_def,
                                       target_type=target_type)
            continue

        # replicate node
        new_node = target_graph_def.node.add()
        new_node.op = node.op
        new_node.name = node.name
        new_node.input.extend(node.input)
        attrs = list(node.attr.keys())

        # keep batch norm params node
        if ("BatchNorm" in node.name) or ('batch_normalization' in node.name):
            for attr in attrs:
                new_node.attr[attr].CopyFrom(node.attr[attr])
            continue

        # replace dtype in node attr with target dtype
        for attr in attrs:
            # keep special node in fp32
            if node.name in keep_fp32_node_name:
                new_node.attr[attr].CopyFrom(node.attr[attr])
                continue

            if node.attr[attr].type == types_pb2.DT_FLOAT:
                # modify node dtype
                node.attr[attr].type = dtype

            if attr == "value":
                tensor = node.attr[attr].tensor
                if tensor.dtype == types_pb2.DT_FLOAT:
                    # if float_val exists
                    if tensor.float_val:
                        float_val = tf.make_ndarray(node.attr[attr].tensor)
                        new_node.attr[attr].tensor.CopyFrom(
                            tf.make_tensor_proto(float_val, dtype=dtype))
                        continue

                    # if tensor content exists
                    if tensor.tensor_content:
                        tensor_shape = [
                            x.size for x in tensor.tensor_shape.dim
                        ]
                        tensor_weights = tf.make_ndarray(tensor)
                        # reshape tensor
                        tensor_weights = np.reshape(tensor_weights,
                                                    tensor_shape)
                        tensor_proto = tf.make_tensor_proto(tensor_weights,
                                                            dtype=dtype)
                        new_node.attr[attr].tensor.CopyFrom(tensor_proto)
                        continue

            new_node.attr[attr].CopyFrom(node.attr[attr])

    # transform graph
    if output_names:
        if not input_name:
            input_name = []
        transforms = ["strip_unused_nodes"]
        target_graph_def = TransformGraph(target_graph_def, input_name,
                                          output_names, transforms)

    # write graph_def to model
    tf.io.write_graph(target_graph_def,
                      logdir=save_path,
                      name=name,
                      as_text=as_text)
    print("Converting done ...")


def convert_graph_to_fp32(model_path,
                          save_path,
                          name,
                          as_text=False,
                          target_type='fp32',
                          input_name=None,
                          output_names=None,
                          keep_fp16_node_name=[]):
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp32':
        dtype = types_pb2.DT_FLOAT

    source_sess = load_graph(model_path)
    source_graph_def = source_sess.graph.as_graph_def()
    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(source_graph_def.versions)

    for node in source_graph_def.node:
        # fused batch norm node
        if node.op == "FusedBatchNorm":
            rewrite_batch_norm_node_v2(node,
                                       target_graph_def,
                                       target_type=target_type)
            continue

        # replicate node
        new_node = target_graph_def.node.add()
        new_node.op = node.op
        new_node.name = node.name
        new_node.input.extend(node.input)
        attrs = list(node.attr.keys())

        # keep batch norm params node
        if ("BatchNorm" in node.name) or ('batch_normalization' in node.name):
            for attr in attrs:
                new_node.attr[attr].CopyFrom(node.attr[attr])
            continue

        # replace dtype in node attr with target dtype
        for attr in attrs:
            # keep special node in fp16
            if node.name in keep_fp16_node_name:
                new_node.attr[attr].CopyFrom(node.attr[attr])
                continue

            if node.attr[attr].type == types_pb2.DT_HALF:
                # modify node dtype
                node.attr[attr].type = dtype

            if attr == "value":
                tensor = node.attr[attr].tensor
                if tensor.dtype == types_pb2.DT_HALF:
                    # if half_val exists
                    if tensor.half_val:
                        half_val = tf.make_ndarray(node.attr[attr].tensor)
                        new_node.attr[attr].tensor.CopyFrom(
                            tf.make_tensor_proto(half_val, dtype=dtype))
                        continue

                    # if tensor content exists
                    if tensor.tensor_content:
                        tensor_shape = [
                            x.size for x in tensor.tensor_shape.dim
                        ]
                        tensor_weights = tf.make_ndarray(tensor)
                        # reshape tensor
                        tensor_weights = np.reshape(tensor_weights,
                                                    tensor_shape)
                        tensor_proto = tf.make_tensor_proto(tensor_weights,
                                                            dtype=dtype)
                        new_node.attr[attr].tensor.CopyFrom(tensor_proto)
                        continue

            new_node.attr[attr].CopyFrom(node.attr[attr])

    # transform graph
    if output_names:
        if not input_name:
            input_name = []
        transforms = ["strip_unused_nodes"]
        target_graph_def = TransformGraph(target_graph_def, input_name,
                                          output_names, transforms)

    # write graph_def to model
    tf.io.write_graph(target_graph_def,
                      logdir=save_path,
                      name=name,
                      as_text=as_text)
    print("Converting done ...")
