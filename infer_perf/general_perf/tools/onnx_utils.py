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

from typing import cast
import numpy as np
from numpy.lib.function_base import append
import onnx
import onnx.helper as helper
import onnxruntime as rt
from onnx import numpy_helper
from onnx.tools import update_model_dims
from onnx import shape_inference, TensorProto
import struct
import copy
import sys
'''
DType Info
'''
ONNX_DTYPE = {
    0: TensorProto.FLOAT,  # UNDEFINE, default as float32
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL,
    10: TensorProto.FLOAT16,
    11: TensorProto.DOUBLE,
    12: TensorProto.UINT32,
    13: TensorProto.UINT64,
}
'''
Nodes
'''


def get_node_by_name(graph, name):
    for node in graph.node:
        if node.name == name:
            return node
    return None


def get_nodes_by_optype(graph, typename):
    nodes = []
    for node in graph.node:
        if node.op_type == typename:
            nodes.append(node)
    return nodes


def get_node_by_output_name(graph, name):
    for node in graph.node:
        if node.output[0] == name:
            return node
    return None


def get_node_successor(graph, target_node):
    successor = []
    for node in graph.node:
        if len(list(set(node.input).intersection(set(
                target_node.output)))) > 0:
            successor.append(node)
    return successor


def get_value_info_by_name(graph, name):
    for val_info in graph.value_info:
        if val_info.name == name:
            return val_info
    return None


def get_shape_from_value_info(val_info):
    shape = [d.dim_value for d in val_info.type.tensor_type.shape.dim]
    return shape


def remove_weights(graph, name_list):
    rm_list = []
    for weight in graph.initializer:
        if weight.name in name_list:
            rm_list.append(weight)
    for weight in rm_list:
        graph.initializer.remove(weight)


def remove_inputs(graph, name_list):
    rm_list = []
    for input_t in graph.input:
        if input_t.name in name_list:
            rm_list.append(input_t)
    for input_t in rm_list:
        graph.input.remove(input_t)


def remove_value_infos(graph, name_list):
    rm_list = []
    for value_info in graph.value_info:
        if value_info.name in name_list:
            rm_list.append(value_info)
    for value_info in rm_list:
        graph.value_info.remove(value_info)


def remove_node_by_name(graph, name):
    target_node = get_node_by_name(graph, name)
    remove_node(graph, target_node)


def remove_node(graph, target_node):
    '''
        remove the node with only one input and only one output
    '''
    node_input = target_node.input[0]
    node_output = target_node.output[0]
    # set input of successor node to predecessor node of target node
    for node in graph.node:
        for i, n in enumerate(node.input):
            if n == node_output:
                node.input[i] = node_input

    target_names = set(target_node.input) & set(
        [weight.name for weight in graph.initializer])
    remove_weights(graph, target_names)
    target_names.add(node_output)
    remove_inputs(graph, target_names)
    remove_value_infos(graph, target_names)
    graph.node.remove(target_node)


'''
Constant & Initializer
'''


def is_initializer(graph, name):
    for tensor in graph.initializer:
        if tensor.name == name:
            return True
    return False


def get_initializer_by_name(graph, name):
    for tensor in graph.initializer:
        if tensor.name == name:
            return tensor
    return None


def get_init_value(tensor):
    return numpy_helper.to_array(tensor)


def set_init_value(graph, weight, data_numpy):
    # NOTE: weight can be stroed in human readable fields(float_data, int32_data, string_data, ...)
    # as well as raw_data, if we set weight by raw_data, we must clear the fields above to make it effective
    # NOTE: data_type between numpy and TensorProto

    raw_shape = tuple([i for i in weight.dims])
    new_shape = np.shape(data_numpy)

    if weight.data_type == 8:
        # string data type is special, it requires to store data in string_data field
        # NOT the raw_data field
        weight.string_data = bytes(data_numpy, encoding="utf8")
        weight.ClearField("raw_data")

        return

    if new_shape != raw_shape:
        print(
            "Warning: the new weight shape is not consistent with original shape!"
        )

        weight.dims[:] = list(new_shape)

        #  in cast is graph input?
        for model_input in graph.input:
            if model_input.name == weight.name:
                # copy from onnx.helper...
                tensor_shape_proto = model_input.type.tensor_type.shape
                tensor_shape_proto.ClearField("dim")
                tensor_shape_proto.dim.extend([])
                for d in new_shape:
                    dim = tensor_shape_proto.dim.add()
                    dim.dim_value = d

    weight.ClearField("float_data")
    weight.ClearField("int32_data")
    weight.ClearField("int64_data")
    weight.raw_data = data_numpy.tobytes()

    return


def is_constant(node):
    if node.op_type == "Constant":
        return True
    else:
        return False


def get_constant_value(node):
    for attr in node.attribute:
        if attr.name == 'value':
            if attr.t.data_type == 1:
                return np.array(struct.unpack('f', attr.t.raw_data))
            elif attr.t.data_type == 2:
                return np.array(struct.unpack('i', attr.t.raw_data))
            elif attr.t.data_type == 3:
                return np.array(struct.unpack('s', attr.t.raw_data))
            elif attr.t.data_type == 4:
                return np.array(struct.unpack('t', attr.t.raw_data))
            elif attr.t.data_type == 5:
                return np.array(struct.unpack('g', attr.t.raw_data))
            elif attr.t.data_type == 6:
                return np.frombuffer(attr.t.raw_data, dtype=np.float32)
            elif attr.t.data_type == 7:
                return np.frombuffer(attr.t.raw_data, dtype=np.int32)
            elif attr.t.data_type == 8:
                return np.frombuffer(attr.t.raw_data, dtype=np.string)
            elif attr.t.data_type == 9:
                return np.frombuffer(attr.t.raw_data, dtype=np.bool)
            elif attr.t.data_type == 10:
                return np.frombuffer(attr.t.raw_data, dtype=np.float16)
            elif attr.t.data_type == 11:
                return np.frombuffer(attr.t.raw_data, dtype=np.double)
            elif attr.t.data_type == 12:
                return np.frombuffer(attr.t.raw_data, dtype=np.uint32)
            elif attr.t.data_type == 13:
                return np.frombuffer(attr.t.raw_data, dtype=np.uint64)
            else:
                print("unsupported attribute data type with attribute name")


def set_constant_value(target_node, value):
    # NOTE : dtype value should match with target_node
    for attr in target_node.attribute:
        if (attr.name == "value"):
            attr.t.raw_data = value.tobytes()


'''
Attributes
'''


def get_attribute_by_name(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return attr


def set_node_attribute(target_node, attr_name, attr_value):
    flag = False
    for attr in target_node.attribute:
        if (attr.name == attr_name):
            if attr.type == 1:  # float value
                attr.f = attr_value
            elif attr.type == 2:  # int value
                attr.i = attr_value
            elif attr.type == 3:  # string value
                attr.s = attr_value
            elif attr.type == 4:  # tensor value
                attr.t = attr_value
            elif attr.type == 5:  # graph value
                attr.g = attr_value
            # NOTE: For repeated composite types, we should use something like
            # del attr.xxx[:]
            # attr.xxx.extend([n1, n2, n3])
            elif attr.type == 6:  # float[]
                attr.floats[:] = attr_value
            elif attr.type == 7:  # int[]
                attr.ints[:] = attr_value
            elif attr.type == 8:  # strings[]
                attr.strings[:] = attr_value
            else:
                print("unsupported attribute data type with attribute name")
                return False
            flag = True

    if not flag:
        # attribute not in original node
        print("Warning: you are appending a new attribute to the node!")
        target_node.attribute.append(
            helper.make_attribute(attr_name, attr_value))
        flag = True

    return flag


'''
Graph Input/Output
'''


def add_extra_output(graph, target_output, target_shape):

    extra_elem_type = 1
    for vi in graph.value_info:
        if vi.name == target_output:
            extra_elem_type = vi.type.tensor_type.elem_type

    extra_output = helper.make_tensor_value_info(target_output,
                                                 extra_elem_type, target_shape)
    '''
    # NOTE
    # if we know the value type and shape, we can alse use this
    def make_tensor_value_info(
        name,  # type: Text
        elem_type,  # type: int
        shape,  # type: Optional[Sequence[Union[Text, int]]]
        doc_string="",  # type: Text
        shape_denotation=None,  # type: Optional[List[Text]]
    ):
    '''

    graph.output.append(extra_output)
    return


def get_graph_input_by_name(graph, name):
    for input in graph.input:
        if input.name == name:
            return input
    return None


def get_graph_output_by_name(graph, name):
    for out in graph.output:
        if out.name == name:
            return out
    return None


def resort_nodes(model):
    new_model = copy.deepcopy(model)
    for n in new_model.graph.node:
        model.graph.node.remove(n)

    ready_tensors = [n.name for n in model.graph.input]
    ready_tensors.extend([n.name for n in model.graph.initializer])
    ready_tensors = set(ready_tensors)
    all_nodes = [n for n in new_model.graph.node]
    while True:
        activate_nodes = []
        for node in all_nodes:
            inputs = set(node.input)
            if len(inputs - ready_tensors) == 0:
                activate_nodes.append(node)

        assert len(activate_nodes) != 0, 'invalid graph'
        for node in activate_nodes:
            model.graph.node.append(node)
            ready_tensors = ready_tensors | set(node.output)
            all_nodes.remove(node)

        if len(all_nodes) == 0:
            break
    return model


'''
Pass
'''


def fix_model_shape(model,
                    in_dim_dict=None,
                    out_dim_dict=None,
                    fully_si=False):

    if in_dim_dict != None and out_dim_dict != None:
        update_model_dims.update_inputs_outputs_dims(model, in_dim_dict,
                                                     out_dim_dict)

    if fully_si:
        input_num = len(model.graph.input)
        tensors = model.graph.initializer
        for i, tensor in enumerate(tensors):
            value_info = helper.make_tensor_value_info(
                tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
            model.graph.input.insert(i + input_num, value_info)

    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)

    return model


def remove_redundant_cast(graph):
    cast_nodes = get_nodes_by_optype(graph, "Cast")
    for node in cast_nodes:
        in_node = get_node_by_output_name(graph, node.input[0])
        if in_node.op_type == "Cast":
            print("Removing redundant cast: ", in_node)
            node.input[0] = in_node.input[0]
            graph.node.remove(in_node)


def onxx_sess_opt(model, opt_model):
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = opt_model
    rt.InferenceSession(model,
                        sess_options,
                        providers=['CPUExecutionProvider'])


# ------------- Model speficted pass --------------------


def convert_fp16_to_fp32(model):
    # handle model.graph.initializer
    to_convert = []
    for init in model.graph.initializer:
        # print(init.name)

        if init.data_type != 10:
            continue
        to_convert.append(init)

    for init in to_convert:
        val = get_init_value(init)
        new_val = val.astype(np.float32)
        new_init = numpy_helper.from_array(new_val, init.name)
        model.graph.initializer.remove(init)
        model.graph.initializer.append(new_init)

    # handle mode.graph.node
    cons_ops = get_nodes_by_optype(model.graph, "Constant")
    for op in cons_ops:
        val_attr = get_attribute_by_name(op, "value")
        if val_attr.t.data_type != 10:
            continue

        # import pdb;pdb.set_trace()
        val = get_constant_value(op)
        new_val = val.astype(np.float32)
        set_constant_value(op, new_val)
        val_attr.t.data_type = 1

    for val_info in model.graph.value_info:
        if val_info.type.tensor_type.elem_type != 10:
            continue
        val_info.type.tensor_type.elem_type = 1

    # handle cast op
    cast_ops = get_nodes_by_optype(model.graph, "Cast")

    to_remove = []
    for cast in cast_ops:
        to = get_attribute_by_name(cast, "to")
        if to.i != 10 and to.i != 1:
            continue

        if to.i == 10:
            up_node = get_node_by_output_name(model.graph, cast.input[0])
            set_node_attribute(cast, "to", 1)

            if up_node.op_type != "Cast":
                continue

            up_to = get_attribute_by_name(up_node, "to")
            if up_to.i != 1:
                continue

        if to.i == 1:
            down_node = get_node_successor(model.graph, cast)
            if len(down_node) == 0:
                continue

            if down_node[0].op_type != "Cast":
                continue

            down_to = get_attribute_by_name(down_node[0], "to")
            if down_to.i != 10:
                continue

        # print(cast.name)
        succs = get_node_successor(model.graph, cast)
        for succ in succs:
            for idx, in_name in enumerate(succ.input):
                if in_name == cast.output[0]:
                    succ.input[idx] = cast.input[0]

        to_remove.append(cast)

    for cast in to_remove:
        out_info = get_graph_output_by_name(model.graph, cast.output[0])
        if out_info == None:
            model.graph.node.remove(cast)
        else:
            node = get_node_by_output_name(model.graph, cast.input[0])
            if node != None:
                for idx, out in enumerate(node.output):
                    if out == cast.input[0]:
                        node.output[idx] = cast.output[0]

            model.graph.node.remove(cast)

    return model


def replace_mask_where(model):
    # pattern: sub -> cast ----|
    #           |-----------> where
    where_ops = get_nodes_by_optype(model.graph, "Where")

    to_replace = []
    for where_node in where_ops:
        cond = where_node.input[0]
        node = get_node_by_output_name(model.graph, cond)
        if node.op_type != "Cast":
            continue

        y_in = where_node.input[2]
        node = get_node_by_output_name(model.graph, y_in)
        if node.op_type != "Sub":
            continue

        to_replace.append(where_node)

    to_remove = []
    for where in to_replace:
        x_in = where.input[1]
        y_in = where.input[2]
        mul_op = onnx.helper.make_node('Mul', [x_in, y_in],
                                       where.output,
                                       name="{}_mask_mul_replaced".format(
                                           where.name))
        model.graph.node.append(mul_op)

        cast_op = get_node_by_output_name(model.graph, where.input[0])
        to_remove.append(cast_op)
        to_remove.append(where)

    for node in to_remove:
        model.graph.node.remove(node)

    return model


def convert_expand_to_tile(model):
    expand_ops = get_nodes_by_optype(model.graph, "Expand")

    for expand_node in expand_ops:
        ifm = expand_node.input[0]
        ofm = expand_node.output[0]

        ifm_vi = get_value_info_by_name(model.graph, expand_node.input[0])
        if ifm_vi == None:
            continue

        init_shape = get_initializer_by_name(model.graph, expand_node.input[1])
        if init_shape == None:
            continue
        shape_val = get_init_value(init_shape)

        ofm_shape = shape_val.tolist()
        ifm_shape = [
            dim.dim_value for dim in ifm_vi.type.tensor_type.shape.dim
        ]

        repeats = [
            1 if i == j else int(j / i) for i, j in zip(ifm_shape, ofm_shape)
        ]

        repeats = np.array(repeats)
        repeats = numpy_helper.from_array(
            repeats, 'Tile_{}_repeats'.format(expand_node.name))
        tile_node = onnx.helper.make_node('Tile', [ifm, repeats.name], [ofm],
                                          name=expand_node.name)

        model.graph.node.append(tile_node)
        model.graph.initializer.append(repeats)
        model.graph.node.remove(expand_node)

    return model


def concat_to_tile(model):
    def is_tile_type(node):
        tile_flag = True
        for idx in range(len(node.input) - 1):
            if node.input[idx] == node.input[idx + 1]:
                continue
            else:
                tile_flag = False
                break
        return tile_flag

    concat_ops = get_nodes_by_optype(model.graph, "Concat")

    for concat in concat_ops:
        if not is_tile_type(concat):
            continue

        print("Converting concat to tile")

        in_val = get_value_info_by_name(model.graph, concat.input[0])
        out_val = get_value_info_by_name(model.graph, concat.output[0])
        ifm_shape = get_shape_from_value_info(in_val)
        ofm_shape = get_shape_from_value_info(out_val)

        repeats = [
            1 if i == j else int(j / i) for i, j in zip(ifm_shape, ofm_shape)
        ]

        repeats = np.array(repeats)
        repeats = numpy_helper.from_array(
            repeats, 'Tile_{}_repeats'.format(concat.name))
        tile_node = onnx.helper.make_node('Tile',
                                          [concat.input[0], repeats.name],
                                          [concat.output[0]],
                                          name=concat.name)

        model.graph.node.append(tile_node)
        model.graph.initializer.append(repeats)
        model.graph.node.remove(concat)


def remove_qdq(model):
    q_ops = get_nodes_by_optype(model.graph, "QuantizeLinear")

    for q_op in q_ops:
        dq = get_node_successor(model.graph, q_op)
        if len(dq) != 1 and dq[0].op_type != "DequantizeLinear":
            continue

        qdq_succ = get_node_successor(model.graph, dq[0])
        for i, n in enumerate(qdq_succ[0].input):
            if n == dq[0].output[0]:
                qdq_succ[0].input[i] = q_op.input[0]

        model.graph.node.remove(q_op)
        model.graph.node.remove(dq[0])


import torch
from onnx2torch import convert
import onnxruntime as ort

if __name__ == "__main__":
    # Path to ONNX model
    onnx_model_path = 'converted_models/no_qdq_2.onnx'
    onnx_model = onnx.load(onnx_model_path)
    in_shape_dict = {
        "data": [2, 10, 3, 256, 256],
    }
    out_shape_dict = {'logits': [2, 2], '1383': [1, 20]}
    onnx_model = fix_model_shape(onnx_model, in_shape_dict, out_shape_dict,
                                 True)
    onnx.save(onnx_model, 'converted_models/no_qdq_3.onnx')

    onxx_sess_opt('converted_models/no_qdq_3.onnx',
                  'converted_models/no_qdq_3.onnx')
    onnx_model = onnx.load('converted_models/no_qdq_3.onnx')

    torch_model_2 = convert(onnx_model)

    # You can pass the path to the onnx model to convert it or...
    # torch_model_1 = convert(onnx_model_path)

    # Create example data
    x = torch.ones((2, 10, 3, 256, 256))

    out_torch = torch_model_2(x)

    trace_model = torch.jit.trace(torch_model_2, x)

    ort_sess = ort.InferenceSession(onnx_model_path)
    outputs_ort = ort_sess.run(None, {'data': x.numpy()})

    print(outputs_ort[0] - out_torch[0].detach().numpy())
    print(outputs_ort[1] - out_torch[1].detach().numpy())

    # Check the Onnx output against PyTorch
    # print(torch.max(torch.abs(outputs_ort[0] - out_torch[0].detach().numpy())))
    # print(torch.max(torch.abs(outputs_ort[1] - out_torch[1].detach().numpy())))
    # print(np.allclose(outputs_ort[0], out_torch[0].detach().numpy(), atol=1.e-7))
    # print(np.allclose(outputs_ort[1], out_torch[1].detach().numpy(), atol=1.e-7))
