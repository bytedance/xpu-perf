# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Dict

import onnx
import poprt

from poprt.passes.apply_ir_pass import ApplyIrPass
from poprt.passes.base_pass import ImmutablePass
from poprt.passes.onnx_helper import clean_info
from poprt.passes.remove_duplicated_initializer import RemoveDuplicatedInitializer

# skip register here
# @register('final_check')
class CustomFinalCheck(ImmutablePass):
    """Final check for data type and shape of the converted model."""

    name = 'final_check'

    def run_transform(
        self, graph: onnx.GraphProto, is_main_graph: bool
    ) -> onnx.GraphProto:
        # Check if all tensors have valid dtype and shape
        output_tensors = []
        for n in graph.node:
            output_tensors.extend(n.output)
        output_tensors = set(output_tensors)

        for t in list(graph.value_info) + list(graph.output):
            tensor_type = t.type.tensor_type
            has_dtype = tensor_type.HasField("elem_type")
            has_shape = (
                tensor_type.HasField("shape")
                and len(tensor_type.shape.ListFields()) > 0
            )
            if has_dtype and has_shape:
                dtype = tensor_type.elem_type
                # If the dtype < 1 (onnx.TensorProto.FLOAT) or dtype > 16 (onnx.TensorProto.BFLOAT16),
                # the dtype is invalid.
                is_valid_dtype = (
                    dtype >= onnx.TensorProto.FLOAT
                    and dtype <= onnx.TensorProto.BFLOAT16
                )
                shape = [dim.dim_value for dim in tensor_type.shape.dim]
                is_valid_shape = 0 not in shape
                if (not is_valid_dtype) or (not is_valid_shape):
                    self.logger.warning(
                        f"{t.name} has no inferred elem_type {dtype} or shape {shape}"
                    )
                if t.name in output_tensors:
                    output_tensors.remove(t.name)

        for t_name in output_tensors:
            self.logger.warning(
                f"Graph {graph.name} tensor {t_name} has no elem_type or shape."
            )
        return graph

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # NOTE: skip SortGraph here
        # Ensure topological for subgraph
        # model = SortGraph().run(model)
        # Infer shape and dtype to make sure all passes process validly.
        model = clean_info(model)
        # Remove duplicated initializer
        model = RemoveDuplicatedInitializer().run(model)
        model.graph.CopyFrom(self.traverse_graph(model.graph, self.run_transform))
        # Ensure each node has a unique name
        model = ApplyIrPass(["unique_name_for_nodes"])(model)
        return model

def custom_get_all_named_subclasses(cls: Any) -> Dict[str, Any]:
    subclasses = {}

    def visit(cls):
        for subclass in cls.__subclasses__():
            if hasattr(subclass, 'name'):
                subclasses[subclass.name] = subclass
            visit(subclass)

    visit(cls)

    # patch
    subclasses['final_check'] = CustomFinalCheck

    return subclasses

# monkey patch
poprt.passes.base_pass.get_all_named_subclasses = custom_get_all_named_subclasses
