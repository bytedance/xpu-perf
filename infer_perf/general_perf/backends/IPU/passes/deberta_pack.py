# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import onnx
from poprt import Pass
from poprt.passes import register
from poprt.passes.onnx_helper import clean_info, get_dtype, topological_sort
from poprt.passes.pattern_helper import PatternMatcher
from poprt.passes.shape_inference import infer_shapes


@register("deberta_pack")
class PackedDeberta(Pass):
    @staticmethod
    def _find(items, search_func, return_all=False):
        results = []
        for i, item in enumerate(items):
            if search_func(item):
                results.append((i, item))
                if not return_all:
                    break
        return results if return_all else (-1, None) if not results else results[0]

    def __init__(self):
        super().__init__()

    def _modify_mask_before_mul_to_input(self, model):
        pattern = ["s:0->Unsqueeze:Unsqueeze->Cast:Cast->Mul:Mul->e:5"]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        if ops:
            Cast = onnx.helper.make_node(
                "Cast",
                name="{}_Cast".format(ops["Unsqueeze"].node.name),
                inputs=[ops["Unsqueeze"].node.input[0]],
                outputs=["{}_Cast:0".format(ops["Unsqueeze"].node.name)],
                to=onnx.TensorProto.BOOL,
            )
            ops["Unsqueeze"].node.input[0] = Cast.output[0]
            model.graph.node.insert(ops["Unsqueeze"].index, Cast)
        return model

    def _modify_attentionmask(self, model):
        pattern = [
            "s:0->Reshape:Reshape->Squeeze:Squeeze->Unsqueeze:Unsqueeze->Mul:Mul->Cast:Cast->Not:Not->e:1",
            "     Reshape:Reshape                                      ->Mul:Mul",
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        if ops:
            input = ops["Reshape"].node.input[0]
            for node in [
                ops[key].node
                for key in ["Reshape", "Squeeze", "Unsqueeze", "Mul", "Cast", "Not"]
            ]:
                model.graph.node.remove(node)
        else:
            return model

        pattern = [
            "s:0         ->WhereV2:WhereV2_1->Softmax:Softmax->WhereV2:WhereV2_2->MatMul:MatMul->e:1",
            "s:2->Add:Add->WhereV2:WhereV2_1",
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)
        AttentionMask, AttentionMaskNot = None, None
        while ops:
            if AttentionMask is None:
                dtype = get_dtype(model.graph, ops["Add"].node.output[0])
                kwargs = {
                    "dataType": "FLOAT"
                    if dtype == onnx.TensorProto.FLOAT
                    else "FLOAT16"
                }
                AttentionMask = onnx.helper.make_node(
                    "AttentionMask",
                    name="AttentionMask",
                    inputs=[input, ops["Add"].node.output[0]],
                    outputs=["{}_AttentionMask".format(ops["Add"].node.output[0])],
                    domain="ai.graphcore",
                    **kwargs,
                )
                Cast = onnx.helper.make_node(
                    "Cast",
                    name="{}_Cast".format(AttentionMask.name),
                    inputs=[AttentionMask.output[0]],
                    outputs=["{}_Cast:0".format(AttentionMask.name)],
                    to=onnx.TensorProto.BOOL,
                )
                Not = onnx.helper.make_node(
                    "Not",
                    name="{}_Not".format(Cast.name),
                    inputs=[Cast.output[0]],
                    outputs=["{}_Not:0".format(Cast.name)],
                )
                AttentionMaskNot = onnx.helper.make_node(
                    "Cast",
                    name="{}_Cast".format(Not.name),
                    inputs=[Not.output[0]],
                    outputs=["{}_Cast:0".format(Not.name)],
                    to=onnx.TensorProto.FLOAT16,
                )
                model.graph.node.insert(ops["Softmax"].index, AttentionMaskNot)
                model.graph.node.insert(ops["Softmax"].index, Not)
                model.graph.node.insert(ops["Softmax"].index, Cast)
                model.graph.node.insert(ops["Softmax"].index, AttentionMask)
            Add = onnx.helper.make_node(
                "Add",
                name="{}_Add".format(ops["Add"].node.output[0]),
                inputs=[AttentionMask.output[0], ops["Add"].node.output[0]],
                outputs=["{}_Add:0".format(ops["Add"].node.output[0])],
            )
            ops["Softmax"].node.input[0] = Add.output[0]
            Mul = onnx.helper.make_node(
                "Mul",
                name="{}_Mul".format(ops["Softmax"].node.output[0]),
                inputs=[AttentionMaskNot.output[0], ops["Softmax"].node.output[0]],
                outputs=["{}_Mul".format(ops["Softmax"].node.output[0])],
            )
            ops["MatMul"].node.input[0] = Mul.output[0]
            softmax_index, _ = self._find(
                model.graph.node, lambda n: n.name == ops["Softmax"].node.name
            )
            model.graph.node.insert(softmax_index + 1, Mul)
            model.graph.node.insert(softmax_index, Add)
            for key in ("WhereV2_1", "WhereV2_2"):
                model.graph.node.remove(ops[key].node)
            ops = pattern_matcher.next_pattern(model.graph)

        return model

    def _add_unpack(self, model):
        max_valid_num, segment_max_size, segment_num = 2 * 10, 384, 1
        pattern = [
            "s:0->Reshape:1->MatMul:2->Add:3->Split:4->e:5",
        ]
        pattern_matcher = PatternMatcher(pattern)
        ops = pattern_matcher.next_pattern(model.graph)

        if ops:
            unpack_info = onnx.helper.make_tensor_value_info(
                "unpack_info",
                onnx.TensorProto.INT32,
                (max_valid_num, segment_num),
            )
            model.graph.input.append(unpack_info)

            unpack_attributes = {
                "max_valid_num": max_valid_num,
                "segment_max_size": [segment_max_size],
            }
            Unpack = onnx.helper.make_node(
                "Unpack",
                name="Unpack",
                inputs=[ops["1"].node.output[0], unpack_info.name],
                outputs=["{}_Unpack:0".format(ops["1"].node.output[0])],
                domain="ai.graphcore",
                **unpack_attributes,
            )
            ops["2"].node.input[0] = Unpack.output[0]
            model.graph.node.insert(ops["2"].index, Unpack)
        return model

    def _add_pack(self, model):
        model = self._modify_mask_before_mul_to_input(model)
        model = self._modify_attentionmask(model)
        sorted_nodes = topological_sort(model.graph)
        model.graph.ClearField("node")
        for node in sorted_nodes:
            model.graph.node.append(node)
        return model

    def __call__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = self._add_pack(model)
        model = infer_shapes(clean_info(model))

        model = self._add_unpack(model)
        model = infer_shapes(clean_info(model))
        return model

    def run(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        onnx_model.CopyFrom(self.traverse_graph(onnx_model.graph, self.__call__))
        return onnx_model
