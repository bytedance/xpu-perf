import os
import json
from typing import Dict, List, Tuple, Union

from prettytable import PrettyTable
import onnx
from onnx import helper, TensorProto

from .op_templates import OP_ZOO



class OpTopologyDAG:
    def __init__(self, stream_allocation_strategy: str = "keep_main", json_file: str = None):
        """
        初始化拓扑管理器（支持从JSON文件恢复）
        :param stream_allocation_strategy: Stream ID 分配策略（仅在不加载JSON时生效）
            - "keep_main": 第一个分支继承 Stream 0，其余从 1 开始 (推荐)
            - "all_new": 所有分支从 1 开始连续分配
        :param json_file: 可选，从指定JSON文件恢复拓扑结构
        """
        # 初始化基础数据结构
        self.op_dict = {}              # 算子参数: op_name -> [params_list]
        self.op_set_func_dict = {}     # 算子参数: op_name --> set_func
        self.instance_dict = {}        # 节点映射: (instance_name, idx) -> (op_name, op_idx)
        self.node_prev = {}            # 前驱节点: node -> [prev_nodes]
        self.node_next = {}            # 后继节点: node -> [next_nodes]
        self.global_instance_index = 0 # 全局节点索引
        self.last_node = None          # 上一个节点（默认串行）
        self.node_creation_order = []  # 节点创建顺序
        
        # Stream ID 核心管理
        self.node_stream_id = {}       # 节点 -> Stream ID
        self.stream_strategy = stream_allocation_strategy

        # 如果指定了JSON文件，从文件恢复拓扑
        if json_file and os.path.exists(json_file):
            self._load_from_json(json_file)
            print(f"✅ 已从JSON文件 {json_file} 恢复拓扑结构")
        elif json_file:
            raise FileNotFoundError(f"JSON文件 {json_file} 不存在！")

    def _parse_node_id(self, node_id_str: str) -> Tuple[str, int]:
        """解析节点ID字符串（如 "Main:0"）为 (instance_name, index) 元组"""
        parts = node_id_str.split(":")
        return (parts[0], int(parts[1]))

    def _load_from_json(self, json_file: str):
        """从JSON文件加载拓扑结构（核心反向解析逻辑）"""
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 1. 恢复基础配置
        self.stream_strategy = json_data.get("stream_strategy", "keep_main")
        
        # 2. 恢复节点信息（按创建顺序）
        for node_info in json_data["nodes"]:
            # 解析节点ID
            node_id = self._parse_node_id(node_info["node_id"])
            instance_name = node_info["instance_name"]
            global_index = node_info["global_index"]
            op_name = node_info["op_name"]
            op_index = node_info["op_index"]
            stream_id = node_info["stream_id"]
            params = node_info["params"]

            # 更新数据结构
            self.instance_dict[node_id] = (op_name, op_index)
            self.node_stream_id[node_id] = stream_id
            self.node_creation_order.append(node_id)
            
            # 更新算子参数
            if op_name not in self.op_dict:
                self.op_dict[op_name] = []
            # 确保参数列表长度足够
            while len(self.op_dict[op_name]) <= op_index:
                self.op_dict[op_name].append({})
            self.op_dict[op_name][op_index] = params

            # 更新全局索引（取最大值）
            if global_index >= self.global_instance_index:
                self.global_instance_index = global_index + 1

        # 3. 恢复节点关系（前驱/后继）
        # 先构建节点ID映射（字符串 -> 元组）
        node_str_to_tuple = {
            self._parse_node_id(n["node_id"]): n["node_id"] 
            for n in json_data["nodes"]
        }
        
        # 恢复前驱节点
        for node_info in json_data["nodes"]:
            node_id = self._parse_node_id(node_info["node_id"])
            prev_nodes_str = node_info["prev_nodes"]
            prev_nodes = [self._parse_node_id(p) for p in prev_nodes_str if p]
            self.node_prev[node_id] = prev_nodes

        # 恢复后继节点
        for node_info in json_data["nodes"]:
            node_id = self._parse_node_id(node_info["node_id"])
            next_nodes_str = node_info["next_nodes"]
            next_nodes = [self._parse_node_id(n) for n in next_nodes_str if n]
            self.node_next[node_id] = next_nodes

        # 4. 设置最后一个节点（创建顺序的最后一个）
        if self.node_creation_order:
            self.last_node = self.node_creation_order[-1]

    def op_process_wrapper(
        self,
        op_name: str,
        instance_name: str,
        params: Dict = None,
        src: Union[Tuple[str, int], List[Tuple[str, int]]] = None,
        force_stream_id: int = None   # 手动强制指定
    ) -> Tuple[str, int]:
        """
        添加节点并自动计算 Stream ID（修复KeyError）
        """
        params = params or {}
        
        # 1. 创建当前节点
        current_node = (instance_name, self.global_instance_index)
        self.instance_dict[current_node] = (op_name, len(self.op_dict.get(op_name, [])))
        self.op_dict[op_name] = self.op_dict.get(op_name, []) + [params]
        self.op_set_func_dict[op_name] = OP_ZOO[op_name]
        self.global_instance_index += 1
        self.node_creation_order.append(current_node)

        # 2. 处理前驱节点
        prev_nodes = []
        if src is None:
            prev_nodes = [self.last_node] if self.last_node is not None else []
        else:
            prev_nodes = [src] if isinstance(src, tuple) else src
        # 过滤无效前驱（None/不存在的节点）
        prev_nodes = [p for p in prev_nodes if p and p in self.instance_dict]
        self.node_prev[current_node] = prev_nodes

        # 3. 先临时记录后继关系（避免循环依赖）
        temp_next = []
        for p in prev_nodes:
            if p not in self.node_next:
                self.node_next[p] = []
            # 先记录当前长度（关键修复：用添加前的长度作为分支索引）
            temp_next.append((p, len(self.node_next[p])))
            self.node_next[p].append(current_node)

        # ===================== 核心：修复后的Stream ID分配逻辑 =====================
        # 优先级1：手动强制指定
        if force_stream_id is not None:
            self.node_stream_id[current_node] = force_stream_id
        
        # 优先级2：汇聚节点（多前驱）→ 强制回归Stream 0
        elif len(prev_nodes) > 1:
            self.node_stream_id[current_node] = 0
        
        # 优先级3：单前驱场景（核心修复）
        elif len(prev_nodes) == 1:
            main_prev = prev_nodes[0]
            prev_stream = self.node_stream_id.get(main_prev, 0)
            
            # 获取添加当前节点前的后继列表长度（即当前分支的索引）
            branch_idx = temp_next[0][1]  # 关键修复：使用添加前的长度
            
            # 判断前驱是否是分叉点（后继数>1）
            if len(self.node_next[main_prev]) > 1:
                # 根据策略分配Stream ID
                if self.stream_strategy == "keep_main":
                    # 策略1：第一个分支(索引0)继承0，其余索引即ID
                    stream_id = 0 if branch_idx == 0 else branch_idx
                else:
                    # 策略2：所有分支从1开始
                    stream_id = branch_idx + 1
                self.node_stream_id[current_node] = stream_id
            else:
                # 非分叉点 → 继承前驱Stream ID
                self.node_stream_id[current_node] = prev_stream
        
        # 优先级4：无依赖节点（根节点）
        else:
            self.node_stream_id[current_node] = 0
        # ======================================================================

        self.last_node = current_node
        return current_node


    def set_bench_info(self, bench_info):
        for op_name, arguments_list in self.op_dict.items():
            set_func = self.op_set_func_dict[op_name]
            for arguments in arguments_list:
                set_func(arguments, bench_info)

    def parse_results(self, targets):
        result_dict = {}
        for (instance_name, instance_index), (op_name, op_index) in self.instance_dict.items():
            target_result = targets[op_name][op_index]

            avail_providers = list(target_result.keys())
            try:
                target_provider = avail_providers[0]
                target_latency = target_result[target_provider]["latency(us)"]
                
                for provider in avail_providers:
                    if target_result[provider]["latency(us)"] < target_latency:
                        target_provider = provider
                        target_latency = target_result[provider]["latency(us)"]
                result_dict[(instance_name, instance_index)] = {
                    target_provider: target_result[target_provider]
                }
            except Exception as e:
                print(f"⚠️  解析结果失败 for node {instance_name}:{instance_index} ({op_name}), error: {e}")
                target_provider = "Unknown"
                target_results = {
                    "latency(us)": 0.
                }
                
        return result_dict

    def _topological_sort(self):
        """拓扑排序"""
        in_degree = {n: len(p) for n, p in self.node_prev.items()}
        queue = [n for n, d in in_degree.items() if d == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for neighbor in self.node_next.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return sorted_nodes

    def calculate_timeline(self, node_latency: Dict[Tuple[str, int], Dict[str, Dict[str, float]]]):
        """
        核心逻辑：
        1. 自动识别每个节点的Provider并取耗时
        2. 新增：所有节点耗时总和（纯累加，不考虑多流并行）
        """
        # 1. 提取每个节点实际使用的Provider和耗时
        node_cost = {}      # 节点耗时
        node_provider = {}  # 记录每个节点用了什么Provider
        total_node_cost = 0.0  # 所有节点耗时总和（核心新增）
        
        for node in self.instance_dict.keys():
            if node in node_latency and node_latency[node]:
                # 取该节点字典里的第一个Provider（实际使用的）
                provider = next(iter(node_latency[node].keys()))
                cost = node_latency[node][provider].get("latency(us)", 0.0)
                node_cost[node] = cost
                node_provider[node] = provider
                total_node_cost += cost  # 累加所有节点耗时（不考虑并行）
            else:
                node_cost[node] = 0.0
                node_provider[node] = "Unknown"

        # 2. 拓扑排序计算开始/结束时间（考虑并行的总延迟）
        sorted_nodes = self._topological_sort()
        node_times = {}
        
        for node in sorted_nodes:
            prev_end_times = [node_times[p][1] for p in self.node_prev[node] if p in node_times]
            start_time = max(prev_end_times) if prev_end_times else 0.0
            end_time = start_time + node_cost[node]
            node_times[node] = (start_time, end_time)

        # 3. 计算关键路径（基于混合Provider的耗时）
        node_to_end = {}
        reversed_nodes = sorted_nodes[::-1]
        
        for node in reversed_nodes:
            next_costs = [node_to_end.get(next_node, 0) for next_node in self.node_next.get(node, [])]
            node_to_end[node] = node_cost[node] + (max(next_costs) if next_costs else 0)

        # 回溯关键路径
        critical_path = []
        start_nodes = [n for n in sorted_nodes if len(self.node_prev.get(n, [])) == 0]
        current_node = max(start_nodes, key=lambda x: node_to_end.get(x, 0)) if start_nodes else None

        while current_node:
            critical_path.append(current_node)
            next_nodes = self.node_next.get(current_node, [])
            if not next_nodes:
                break
            current_node = max(next_nodes, key=lambda x: node_to_end.get(x, 0))

        # 考虑并行的总延迟
        total_latency = max([node_times[n][1] for n in node_times.keys()]) if node_times else 0.0
        # 关键路径总耗时
        critical_total = sum([node_cost[n] for n in critical_path])

        return node_times, critical_path, total_latency, node_cost, node_provider, total_node_cost, critical_total

    def print_schedule(self, node_latency: Dict[Tuple[str, int], Dict[str, Dict[str, float]]]):
        """打印调度表（新增所有节点耗时总和展示）"""
        try:
            node_times, critical_path, total_latency, node_cost, node_provider, total_node_cost, critical_total = self.calculate_timeline(node_latency)
        except Exception as e:
            print(f"计算失败：{e}")
            return

        # 创建表格
        table = PrettyTable()
        table.field_names = [
            "节点ID", "Stream ID", "算子名称", 
            "实际Provider", "耗时(us)", "开始时间(us)", "结束时间(us)", "状态"
        ]
        table.align = "l"
        table.float_format = ".2f"

        # 填充数据
        for node in self.node_creation_order:
            if node not in node_times:
                continue
            op_name, _ = self.instance_dict[node]
            stream_id = self.node_stream_id.get(node, 0)
            provider = node_provider.get(node, "Unknown")
            cost = node_cost[node]
            start, end = node_times[node]
            status = "🔴关键路径" if node in critical_path else "⚪️普通节点"

            table.add_row([
                f"{node[0]}:{node[1]}",
                stream_id,
                op_name,
                provider,
                cost,
                start,
                end,
                status
            ])

        # 打印结果（新增所有节点耗时总和）
        print("\n" + "=" * 150)
        print(f"📌 耗时统计（核心指标对比）")
        print(f"   📊 所有节点耗时总和（纯累加）: {total_node_cost:.2f} us")
        print(f"   🚀 关键路径总耗时: {critical_total:.2f} us")
        print(f"   ⏱️  实际总延迟（考虑多流并行）: {total_latency:.2f} us")
        print("=" * 150)
        print(table)
        print("=" * 150)
        cp_str = " → ".join([f"{n[0]}:{n[1]}({node_provider.get(n, '?')})" for n in critical_path])
        print(f"🔍 关键路径: {cp_str}")
        print(f"💡 并行加速比: {total_node_cost/total_latency:.2f}x (总计算量/实际延迟)")

    def print_topo_pretty(self):
        """可视化拓扑结构"""
        if not self.instance_dict:
            print("\n⚠️  当前拓扑为空！")
            return

        table = PrettyTable()
        table.field_names = ["序号", "Stream ID", "节点ID", "算子", "前驱 (流)", "后继 (流)"]
        table.align = "l"
        table.valign = "t"

        for idx, node in enumerate(self.node_creation_order, 1):
            op_name, _ = self.instance_dict[node]
            stream_id = self.node_stream_id.get(node, 0)
            
            # 格式化前驱
            prev_info = []
            for p in self.node_prev.get(node, []):
                p_stream = self.node_stream_id.get(p, 0)
                prev_info.append(f"{p[0]}:{p[1]} (S{p_stream})")
            prev_str = "\n".join(prev_info) if prev_info else "无"
            
            # 格式化后继
            next_info = []
            for n in self.node_next.get(node, []):
                n_stream = self.node_stream_id.get(n, 0)
                next_info.append(f"{n[0]}:{n[1]} (S{n_stream})")
            next_str = "\n".join(next_info) if next_info else "无"

            # 特殊标记
            node_display = f"{node[0]}:{node[1]}"
            if len(self.node_next.get(node, [])) > 1:
                node_display += " ⭐(分叉点)"
            if len(self.node_prev.get(node, [])) > 1:
                node_display += " 🔄(汇聚回S0)"

            table.add_row([idx, stream_id, node_display, op_name, prev_str, next_str])

        print(f"\n" + "=" * 110)
        print(f"📊 算子拓扑结构表 (策略: {self.stream_strategy})")
        print("=" * 110)
        print(table)

        # 统计
        streams_used = set(self.node_stream_id.values())
        print(f"\n📈 统计：总流数 {len(streams_used)} (使用了 Stream {sorted(streams_used)})")

    def export_json(self, filename="topology.json", indent=4):
        """
        导出拓扑结构为JSON文件（包含Stream ID信息）
        :param filename: 输出文件名
        :param indent: JSON缩进
        """
        if not self.instance_dict:
            print("\n⚠️  拓扑为空，无法导出JSON！")
            return
        
        # 构建JSON数据结构
        json_data = {
            "stream_strategy": self.stream_strategy,
            "total_nodes": len(self.instance_dict),
            "streams_used": sorted(list(set(self.node_stream_id.values()))),
            "nodes": [],
            "edges": []
        }

        # 节点信息
        for node in self.node_creation_order:
            op_name, op_idx = self.instance_dict[node]
            node_info = {
                "node_id": f"{node[0]}:{node[1]}",
                "instance_name": node[0],
                "global_index": node[1],
                "op_name": op_name,
                "op_index": op_idx,
                "stream_id": self.node_stream_id.get(node, 0),
                "is_split_node": len(self.node_next.get(node, [])) > 1,
                "is_merge_node": len(self.node_prev.get(node, [])) > 1,
                "params": self.op_dict[op_name][op_idx],
                "prev_nodes": [f"{p[0]}:{p[1]}" for p in self.node_prev.get(node, [])],
                "next_nodes": [f"{n[0]}:{n[1]}" for n in self.node_next.get(node, [])]
            }
            json_data["nodes"].append(node_info)

        # 边信息
        for node in self.instance_dict:
            for next_node in self.node_next.get(node, []):
                edge_info = {
                    "from_node": f"{node[0]}:{node[1]}",
                    "to_node": f"{next_node[0]}:{next_node[1]}",
                    "from_stream": self.node_stream_id.get(node, 0),
                    "to_stream": self.node_stream_id.get(next_node, 0)
                }
                json_data["edges"].append(edge_info)

        # 写入文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=indent)
        
        print(f"\n✅ JSON拓扑文件已导出至: {filename}")
        return json_data

    def export_dot(self, filename="topology.dot", with_stream_color=True):
        """
        导出拓扑为DOT文件（可用于Graphviz可视化）
        :param filename: 输出文件名
        :param with_stream_color: 是否按Stream ID着色
        """
        if not self.instance_dict:
            print("\n⚠️  拓扑为空，无法导出DOT！")
            return
        
        # Stream ID配色方案
        stream_colors = {
            0: "lightblue",
            1: "lightgreen",
            2: "lightyellow",
            3: "lightpink",
            4: "lightcyan",
            5: "lavender"
        }

        # 构建DOT内容
        dot_content = [
            "digraph OpTopology {",
            "  rankdir=TB;",
            "  node [shape=box, style=filled, fontname=Arial];",
            "  edge [fontname=Arial];",
            ""
        ]

        # 添加节点
        for node in self.node_creation_order:
            op_name, _ = self.instance_dict[node]
            stream_id = self.node_stream_id.get(node, 0)
            node_id = f"node_{node[0]}_{node[1]}"
            node_label = f"{node[0]}:{node[1]}\\n{op_name}\\nStream {stream_id}"
            
            # 添加特殊标记
            if len(self.node_next.get(node, [])) > 1:
                node_label += "\\n⭐分叉点"
            if len(self.node_prev.get(node, [])) > 1:
                node_label += "\\n🔄汇聚点"
            
            # 设置颜色
            color = stream_colors.get(stream_id, "white")
            dot_content.append(f'  {node_id} [label="{node_label}", fillcolor="{color}"];')

        # 添加边
        dot_content.append("")
        for node in self.instance_dict:
            from_node_id = f"node_{node[0]}_{node[1]}"
            for next_node in self.node_next.get(node, []):
                to_node_id = f"node_{next_node[0]}_{next_node[1]}"
                dot_content.append(f"  {from_node_id} -> {to_node_id};")

        dot_content.append("}")

        # 写入文件
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(dot_content))
        
        print(f"\n✅ DOT拓扑文件已导出至: {filename}")
        print("   📝 提示：可使用Graphviz或在线工具（如https://edotor.net/）可视化此文件")

    def export_onnx(self, filename="model.onnx"):
        """导出 ONNX 模型"""
        nodes = []
        inputs = []
        outputs = []
        value_info = []
        
        # 简单的占位逻辑，实际使用需根据算子类型定义
        for node in self._topological_sort():
            op_name, _ = self.instance_dict[node]
            # 假设输入输出名称
            input_names = [f"{p[0]}_{p[1]}" for p in self.node_prev.get(node, [])]
            if not input_names:
                input_names = [f"input_{node[0]}_{node[1]}"]
                inputs.append(helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, [1, 3, 224, 224]))
            
            output_name = f"{node[0]}_{node[1]}"
            nodes.append(helper.make_node(op_name, input_names, [output_name], name=output_name))
            value_info.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, 224, 224]))

        # 标记输出节点
        for node in self.instance_dict:
            if not self.node_next.get(node):
                output_name = f"{node[0]}_{node[1]}"
                outputs.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, 224, 224]))

        graph = helper.make_graph(nodes, "MultiStreamGraph", inputs, outputs, value_info)
        model = helper.make_model(graph)
        onnx.save(model, filename)
        print(f"\n✅ ONNX 模型已导出至: {filename}")





def add_moe_graph(
    model_topo: OpTopologyDAG, 
    pre_node: Tuple[str, int], 
    
    hidden_size: int = 4096, 

    num_experts: int = 128, 
    moe_topk: int = 8, 
    moe_intermediate_size: int = 1536, 
    moe_mlp_config: Dict[str, str] = {},

    num_share_experts: int = 0, 
    share_intermediate_size: int = 0, 
    share_mlp_config: Dict[str, str] = {},

    is_pre_softmax: bool = True, 

    sp_size: int = 1, 
    ep_size: int = 1, 
    
    use_deepep: bool = False, 
    fuse_combine: bool = False,

    default_dtype: str = "bfloat16", 
    module_name: str = "moe"
):
    # 使用 deepep 的方式来 dispatch tokens
    if ep_size > 1 and use_deepep:
        raise NotImplementedError("DeepEP 模式暂未支持")

    else: 

        merged_nodes = []


        """
        stream0: 一个大的 all_gather
        """
        stream0 = model_topo.op_process_wrapper(
            "all_gather", f"{module_name}_ag0", 
            {
                "dtype": default_dtype, 
                "world_size": ep_size, 
                "hidden_size": hidden_size
            }, 
            src=pre_node
        )
        merged_nodes.append(stream0)

        """
        stream1: gating + softmax + topk
        """
        model_topo.op_process_wrapper(
            "moe_gating_gemm", f"{module_name}_gating", 
            {
                "dtype": "float32",
                "compute_dtype": "float32", 
                "dst_dtype": "float32", 
                "num_experts": num_experts, 
                "hidden_size": hidden_size, 
                "sp_size": sp_size
            }, 
            src=pre_node
        )

        model_topo.op_process_wrapper(
            "moe_softmax_topk", f"{module_name}_softmax_topk", 
            {
                "dtype": "float32", 
                "num_experts": num_experts, 
                "topk": moe_topk, 
                "compute_mode": "pre-softmax" if is_pre_softmax else "post-softmax",
                "sp_size": sp_size
            }
        )

        model_topo.op_process_wrapper(
            "all_gather", f"{module_name}_ag1", 
            {
                "dtype": "float32", 
                "world_size": sp_size, 
                "hidden_size": moe_topk
            }
        )

        stream1 = model_topo.op_process_wrapper(
            "all_gather", f"{module_name}_ag1", 
            {
                "dtype": "int32", 
                "world_size": sp_size, 
                "hidden_size": moe_topk
            }
        )
        merged_nodes.append(stream1)


        """
        stream2: share experts
        """
        if num_share_experts > 0:
            share_dtype = share_mlp_config.get("dtype", "int8")
            share_w_dtype = share_mlp_config.get("w_dtype", "int8")
            share_compute_dtype = share_mlp_config.get("compute_dtype", "int8")
            
            model_topo.op_process_wrapper(
                "scale_dynamic_quant", f"{module_name}_share_quant", 
                {
                    "dtype": default_dtype, 
                    "dst_dtype": share_dtype, 
                    "sp_size": sp_size,
                    "hidden_size": hidden_size, 
                }, 
                src=pre_node
            )

            model_topo.op_process_wrapper(
                "quant_matmul", f"{module_name}_share_up_gemm", 
                {
                    "dtype": share_dtype, 
                    "w_dtype": share_w_dtype, 
                    "compute_dtype": share_compute_dtype, 
                    "dst_dtype": default_dtype, 
                    "sp_size": sp_size,
                    "hidden_size": hidden_size, 
                    "new_hidden_size": share_intermediate_size * num_share_experts * 2,
                }
            )

            model_topo.op_process_wrapper(
                "swiglu_dynamic_quant", f"{module_name}_share_swiglu", 
                {
                    "dtype": default_dtype, 
                    "dst_dtype": share_dtype, 

                    "hidden_size": share_intermediate_size * num_share_experts, 
                    "sp_size": sp_size,
                }
            )

            stream2 = model_topo.op_process_wrapper(
                "quant_matmul", f"{module_name}_share_down_gemm", 
                {
                    "dtype": share_dtype, 
                    "w_dtype": share_w_dtype, 
                    "compute_dtype": share_compute_dtype, 
                    "dst_dtype": default_dtype, 
                    "sp_size": sp_size,
                    "hidden_size": share_intermediate_size * num_share_experts, 
                    "new_hidden_size": hidden_size,
                }
            )
            merged_nodes.append(stream2)


        moe_dtype = moe_mlp_config.get("dtype", "int8")
        moe_w_dtype = moe_mlp_config.get("w_dtype", "int8")
        moe_compute_dtype = moe_mlp_config.get("compute_dtype", "int8")

        model_topo.op_process_wrapper(
            "moe_scatter_dynamic_quant", f"{module_name}_scatter", 
            {
                "dtype": default_dtype, 
                "dst_dtype": moe_dtype,

                "ep_size": ep_size, 
                "num_experts": num_experts,
                "topk": moe_topk,

                "hidden_size": hidden_size,
            }, 
            src=merged_nodes
        )

        model_topo.op_process_wrapper(
            "moe_quant_group_gemm", f"{module_name}_moe_up_gemm", 
            {
                "dtype": moe_dtype, 
                "w_dtype": moe_w_dtype, 
                "compute_dtype": moe_compute_dtype, 
                "dst_dtype": default_dtype, 

                "ep_size": ep_size, 
                "num_experts": num_experts,
                "topk": moe_topk,

                "hidden_size": hidden_size,
                "new_hidden_size": moe_intermediate_size * 2
            }
        )

        model_topo.op_process_wrapper(
            "moe_swiglu_dynamic_quant", f"{module_name}_moe_swiglu", 
            {
                "dtype": default_dtype, 
                "dst_dtype": moe_dtype, 

                "ep_size": ep_size, 
                "num_experts": num_experts, 
                "topk": moe_topk,
                "hidden_size": moe_intermediate_size,
            }
        )


        output_node = None

        if fuse_combine:

            common_config = {
                "dtype": moe_dtype,     
                "w_dtype": moe_w_dtype, 
                "compute_dtype": moe_compute_dtype, 
                "dst_dtype": default_dtype, 

                "ep_size": ep_size, 
                "num_experts": num_experts,
                "topk": moe_topk,

                "hidden_size": moe_intermediate_size,
                "new_hidden_size": hidden_size
            }
            if num_share_experts > 0:
                common_config.update(
                    {
                        "sp_size": sp_size
                    }
                )

            output_node = model_topo.op_process_wrapper(
                "moe_quant_group_gemm_combine", f"{module_name}_moe_down_gemm", 
                common_config
            )

        else:
            model_topo.op_process_wrapper(
                "moe_quant_group_gemm", f"{module_name}_moe_down_gemm", 
                {
                    "dtype": moe_dtype, 
                    "w_dtype": moe_w_dtype, 
                    "compute_dtype": moe_compute_dtype, 
                    "dst_dtype": default_dtype, 

                    "ep_size": ep_size, 
                    "num_experts": num_experts,
                    "topk": moe_topk,

                    "hidden_size": moe_intermediate_size,
                    "new_hidden_size": hidden_size
                }
            )

            common_config = {
                "dtype": default_dtype, 

                "ep_size": ep_size, 
                "num_experts": num_experts,
                "topk": moe_topk,

                "hidden_size": hidden_size, 
            }
            if num_share_experts > 0:
                common_config.update(
                    {
                        "sp_size": sp_size
                    }
                )

            model_topo.op_process_wrapper(
                "moe_gather", f"{module_name}_moe_gather", 
                common_config
            )

        return output_node



