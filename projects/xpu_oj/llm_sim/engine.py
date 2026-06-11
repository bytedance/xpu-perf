import csv
import json
import copy
import shutil
import pathlib
import datetime
import importlib
import prettytable
from dataclasses import asdict
from collections import defaultdict
from typing import Dict, List, Union, Optional, Any


from xpu_perf.model_perf.utils import BenchTestCase, BenchClient, DistributionInfo
from xpu_perf.model_perf.utils import get_unique_id, print_server_info

FILE_DIR = pathlib.Path(__file__).parent.absolute()


DEFAULT_WORKSPACE_DIR = FILE_DIR.joinpath("workspaces")
DEFAULT_REPORT_DIR = FILE_DIR.joinpath("reports")
DEFAULT_ALL_REPORT_DIR = FILE_DIR.joinpath("all_reports")



class XpuPerfSimEngine:
    def __init__(
        self, 
        ip, port, run_mode, 
        model_config_dict, 
        **kwargs
    ):
        # 服务信息
        self.disable_print_result = bool(kwargs.pop("disable_print_result", False))
        bench_stream = bool(kwargs.pop("bench_stream_progress", True))
        self.client = BenchClient(ip=ip, port=port, stream=bench_stream)

        # detect server info
        self.detect_server_info()

        self.run_mode = run_mode

        # bench_config.json
        self.bench_config = model_config_dict

        # seed_oss / seed-oss-36b
        self.base_model_name = self.bench_config["base_model_name"]
        self.model_name = self.bench_config["model_name"]



        model_zoo_module = importlib.import_module("model_zoo")


        template_module_path = self.bench_config["template"]
        template_module = importlib.import_module(template_module_path)

        
        self.model_config_values = model_zoo_module.BASE_MODEL_MAPPING[self.base_model_name][self.model_name]
        self.model_config = self.model_config_values["common"]
        self.src_model_config = self.model_config_values["source"]

        generate_func = template_module.generate
        self.model_topo = generate_func(
            self.src_model_config, 
            self.bench_config,
        )
        
        # 解析模型
        self.parse_model(**kwargs)

        self.model_topo.print_topo_pretty()


        """
        固定输入, 可以绘制曲线, 
        用 Dict 保存, key 是 (batch_size, cache_len, q_len)
        {
            (batch_size, cache_len, q_len): {
                "test_case": test_case, 
                "perf_info": {
                    "total_latency": 端到端耗时
                    "total_node_latency": 节点总耗时
                    "breakdown": {
                        (instance_name, instance_index): {
                            provider_name: {arguments}
                        }
                    }                
                }
            }
        }
        """
        self.fix_data_dict: Dict = {}


        """
        可变输入, 可用于统计平均性能
        用 List 保存
        [
            {
                "test_case": test_case, 
                "perf_info": {
                    "total_latency": 端到端耗时
                    "total_node_latency": 节点总耗时
                    "breakdown": {
                        (instance_name, instance_index): {
                            provider_name: {arguments}
                        }
                    }                
                }
            }
        ]
        """
        self.var_data_dict: List[Dict] = []

        """
        保存所有 test_case 的结果, 按输入顺序排列, 包含 index
        [
            {
                "index": case_idx,
                "test_case": test_case,
                "perf_info": { ... }
            }
        ]
        """
        self.all_data_list: List[Dict] = []

        self.create_engine()


    def detect_server_info(self):
        self.info_dict = self.client.get_info()
        print_server_info(self.info_dict)

        self.backend = self.info_dict["backend_type"]
        self.device_name = self.info_dict["backend"]["device_name"]
        self.path_device_name = self.device_name.replace(" ", "_")
        self.actual_device_num = self.info_dict["backend"]["device_count"]


    def create_engine(self, **kwargs):
        self.prepare_workspace(**kwargs)


    def prepare_workspace(self, **kwargs):
        self.cur_datetime = get_unique_id()
        self.pwd_dir = pathlib.Path.cwd()

        # 放置当前运行的所有数据
        workspace_path = kwargs.get("workspace_path", str(DEFAULT_WORKSPACE_DIR))
        self.workspace_dir = pathlib.Path(workspace_path).absolute()

        # 放置每次生成的workloads的数据
        self.workload_dir = self.workspace_dir.joinpath("workloads")
        if self.workload_dir.exists():
            shutil.rmtree(self.workload_dir)
        self.workload_dir.mkdir(parents=True)

        # 放置每次workloads的实测结果
        self.result_dir = self.workspace_dir.joinpath("results")
        if self.result_dir.exists():
            shutil.rmtree(self.result_dir)
        self.result_dir.mkdir(parents=True)

    @staticmethod
    def _safe_op_filename(op_name: str) -> str:
        return op_name.replace("/", "_").replace("\\", "_")

    @staticmethod
    def _flatten_nested_dict(obj: Union[Dict, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """将嵌套 dict 平铺为单层 key（a.b.c）；list 等非 dict 标量直接写入。"""
        items: Dict[str, Any] = {}
        if not isinstance(obj, dict):
            key = parent_key or "value"
            items[key] = obj
            return items
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.update(XpuPerfSimEngine._flatten_nested_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def _csv_scalar(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    @staticmethod
    def _fmt_print_float(v: Any, decimals: int = 3, omit_zero: bool = False) -> str:
        """控制台表格用，固定小数位，避免 float 二进制展开一长串。omit_zero 时数值为 0 返回空串。"""
        if v is None or v == "":
            return ""
        try:
            fv = float(v)
            if omit_zero and fv == 0.0:
                return ""
            return format(fv, f".{decimals}f")
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _ordered_result_csv_keys(flat_rows: List[Dict]) -> List[str]:
        """CSV 列顺序：主字段固定在前；arguments.*、targets.* 按各行 dict 迭代中首次出现的顺序，不做字母序重排。"""
        if not flat_rows:
            return []
        ordered: List[str] = []
        seen: set = set()
        for k in ("sku_name", "op_name", "provider"):
            if any(k in row for row in flat_rows):
                ordered.append(k)
                seen.add(k)
        for row in flat_rows:
            for k in row:
                if k in seen or not k.startswith("arguments."):
                    continue
                ordered.append(k)
                seen.add(k)
        for row in flat_rows:
            for k in row:
                if k in seen or not k.startswith("targets."):
                    continue
                ordered.append(k)
                seen.add(k)
        for row in flat_rows:
            for k in row:
                if k not in seen:
                    ordered.append(k)
                    seen.add(k)
        return ordered

    def _save_merged_bench_io(self, merged_workloads: Dict[str, List], bench_results_merged: Dict[str, List]) -> None:
        """
        workload: 与 micro_perf fa.json 一致，单文件 {op_name: [ 实例 dict, ... ] }。
        result: results/<op_name>/<provider>/<op_name>.jsonl|.csv；jsonl 字段：
        sku_name(同 device_name 展示名)、op_name、provider、arguments、targets。
        同一 workload 若多个 provider 有数据，会在各 provider 子目录各写一行；CSV 列主项在前，arguments/targets 子键保持原始出现顺序。
        """
        sku_name = self.device_name

        for op_name, workloads in merged_workloads.items():
            safe = self._safe_op_filename(op_name)
            wl_path = self.workload_dir / f"{safe}.json"
            with open(wl_path, "w", encoding="utf-8") as wf:
                json.dump(
                    {op_name: workloads},
                    wf,
                    ensure_ascii=False,
                    indent=4,
                    default=str,
                )

            results = bench_results_merged.get(op_name, [])
            provider_to_records: Dict[str, List[Dict]] = defaultdict(list)

            for i, workload in enumerate(workloads):
                raw = results[i] if i < len(results) else {}
                if not isinstance(raw, dict):
                    raw = {}
                if not raw:
                    provider_to_records["Unknown"].append(
                        {
                            "sku_name": sku_name,
                            "op_name": op_name,
                            "provider": "Unknown",
                            "arguments": copy.deepcopy(workload),
                            "targets": {"latency(us)": 0.0},
                        }
                    )
                    continue
                for provider, targets in raw.items():
                    if not isinstance(targets, dict):
                        continue
                    provider_to_records[str(provider)].append(
                        {
                            "sku_name": sku_name,
                            "op_name": op_name,
                            "provider": provider,
                            "arguments": copy.deepcopy(workload),
                            "targets": copy.deepcopy(targets),
                        }
                    )

            op_result_root = self.result_dir / safe
            for provider, records in provider_to_records.items():
                psafe = self._safe_op_filename(provider)
                out_dir = op_result_root / psafe
                out_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = out_dir / f"{safe}.jsonl"
                csv_path = out_dir / f"{safe}.csv"
                with open(jsonl_path, "w", encoding="utf-8") as jf:
                    for rec in records:
                        jf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

                flat_rows = [self._flatten_nested_dict(rec) for rec in records]
                if not flat_rows:
                    csv_path.write_text("", encoding="utf-8")
                    continue
                all_keys = self._ordered_result_csv_keys(flat_rows)
                with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.DictWriter(cf, fieldnames=all_keys, extrasaction="ignore")
                    writer.writeheader()
                    for row in flat_rows:
                        writer.writerow({k: self._csv_scalar(row.get(k)) for k in all_keys})

    @staticmethod
    def _normalize_breakdown_for_summary(breakdown: Dict) -> Dict:
        """
        parse_results 产出: (instance_name, idx) -> {provider: targets}
        summary 逻辑需要: (instance_name, idx) -> {provider: str, targets: dict}
        """
        out = {}
        for node, prov_map in breakdown.items():
            if not isinstance(prov_map, dict) or not prov_map:
                out[node] = {"provider": "Unknown", "targets": {"latency(us)": 0.0}}
                continue
            provider = next(iter(prov_map))
            targets = prov_map[provider]
            if not isinstance(targets, dict):
                targets = {"latency(us)": 0.0}
            out[node] = {"provider": provider, "targets": targets}
        return out

    
    def dump_extra_files(self):
        target_dir = DEFAULT_REPORT_DIR.joinpath(
            self.path_device_name, 
            self.model_name, 
            self.parallel_config_str, 
            self.infer_dtype, 
            self.run_mode
        )
        config_dict = {
            "device_name": self.device_name,
            "model_name": self.model_name,
            "impl_framework": "xpu_sim", 
            "infer_dtype": self.infer_dtype,
            "deploy_config_str": self.parallel_config_str,
            "deploy_config": {
                "node_num": self.dist_info.node_num,
                "device_num": self.dist_info.device_num,
                "pp_size": self.dist_info.pp_size,
                "dp_size": self.dist_info.dp_size,
                "sp_size": self.dist_info.sp_size,
                "tp_size": self.dist_info.tp_size,
                "ep_size": self.dist_info.ep_size,
            }, 
            "bench_mode": self.run_mode, 
            "envs": {}, 
            "extra_info": {
                "test_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)

        config_json_file = target_dir.joinpath("config.json")
        with open(config_json_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        latency_csv_file = target_dir.joinpath("latency.csv")
        latency_fieldnames = [
            "batch_size",
            "cache_len",
            "q_len",
            "stage_latency",
            "kernel_latency",
            "e2e_latency",
        ]
        with open(latency_csv_file, "w", newline="", encoding="utf-8") as f:
            lw = csv.DictWriter(f, fieldnames=latency_fieldnames)
            lw.writeheader()
            for (bs, cl, ql) in sorted(self.fix_data_dict.keys()):
                perf = self.fix_data_dict[(bs, cl, ql)]["perf_info"]
                stage_latency = round(perf["total_latency"], 3)
                kernel_latency = round(perf["total_node_latency"], 3)
                e2e_latency = round(stage_latency * self.num_layers * self.dist_info.pp_size, 3)
                lw.writerow(
                    {
                        "batch_size": bs,
                        "cache_len": cl,
                        "q_len": ql,
                        "stage_latency": stage_latency,
                        "kernel_latency": kernel_latency,
                        "e2e_latency": e2e_latency,
                    }
                )

        """
        save breakdown
        - perf.json
        - layers.json
        - head_non_layers.json
        - end_non_layers.json
        """
        breakdown_dir = target_dir.joinpath("summary")
        breakdown_dir.mkdir()

        for (batch_size, cache_len, q_len), entry in sorted(self.fix_data_dict.items()):
            breakdown_dict = self._normalize_breakdown_for_summary(entry["perf_info"]["breakdown"])
            summary_dir = breakdown_dir.joinpath(f"bs_{batch_size}.cache_{cache_len}.q_{q_len}")
            summary_dir.mkdir()

            with open(summary_dir.joinpath("test_case.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(entry["test_case"]), f, indent=4, ensure_ascii=False)

            head_non_layers_kernel_num = 0
            end_non_layers_kernel_num = 0
            per_layer_kernel_num = len(breakdown_dict)
            layers_kernel_num = per_layer_kernel_num * self.num_layers
            total_kernel_num = layers_kernel_num
            
        
            bubble_latency = 0.
            head_non_layers_latency = 0.
            end_non_layers_latency = 0.


            bubble_ratio = 0.
            head_non_layers_ratio = 0.
            end_non_layers_ratio = 0.
            layers_ratio = 0.


            head_non_layers_formatted_data = []
            end_non_layers_formatted_data = []
            layers_formatted_data = []

            per_layer_latency = 0.
            for (instance_name, instance_index), occur_data in breakdown_dict.items():
                per_layer_latency += occur_data["targets"].get("latency(us)", 0)
            layers_latency = per_layer_latency * self.num_layers
            model_latency = layers_latency
            


            for (instance_name, instance_index), occur_data in breakdown_dict.items():
                latency_us = occur_data["targets"].get("latency(us)", 0)
                kernel_provider = occur_data.get("provider", "unknown")
                kernel_mem_bw = occur_data["targets"].get("mem_bw(GB/s)", 0)
                kernel_comm_bw = occur_data["targets"].get("bus_bw(GB/s)", 0)
                kernel_flops = occur_data["targets"].get("calc_flops_power(tflops)", 0)
                
                kernel_ratio = round(
                    (latency_us / per_layer_latency) if per_layer_latency > 0 else 0.0,
                    5,
                )
                formatted_data = {
                    "kernel_type": "layers",
                    "kernel_name": instance_name,
                    "kernel_provider": kernel_provider, 
                    "kernel_index": instance_index,
                    "kernel_occurs": self.num_layers,
                    # "kernel_occur_indices": [per_layer_kernel_num * layer_id + instance_index for layer_id in range(self.num_layers)],
                    "kernel_occur_indices": [], 
                    "latency": round(latency_us, 3),
                    # "latency_list": [latency_us] * self.num_layers,
                    "latency_list": [], 
                    "kernel_ratio": kernel_ratio,
                    "kernel_ratio_str": f"{kernel_ratio * 100:.2f}%",
                    "kernel_mem_bw": kernel_mem_bw, 
                    "kernel_comm_bw": kernel_comm_bw, 
                    "kernel_flops": kernel_flops, 
                }
                layers_formatted_data.append(formatted_data)

            fixed_formatted_data = {
                "ori_num_layers": self.ori_num_layers,
                "num_layers": self.num_layers,
                "total_kernel_num": total_kernel_num, 
                "head_non_layers_kernel_num": head_non_layers_kernel_num,
                "end_non_layers_kernel_num": end_non_layers_kernel_num,
                "layers_kernel_num": layers_kernel_num,
                

                "per_layer_kernel_num": per_layer_kernel_num,
                "model_latency": round(model_latency, 3),
                "all_kernels_latency": round(model_latency, 3),
                "bubble_latency": round(bubble_latency, 3), 
                "head_non_layers_latency": round(head_non_layers_latency, 3),
                "end_non_layers_latency": round(end_non_layers_latency, 3),
                "layers_latency": round(layers_latency, 3),
                "per_layer_latency": round(per_layer_latency, 3),
                
                
                "bubble_ratio": bubble_ratio,
                "head_non_layers_ratio": head_non_layers_ratio,
                "end_non_layers_ratio": end_non_layers_ratio,
                "layers_ratio": layers_ratio,
                

                "bubble_ratio_str": f"{bubble_ratio * 100:.2f}%",
                "head_non_layers_ratio_str": f"{head_non_layers_ratio * 100:.2f}%",
                "end_non_layers_ratio_str": f"{end_non_layers_ratio * 100:.2f}%",
                "layers_ratio_str": f"{layers_ratio * 100:.2f}%",
            }

            with open(summary_dir.joinpath("perf.json"), "w", encoding="utf-8") as f:
                json.dump(fixed_formatted_data, f, indent=4, ensure_ascii=False)
            with open(summary_dir.joinpath("head_non_layers.json"), "w", encoding="utf-8") as f:
                json.dump(head_non_layers_formatted_data, f, indent=4, ensure_ascii=False)
            with open(summary_dir.joinpath("end_non_layers.json"), "w", encoding="utf-8") as f:
                json.dump(end_non_layers_formatted_data, f, indent=4, ensure_ascii=False)
            with open(summary_dir.joinpath("layers.json"), "w", encoding="utf-8") as f:
                json.dump(layers_formatted_data, f, indent=4, ensure_ascii=False)

            with open(summary_dir.joinpath("head_non_layers.csv"), "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(f, fieldnames=["kernel_name", "latency", "kernel_ratio", "kernel_ratio_str"])
                csv_writer.writeheader()
                for data in head_non_layers_formatted_data:
                    csv_writer.writerow(
                        {
                            "kernel_name": data["kernel_name"], 
                            "latency": data["latency"], 
                            "kernel_ratio": data["kernel_ratio"],
                            "kernel_ratio_str": data["kernel_ratio_str"],
                        }
                    )
            with open(summary_dir.joinpath("end_non_layers.csv"), "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(f, fieldnames=["kernel_name", "latency", "kernel_ratio", "kernel_ratio_str"])
                csv_writer.writeheader()
                for data in end_non_layers_formatted_data:
                    csv_writer.writerow(
                        {
                            "kernel_name": data["kernel_name"], 
                            "latency": data["latency"], 
                            "kernel_ratio": data["kernel_ratio"],
                            "kernel_ratio_str": data["kernel_ratio_str"],
                        }
                    )
            with open(summary_dir.joinpath("layers.csv"), "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(f, fieldnames=["kernel_name", "latency", "kernel_ratio", "kernel_ratio_str"])
                csv_writer.writeheader()
                for data in layers_formatted_data:
                    csv_writer.writerow(
                        {
                            "kernel_name": data["kernel_name"], 
                            "latency": data["latency"], 
                            "kernel_ratio": data["kernel_ratio"],
                            "kernel_ratio_str": data["kernel_ratio_str"],
                        }
                    )

            pt = prettytable.PrettyTable()
            pt.align = "l"
            pt.field_names = ["kernel_name", "kernel_provider", "num_occurs", "aver_latency (us)", "kernel_flops", "kernel_mem_bw", "kernel_comm_bw", "layer kernel (us)"]
            for data in layers_formatted_data:
                pt.add_row(
                    [
                        data["kernel_name"],
                        data["kernel_provider"],
                        self.num_layers,
                        self._fmt_print_float(data["latency"], 3, omit_zero=True),
                        self._fmt_print_float(data["kernel_flops"], 3, omit_zero=True),
                        self._fmt_print_float(data["kernel_mem_bw"], 3, omit_zero=True),
                        self._fmt_print_float(data["kernel_comm_bw"], 3, omit_zero=True),
                        self._fmt_print_float(per_layer_latency, 3, omit_zero=True),
                    ]
                )
            if not self.disable_print_result:
                print(pt)


    def dump_all_reports(self):
        if not self.all_data_list:
            return

        target_dir = DEFAULT_ALL_REPORT_DIR.joinpath(
            self.path_device_name,
            self.model_name,
            self.parallel_config_str,
            self.infer_dtype,
            self.run_mode
        )
        config_dict = {
            "device_name": self.device_name,
            "model_name": self.model_name,
            "impl_framework": "xpu_sim",
            "infer_dtype": self.infer_dtype,
            "deploy_config_str": self.parallel_config_str,
            "deploy_config": {
                "node_num": self.dist_info.node_num,
                "device_num": self.dist_info.device_num,
                "pp_size": self.dist_info.pp_size,
                "dp_size": self.dist_info.dp_size,
                "sp_size": self.dist_info.sp_size,
                "tp_size": self.dist_info.tp_size,
                "ep_size": self.dist_info.ep_size,
            },
            "bench_mode": self.run_mode,
            "envs": {},
            "extra_info": {
                "test_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)

        config_json_file = target_dir.joinpath("config.json")
        with open(config_json_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        shutil.copytree(self.workload_dir, target_dir.joinpath("workloads"))
        shutil.copytree(self.result_dir, target_dir.joinpath("results"))

        latency_csv_file = target_dir.joinpath("latency.csv")
        latency_fieldnames = [
            "index",
            "batch_size",
            "cache_len",
            "q_len",
            "cache_lens",
            "q_lens",
            "total_latency",
        ]
        with open(latency_csv_file, "w", newline="", encoding="utf-8") as f:
            lw = csv.DictWriter(f, fieldnames=latency_fieldnames)
            lw.writeheader()
            for entry in self.all_data_list:
                test_case = entry["test_case"]
                perf = entry["perf_info"]
                row = {"index": entry["index"], "total_latency": round(perf["total_latency"], 3)}
                if not test_case.is_var_input:
                    row["batch_size"] = test_case.batch_size
                    row["cache_len"] = test_case.cache_len
                    row["q_len"] = test_case.q_len
                else:
                    row["cache_lens"] = json.dumps(test_case.cache_lens)
                    row["q_lens"] = json.dumps(test_case.q_lens)
                lw.writerow(row)

        breakdown_dir = target_dir.joinpath("summary")
        breakdown_dir.mkdir()

        for entry in self.all_data_list:
            case_idx = entry["index"]
            test_case = entry["test_case"]
            breakdown_dict = self._normalize_breakdown_for_summary(entry["perf_info"]["breakdown"])
            summary_dir = breakdown_dir.joinpath(str(case_idx))
            summary_dir.mkdir()

            with open(summary_dir.joinpath("test_case.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(test_case), f, indent=4, ensure_ascii=False)

            head_non_layers_kernel_num = 0
            end_non_layers_kernel_num = 0
            per_layer_kernel_num = len(breakdown_dict)
            layers_kernel_num = per_layer_kernel_num * self.num_layers
            total_kernel_num = layers_kernel_num

            bubble_latency = 0.
            head_non_layers_latency = 0.
            end_non_layers_latency = 0.

            bubble_ratio = 0.
            head_non_layers_ratio = 0.
            end_non_layers_ratio = 0.
            layers_ratio = 0.

            head_non_layers_formatted_data = []
            end_non_layers_formatted_data = []
            layers_formatted_data = []

            per_layer_latency = 0.
            for (instance_name, instance_index), occur_data in breakdown_dict.items():
                per_layer_latency += occur_data["targets"].get("latency(us)", 0)
            layers_latency = per_layer_latency * self.num_layers
            model_latency = layers_latency

            for (instance_name, instance_index), occur_data in breakdown_dict.items():
                latency_us = occur_data["targets"].get("latency(us)", 0)
                kernel_provider = occur_data.get("provider", "unknown")
                kernel_mem_bw = occur_data["targets"].get("mem_bw(GB/s)", 0)
                kernel_comm_bw = occur_data["targets"].get("bus_bw(GB/s)", 0)
                kernel_flops = occur_data["targets"].get("calc_flops_power(tflops)", 0)

                kernel_ratio = round(
                    (latency_us / per_layer_latency) if per_layer_latency > 0 else 0.0,
                    5,
                )
                formatted_data = {
                    "kernel_type": "layers",
                    "kernel_name": instance_name,
                    "kernel_provider": kernel_provider,
                    "kernel_index": instance_index,
                    "kernel_occurs": self.num_layers,
                    "kernel_occur_indices": [],
                    "latency": round(latency_us, 3),
                    "latency_list": [],
                    "kernel_ratio": kernel_ratio,
                    "kernel_ratio_str": f"{kernel_ratio * 100:.2f}%",
                    "kernel_mem_bw": kernel_mem_bw,
                    "kernel_comm_bw": kernel_comm_bw,
                    "kernel_flops": kernel_flops,
                }
                layers_formatted_data.append(formatted_data)

            fixed_formatted_data = {
                "ori_num_layers": self.ori_num_layers,
                "num_layers": self.num_layers,
                "total_kernel_num": total_kernel_num,
                "head_non_layers_kernel_num": head_non_layers_kernel_num,
                "end_non_layers_kernel_num": end_non_layers_kernel_num,
                "layers_kernel_num": layers_kernel_num,

                "per_layer_kernel_num": per_layer_kernel_num,
                "model_latency": round(model_latency, 3),
                "all_kernels_latency": round(model_latency, 3),
                "bubble_latency": round(bubble_latency, 3),
                "head_non_layers_latency": round(head_non_layers_latency, 3),
                "end_non_layers_latency": round(end_non_layers_latency, 3),
                "layers_latency": round(layers_latency, 3),
                "per_layer_latency": round(per_layer_latency, 3),

                "bubble_ratio": bubble_ratio,
                "head_non_layers_ratio": head_non_layers_ratio,
                "end_non_layers_ratio": end_non_layers_ratio,
                "layers_ratio": layers_ratio,

                "bubble_ratio_str": f"{bubble_ratio * 100:.2f}%",
                "head_non_layers_ratio_str": f"{head_non_layers_ratio * 100:.2f}%",
                "end_non_layers_ratio_str": f"{end_non_layers_ratio * 100:.2f}%",
                "layers_ratio_str": f"{layers_ratio * 100:.2f}%",
            }

            with open(summary_dir.joinpath("perf.json"), "w", encoding="utf-8") as f:
                json.dump(fixed_formatted_data, f, indent=4, ensure_ascii=False)
            with open(summary_dir.joinpath("head_non_layers.json"), "w", encoding="utf-8") as f:
                json.dump(head_non_layers_formatted_data, f, indent=4, ensure_ascii=False)
            with open(summary_dir.joinpath("end_non_layers.json"), "w", encoding="utf-8") as f:
                json.dump(end_non_layers_formatted_data, f, indent=4, ensure_ascii=False)
            with open(summary_dir.joinpath("layers.json"), "w", encoding="utf-8") as f:
                json.dump(layers_formatted_data, f, indent=4, ensure_ascii=False)

            csv_fieldnames = ["kernel_name", "latency", "kernel_ratio", "kernel_ratio_str"]
            with open(summary_dir.joinpath("head_non_layers.csv"), "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                csv_writer.writeheader()
                for data in head_non_layers_formatted_data:
                    csv_writer.writerow({k: data[k] for k in csv_fieldnames})
            with open(summary_dir.joinpath("end_non_layers.csv"), "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                csv_writer.writeheader()
                for data in end_non_layers_formatted_data:
                    csv_writer.writerow({k: data[k] for k in csv_fieldnames})
            with open(summary_dir.joinpath("layers.csv"), "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                csv_writer.writeheader()
                for data in layers_formatted_data:
                    csv_writer.writerow({k: data[k] for k in csv_fieldnames})


    def parse_model(self, **kwargs):
        # 解析当前 bench 配置
        self.model_name = self.bench_config.get("model_name", "")
        self.infer_dtype = self.bench_config.get("infer_dtype", "")
        if self.model_name == "" or self.infer_dtype == "":
            raise ValueError("model_name or infer_dtype is empty")
        
        # 解析并行方式（语义、约束与展示串格式见 model_perf.utils.DistributionInfo）
        parallel_config = self.bench_config.get("parallel_config", {})
        device_num = parallel_config.get("device_num", 1)
        if self.actual_device_num < device_num:
            raise ValueError(
                f"actual_device_num: {self.actual_device_num} < device_num: {device_num}"
            )

        self.dist_info = DistributionInfo(
            node_num=1,  # 暂时只支持单 node
            device_num=device_num,
            dp_size=parallel_config.get("dp_size", 1),
            pp_size=parallel_config.get("pp_size", 1),
            sp_size=parallel_config.get("sp_size", 1),
            tp_size=parallel_config.get("tp_size", 1),
            ep_size=parallel_config.get("ep_size", 1),
        )
        self.parallel_config_str = self.dist_info.get_dist_info_str()


        # 获取 bench_mode
        self.run_mode = kwargs.get("run_mode", self.run_mode)
        self.ori_num_layers = self.model_config.num_layers[0]
        self.test_num_layers = self.ori_num_layers
        num_layers = self.ori_num_layers
        num_mirror_layers = self.model_config.num_mirror_layers
        if self.run_mode == "prefill":
            num_layers -= num_mirror_layers
            self.test_num_layers -= num_mirror_layers

        # 根据 pp_size 调整层数和 xpu_config
        if self.dist_info.pp_size > 1:
            num_layers = (num_layers + self.dist_info.pp_size - 1) // self.dist_info.pp_size

        self.num_layers = num_layers


        # 打印当前配置项
        pt = prettytable.PrettyTable(["attr", "value"])
        pt.align = "l"

        # 基础模型配置
        pt.add_row(["model_name", self.model_name])
        pt.add_row(["infer_dtype", self.infer_dtype])
        pt.add_row(["parallel_config", self.parallel_config_str])
        pt.add_row(["run_mode", self.run_mode])
        pt.add_row(["-" * 25, "-" * 50])

        # 并行策略（展示顺序 PP → DP → SP → TP → EP）
        pt.add_row(["deploy_node_num", self.dist_info.node_num])
        pt.add_row(["deploy_device_num", self.dist_info.device_num])
        pt.add_row(["pp_size", self.dist_info.pp_size])
        pt.add_row(["dp_size", self.dist_info.dp_size])
        pt.add_row(["sp_size", self.dist_info.sp_size])
        pt.add_row(["tp_size", self.dist_info.tp_size])
        pt.add_row(["ep_size", self.dist_info.ep_size])

        # 模型配置
        pt.add_row(["ori_num_layers", self.ori_num_layers])
        pt.add_row(["test_num_layers", self.test_num_layers])
        pt.add_row(["num_layers", num_layers])
        print(pt)
        print("")


    def bench(self, test_cases: List[BenchTestCase]):
        self.model_topo.set_bench_info(test_cases, run_mode=self.run_mode)
        merged_workloads = self.model_topo.merge_batched_bench_workloads()
        bench_results_merged = self.send_bench_request(merged_workloads)
        self._save_merged_bench_io(merged_workloads, bench_results_merged)
        bench_results_list = self.model_topo.split_batched_bench_results(bench_results_merged)
        parsed_results_list = self.model_topo.parse_results(bench_results_list)

        for case_idx, (test_case, parsed_results) in enumerate(
            zip(test_cases, parsed_results_list)
        ):
            if not self.disable_print_result and len(test_cases) > 1:
                print(f"\n========== bench test case {case_idx} ==========\n")

            node_times, critical_path, total_latency, node_cost, node_provider, total_node_cost, critical_total = self.model_topo.calculate_timeline(parsed_results)
            if not self.disable_print_result:
                self.model_topo.print_schedule(parsed_results)

            temp_pt = prettytable.PrettyTable(["instance_name", "instance_index", "provider", "latency(us)", "tflops", "mem_bw", "comm_bw"])
            temp_pt.align = "l"
            for (instance_name, instance_index), result_info in parsed_results.items():
                provider = list(result_info.keys())[0]
                provider_result = result_info[provider]

                tflops_value = provider_result.get("calc_flops_power(tflops)", 0)
                mem_bw_value = provider_result.get("mem_bw(GB/s)", 0)
                bus_bw_value = provider_result.get("bus_bw(GB/s)", 0)

                temp_pt.add_row([
                    instance_name, instance_index,
                    provider, provider_result["latency(us)"],
                    tflops_value if tflops_value > 0 else "",
                    mem_bw_value if mem_bw_value > 0 else "",
                    bus_bw_value if bus_bw_value > 0 else ""])
            if not self.disable_print_result:
                print(temp_pt)

            perf_info = {
                "e2e_latency": round(total_latency * self.num_layers / 1000, 3),
                "total_latency": total_latency,
                "total_node_latency": total_node_cost,
                "breakdown": parsed_results
            }

            self.all_data_list.append({
                "index": case_idx,
                "test_case": test_case,
                "perf_info": perf_info
            })

            if not test_case.is_var_input:
                item_key = (test_case.batch_size, test_case.cache_len, test_case.q_len)
                self.fix_data_dict[item_key] = {
                    "test_case": test_case,
                    "perf_info": perf_info
                }
            else:
                self.var_data_dict.append({
                    "test_case": test_case,
                    "perf_info": perf_info
                })

        self.dump_info()
            



    def send_bench_request(self, workloads):
        self.client.reset_progress()
        empty_results: Dict = {}
        for op_name in workloads:
            empty_results[op_name] = [{} for _ in workloads[op_name]]

        result = self.client.send_bench({"type": "normal", "data": workloads})
        return result if result else empty_results



    def summary_results(self, data_dict):   
        test_case = data_dict["test_case"].get_summary_dict()

        pt = prettytable.PrettyTable()
        pt.field_names = ["key", "value"]
        pt.align = "l"
        pt.add_row(["run_mode", self.run_mode])
        for key, value in test_case.items():
            pt.add_row([key, value])
        pt.add_row(["-" * 25, "-" * 50])
        pt.add_row(["total_latency", f"{data_dict['perf_info']['e2e_latency']} ms"])
        if not self.disable_print_result:
            print(pt)
            print("")



    def dump_info(self):
        if self.fix_data_dict:
            for value in self.fix_data_dict.values():
                self.summary_results(value)

        if self.var_data_dict:
            for value in self.var_data_dict:
                self.summary_results(value)

        self.dump_extra_files()
        self.dump_all_reports()

