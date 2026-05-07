import sys
import csv
import copy
import json
import string
import random
import hashlib
import pathlib
import datetime
import requests
import importlib
import prettytable
from typing import Any, List, Dict, Tuple, ClassVar, Optional, Callable
from dataclasses import dataclass, field, asdict, fields
from collections import OrderedDict


def flexible_json_parse(value: str):
    """
    宽松地将 CSV 单元格字符串解析为 Python 对象。
    解析优先级: json.loads 直接解析 -> 补齐方括号再解析 -> 逗号切分为字符串列表 -> 原样返回
    """
    v = value.strip()
    if not v:
        return v

    # 1. 尝试直接解析 (处理数字、已带括号的列表、true/false等)
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        pass

    # 2. 如果包含逗号且不带括号，尝试补齐括号再解析
    if ',' in v and not (v.startswith('[') and v.endswith(']')):
        try:
            # 补齐括号并再次尝试
            return json.loads(f"[{v}]")
        except json.JSONDecodeError:
            # 如果补齐括号还失败，说明可能是 "qwen, moe" 这种没加引号的字符串列表
            # 处理方式：按逗号切分并去除每个元素的空格
            return [item.strip() for item in v.split(',')]

    # 3. 万策尽，返回原字符串
    return value



"""
分布式并行配置, 描述模型部署时各维度的并行方式。

device_num:
    - 实际参与计算的逻辑设备, 对应在硬件上有对应的实际拓扑和链路带宽

pp_size: 
    - 层间并行, 切分 num_layers
    - 测试时不会考虑这一维度并行, 仅在组装端到端性能时考虑
    - 会引入 PP Unbalance

dp_size: 
    - 数据并行, 切分 batch_size, 一般用于 decode 的 attn 部分
    - 测试时只需要考虑 单dp_rank (可能包含多个device) 的算子性能
      和 sp_size 互斥
    - 会引入 DP Unbalance

tp_size: 
    - 数据并行, 对于 attn 模块切分 head, 对于 mlp 部分切分两个gemm之间的 N/K 轴。
    - 测试时只需要考虑 单tp_rank 的算子性能
    - 不引入 Unbalance

sp_size:
    - 数据并行, 切分 sum(q_lens) = num_tokens, 
      贯穿所有模块, 在 attn 模块中需要适时转换成 tp 再转回 sp
      因此通常和 tp_size 成对出现, 与 dp_size 互斥
    - 不引入 Unbalance

ep_size: 
    - 专家并行, 切分 num_experts
    - 测试时只需要考虑 单ep_rank 的算子性能
    - 会引入 EP Unbalance

目前常用的并行方式有:
    - pp8
    - pp16
    - pp8-tp2-ep2
    - tp8
    - tp8-ep8
    - sp8-tp8-ep8
    - dp16-ep16
"""
@dataclass
class DistributionInfo:
    node_num: int = 1
    device_num: int = 1
    bench_device_num: int = 1   # 实际测试时需要采用的
    
    pp_size: int = 1
    dp_size: int = 1
    sp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1

    def __post_init__(self):
        if self.device_num < 1 \
            or self.pp_size < 1 \
            or self.dp_size < 1 \
            or self.sp_size < 1 \
            or self.tp_size < 1 \
            or self.ep_size < 1:
            raise ValueError("device_num, pp_size, dp_size, sp_size, tp_size, ep_size must be greater than 0")

        if self.device_num == 1:
            self.pp_size = 1
            self.dp_size = 1
            self.sp_size = 1
            self.tp_size = 1
            self.ep_size = 1
            return
        else:
            if self.pp_size > self.device_num:
                raise ValueError("pp_size must be less than or equal to device_num")
            if self.dp_size > self.device_num:
                raise ValueError("dp_size must be less than or equal to device_num")
            if self.sp_size > self.device_num:
                raise ValueError("sp_size must be less than or equal to device_num")
            if self.tp_size > self.device_num:
                raise ValueError("tp_size must be less than or equal to device_num")
            if self.ep_size > self.device_num:
                raise ValueError("ep_size must be less than or equal to device_num")

        self.bench_device_num = self.device_num // self.pp_size


        """
        - 如果有 dp, 一般用于decode:  attn DP+TP, dp rank 之间完全独立, 然后 ffn 纯EP
        - 如果有 sp, 一般用于prefill: attn SP+TP, sp 与 tp 互斥, 然后 ffn 纯EP
        - 如果只有 tp, prefill 和 decode 都可以使用
        现在一般要求 moe ffn部分是纯粹的ep
        """
        if self.dp_size > 1:
            if self.sp_size > 1:
                raise ValueError("dp_size and sp_size cannot be set at the same time")
            if self.dp_size * self.tp_size != self.bench_device_num:
                raise ValueError("dp_size * tp_size must be equal to bench_device_num")
            if self.ep_size > 1 and self.ep_size != self.bench_device_num:
                raise ValueError("ep_size must be equal to bench_device_num when dp_size > 1")
        elif self.sp_size > 1:
            if self.dp_size > 1:
                raise ValueError("dp_size and sp_size cannot be set at the same time")
            if self.sp_size != self.tp_size or self.sp_size != self.bench_device_num:
                raise ValueError("sp_size must be equal to tp_size and bench_device_num when sp_size > 1")
            if self.ep_size > 1 and self.ep_size != self.bench_device_num:
                raise ValueError("ep_size must be equal to bench_device_num when sp_size > 1")
        else:
            if self.tp_size != self.bench_device_num:
                raise ValueError("tp_size must be equal to bench_device_num when dp_size and sp_size are not set")
            if self.ep_size > 1 and self.ep_size != self.bench_device_num:
                raise ValueError("ep_size must be equal to bench_device_num when tp_size > 1")

    @classmethod
    def from_bench_config(cls, config: Dict[str, Any]):
        return cls(**config)
    

    def get_dist_info_str(self) -> str:
        """并行展示串：{node_num}N{device_num}D[_PP..][_DP..][_SP..][_TP..][_EP..]（报表路径与 xpu_sim 一致）。"""
        parallel_config_str = f"{self.node_num}N{self.device_num}D"
        if self.pp_size > 1:
            parallel_config_str += f"_PP{self.pp_size}"
        if self.dp_size > 1:
            parallel_config_str += f"_DP{self.dp_size}"
        if self.sp_size > 1:
            parallel_config_str += f"_SP{self.sp_size}"
        if self.tp_size > 1:
            parallel_config_str += f"_TP{self.tp_size}"
        if self.ep_size > 1:
            parallel_config_str += f"_EP{self.ep_size}"
        return parallel_config_str

    def to_wire_dict(self) -> Dict[str, Any]:
        """服务端 / 客户端交互用的并行配置字典（可与 from_wire_dict 对称解析）。"""
        return {
            "node_num": self.node_num,
            "device_num": self.device_num,
            "pp_size": self.pp_size,
            "dp_size": self.dp_size,
            "sp_size": self.sp_size,
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
        }

    @classmethod
    def from_wire_dict(cls, d: Dict[str, Any]) -> "DistributionInfo":
        """由 get_info / deploy_config / distribution 字段还原。"""
        return cls(
            node_num=int(d.get("node_num", 1)),
            device_num=int(d.get("device_num", 1)),
            pp_size=int(d.get("pp_size", 1)),
            dp_size=int(d.get("dp_size", 1)),
            sp_size=int(d.get("sp_size", 1)),
            tp_size=int(d.get("tp_size", 1)),
            ep_size=int(d.get("ep_size", 1)),
        )

    @classmethod
    def from_parallel_config_dict(
        cls, parallel_config: Dict[str, Any], *, node_num: int = 1
    ) -> "DistributionInfo":
        """由 bench_config.parallel_config 构造（xpu_gpt engine）。"""
        return cls(
            node_num=int(node_num),
            device_num=int(parallel_config.get("device_num", 1)),
            pp_size=int(parallel_config.get("pp_size", 1)),
            dp_size=int(parallel_config.get("dp_size", 1)),
            sp_size=int(parallel_config.get("sp_size", 1)),
            tp_size=int(parallel_config.get("tp_size", 1)),
            ep_size=int(parallel_config.get("ep_size", 1)),
        )



"""
性能测试用例, 描述一次推理请求的输入形状和 KV Cache 布局。

核心职责:
1. 统一表达 prefill / decode 阶段的输入参数:
   - 均匀输入: 通过 (batch_size, cache_len, q_len) 指定, 每条序列形状相同
   - 变长输入: 通过 (cache_lens, q_lens) 列表直接指定, 每条序列可不同
2. 根据 block_size 自动计算 KV Cache 的物理布局:
   - block_size = 0  -> linear cache, 使用 slot_mapping 映射
   - block_size > 0  -> paged cache,  使用 block_table 管理物理块
3. 提供从命令行参数 / CSV 文件批量构建测试用例的工厂方法 (from_args)
4. 跨进程 / 网络传递: get_dict_info() 与 from_serialized_dict() 对称; get_summary_dict() 仅用于展示
"""
@dataclass
class BenchTestCase:
    ALLOWED_KEYS: ClassVar[Tuple[str, ...]] = (
        "batch_size", 
        "cache_len", 
        "q_len", 

        "cache_lens", 
        "q_lens", 

        "block_size", 
        "slot_mapping", 
        "block_table", 
    )


    batch_size: int = 1
    cache_len: int = 0
    q_len: int = 10240

    cache_lens: List[int] = field(default_factory=list)
    q_lens: List[int] = field(default_factory=list)
    kv_lens: List[int] = field(default_factory=list)

    num_tokens: int = 0

    is_var_input: bool = False
    is_q_len_fixed: bool = True
    max_q_len: int = 0
    max_cache_len: int = 0
    max_kv_len: int = 0 

    """
    when block_size = 0, use linear cache
    when block_size > 0, use paged cache
    """
    block_size: int = 512
    cache_type: str = "paged"
    
    # used for linear cache
    slot_mapping: List[int] = field(default_factory=list)

    # used for paged cache
    block_table: List[List[int]] = field(default_factory=list)

    cache_blocks: List[int] = field(default_factory=list)
    q_blocks: List[int] = field(default_factory=list)
    kv_blocks: List[int] = field(default_factory=list)

    num_cache_blocks: int = 0
    num_q_blocks: int = 0
    num_kv_blocks: int = 0

    max_num_blocks_per_seq: int = 0
    total_num_blocks: int = 0
    max_block_id: int = -1    

    
    def __post_init__(self):
        if not self.cache_lens and not self.q_lens:
            self.cache_lens = [self.cache_len] * self.batch_size
            self.q_lens = [self.q_len] * self.batch_size
            self.is_var_input = False
        else:
            min_batch_size = min(len(self.cache_lens), len(self.q_lens))
            self.cache_lens = self.cache_lens[:min_batch_size]
            self.q_lens = self.q_lens[:min_batch_size]
            self.batch_size = min_batch_size
            self.is_var_input = True

        self.kv_lens = [cache_len + q_len for cache_len, q_len in zip(self.cache_lens, self.q_lens)]
        
        self.is_q_len_fixed = (len(set(self.q_lens)) == 1)
        self.max_cache_len = max(self.cache_lens)
        self.max_q_len = max(self.q_lens)
        self.max_kv_len = max(self.kv_lens)

        self.num_tokens = sum(self.q_lens)


        if self.block_size == 0:
            self.cache_type = "linear"

            if not self.slot_mapping:
                default_slot_mapping = list(range(self.batch_size))
                self.slot_mapping = default_slot_mapping
            else:
                # 校验输入的 slot_mapping 是否合法
                if set(self.slot_mapping) != set(range(self.batch_size)):
                    raise ValueError(f"slot_mapping must be a list of integers from 0 to {self.batch_size - 1}")

        elif self.block_size > 0:
            self.cache_type = "paged"

            self.cache_blocks = [(cache_len + (self.block_size - 1)) // self.block_size for cache_len in self.cache_lens]
            self.q_blocks = [(q_len + (self.block_size - 1)) // self.block_size for q_len in self.q_lens]
            self.kv_blocks = [(kv_len + (self.block_size - 1)) // self.block_size for kv_len in self.kv_lens]

            self.num_cache_blocks = sum(self.cache_blocks)
            self.num_q_blocks = sum(self.q_blocks)
            self.num_kv_blocks = sum(self.kv_blocks)

            self.max_num_blocks_per_seq = max(self.kv_blocks)
            self.total_num_blocks = self.batch_size * self.max_num_blocks_per_seq
            
            # 如果未提供 block_table，则生成默认的 block_table
            if not self.block_table:
                default_block_table = []
                block_idx = 0
                for cur_seq_id in range(self.batch_size):
                    cur_seq_blocks = []
                    for _ in range(self.kv_blocks[cur_seq_id]):
                        cur_seq_blocks.append(block_idx)
                        block_idx += 1
                    cur_seq_blocks.extend([-1] * (self.max_num_blocks_per_seq - len(cur_seq_blocks)))
                    default_block_table.append(cur_seq_blocks)
                self.block_table = default_block_table
                self.max_block_id = self.total_num_blocks - 1

            # 如果提供 block_table，则校验输入的 block_table 是否合法
            else:
                if len(self.block_table) < self.batch_size:
                    raise ValueError(f"block_table must have at least {self.batch_size} sequences")
                if len(set([len(seq_blocks) for seq_blocks in self.block_table])) != 1:
                    raise ValueError(f"block_table must have the same number of blocks per sequence")
                for cur_seq_id, cur_seq_blocks in enumerate(self.block_table):
                    if len(cur_seq_blocks) > self.max_num_blocks_per_seq:
                        cur_seq_blocks = cur_seq_blocks[:self.max_num_blocks_per_seq]
                    elif len(cur_seq_blocks) < self.kv_blocks[cur_seq_id]:
                        raise ValueError(f"block_table[{cur_seq_id}] must have at least {self.kv_blocks[cur_seq_id]} blocks")
                    else:
                        cur_seq_blocks.extend([-1] * (self.max_num_blocks_per_seq - len(cur_seq_blocks)))
                    self.max_block_id = max(self.max_block_id, max(cur_seq_blocks))
            self.total_num_blocks = self.max_block_id + 1
        else:
            raise ValueError(f"invalid block_size: {self.block_size}")


    @classmethod
    def from_args(cls, args):
        test_cases = []

        if args.generator is not None:
            pass

        elif args.csv is not None:
            with open(args.csv, "r") as f:
                reader = csv.DictReader(f)
                for data in reader:
                    filter_args = {k: v for k, v in data.items() if k in cls.ALLOWED_KEYS}
                    for key, value in filter_args.items():
                        filter_args[key] = flexible_json_parse(value)
                    filter_args["block_size"] = args.block_size
                    test_cases.append(cls(**filter_args))
        else:
            test_cases.append(cls(
                batch_size=args.batch_size,
                cache_len=args.cache_len,
                q_len=args.q_len,
                block_size=args.block_size,
            ))
        return test_cases

    def get_dict_info(self) -> Dict[str, Any]:
        """导出全部 dataclass 字段，用于多进程 / HTTP 等对 TestCase 的完整传递（与 from_serialized_dict 对称）。"""
        return asdict(self)

    def get_summary_dict(self) -> Dict[str, Any]:
        """仅含主要输入与布局相关字段，用于日志、表格展示等简要场景。"""
        if self.is_var_input:
            out: Dict[str, Any] = {
                "batch_size": self.batch_size,
                "cache_lens": self.cache_lens,
                "q_lens": self.q_lens,
                "block_size": self.block_size,
            }
        else:
            out = {
                "batch_size": self.batch_size,
                "cache_len": self.cache_len,
                "q_len": self.q_len,
                "block_size": self.block_size,
            }
        return out

    @classmethod
    def from_serialized_dict(cls, d: Dict[str, Any]) -> "BenchTestCase":
        """
        由 get_dict_info() / JSON 全量字典还原实例，不再次执行 __post_init__。

        要求包含全部 dataclass 字段；多出的键忽略。用于 /bench、多进程等与 get_dict_info 对称的一进一出。
        """
        field_names = {f.name for f in fields(cls)}
        missing = field_names - set(d.keys())
        if missing:
            raise ValueError(
                "dict must include every field from BenchTestCase.get_dict_info(); "
                f"missing: {sorted(missing)}"
            )
        obj = object.__new__(cls)
        for f in fields(cls):
            setattr(obj, f.name, copy.deepcopy(d[f.name]))
        return obj



"""
通用的 bench 请求客户端, 封装与 micro_perf / xpu_gpt 等后端服务的 HTTP 通信。

核心能力:
1. GET /info  — 获取服务端信息
2. POST /bench — 发送测试请求, 支持普通和 NDJSON 流式两种模式
3. 流式模式下按算子维度实时展示进度条
"""
class BenchClient:
    def __init__(self, ip: str = "localhost", port: int = 49371, stream: bool = True):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"
        self.info_url = f"{self.base_url}/info"
        self.bench_url = f"{self.base_url}/bench"
        self.stream = stream

        self._bench_op_progress: OrderedDict[str, Dict[str, int]] = OrderedDict()
        self._bench_op_progress_line_count = 0

    def get_info(self) -> Dict:
        try:
            response = requests.get(self.info_url)
            return response.json()
        except Exception:
            return {}

    def send_bench(
        self,
        data: Dict,
        *,
        per_case_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        session_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict:
        """
        发送 bench 请求。data 为完整的请求体, 由调用方负责构造。
        当 self.stream=True 时启用 NDJSON 流式响应并展示进度条。
        若服务端先推送 event=session，可 session_callback(data)（如 workspace_dir、model_arch_txt_b64）。
        若服务端逐条推送 event=case_done，可对每条调用 per_case_callback(index, result_dict)。
        """
        try:
            if self.stream:
                data_with_stream = {**data, "stream_progress": True}
                response = requests.post(
                    self.bench_url,
                    json=data_with_stream,
                    stream=True,
                    timeout=None,
                )
                if not response.ok:
                    snippet = (response.text or "")[:800]
                    print(
                        f"[BenchClient] POST /bench HTTP {response.status_code}: {snippet}",
                        file=sys.stderr,
                    )
                    return {}
                return self._consume_stream(
                    response,
                    per_case_callback=per_case_callback,
                    session_callback=session_callback,
                )

            response = requests.post(self.bench_url, json=data)
            if not response.ok:
                snippet = (response.text or "")[:800]
                print(
                    f"[BenchClient] POST /bench HTTP {response.status_code}: {snippet}",
                    file=sys.stderr,
                )
                return {}
            try:
                return response.json()
            except Exception as e:
                print(f"[BenchClient] bench response is not JSON: {e}", file=sys.stderr)
                return {}
        except Exception as e:
            print(f"[BenchClient] send_bench failed: {e}", file=sys.stderr)
            return {}

    # ------------------------------------------------------------------
    # NDJSON 流式响应处理 + 进度条 UI
    # ------------------------------------------------------------------

    def _consume_stream(
        self,
        response,
        *,
        per_case_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        session_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict:
        """解析 application/x-ndjson 流; 若服务端为旧版单次 JSON 则整段当结果。"""
        legacy: Dict = {}
        saw_progress = False
        case_done_results: List[Dict[str, Any]] = []
        session_data: Dict[str, Any] = {}
        try:
            for line in response.iter_lines(decode_unicode=True, chunk_size=65536):
                if not line or not str(line).strip():
                    continue
                obj = json.loads(line)
                ev = obj.get("event")
                if ev == "progress":
                    saw_progress = True
                    self._on_progress(obj)
                elif ev == "session":
                    saw_progress = True
                    session_data = dict(obj.get("data") or {})
                    if session_callback is not None:
                        session_callback(session_data)
                elif ev == "case_done":
                    saw_progress = True
                    payload = obj.get("data") or {}
                    if not isinstance(payload, dict):
                        payload = {}
                    case_done_results.append(payload)
                    idx = int(obj.get("index", len(case_done_results) - 1))
                    if per_case_callback is not None:
                        per_case_callback(idx, payload)
                elif ev == "done":
                    self._redraw_progress_bars()
                    tail = obj.get("data") or {}
                    merged = dict(session_data)
                    merged.update(tail)
                    if case_done_results:
                        merged["batch_results"] = case_done_results
                        return merged
                    return merged if session_data else tail
                elif ev == "error":
                    raise RuntimeError(obj.get("message", "bench server error"))
                else:
                    legacy = obj
            if saw_progress:
                print(
                    "[BenchClient] NDJSON stream ended before event=done "
                    "(proxy buffer / disconnect / server crash?).",
                    file=sys.stderr,
                )
                return {}
            return legacy
        finally:
            self._end_progress_ui()

    def _on_progress(self, obj: Dict) -> None:
        op = obj.get("op_name") or "?"
        tot = int(obj.get("op_total_steps") or obj.get("cases_for_op") or 1)
        tot = max(tot, 1)
        if op not in self._bench_op_progress:
            self._bench_op_progress[op] = {"done": 0, "total": tot}
        st = self._bench_op_progress[op]
        st["total"] = max(st["total"], tot)
        st["done"] += 1
        d, t = st["done"], st["total"]
        if d >= t:
            should_redraw = True
        elif t <= 64:
            # 少量 step（如 xpu_gpt 按用例数）每步刷新，避免条数 <10 时只在最后刷一次
            should_redraw = True
        else:
            should_redraw = d % 10 == 0
        if should_redraw:
            self._redraw_progress_bars()

    def _redraw_progress_bars(self) -> None:
        bar_w = 24
        lines = []
        for op, st in self._bench_op_progress.items():
            d, t = st["done"], max(st["total"], 1)
            pct = min(100.0, 100.0 * d / t)
            filled = min(bar_w, int(bar_w * d / t))
            bar = "#" * filled + "-" * (bar_w - filled)
            disp = op if len(op) <= 37 else op[:36] + "…"
            lines.append(f"  {disp:37} [{bar}] {d:>4}/{t:<4} {pct:5.1f}%")
        if not lines:
            return
        out = sys.stdout
        if out.isatty():
            n = len(lines)
            if self._bench_op_progress_line_count > 0:
                out.write(f"\033[{self._bench_op_progress_line_count}A")
            for line in lines:
                out.write("\r\033[K" + line + "\n")
            out.flush()
            self._bench_op_progress_line_count = n
        else:
            for line in lines:
                print(line, flush=True)
            self._bench_op_progress_line_count = len(lines)

    def _end_progress_ui(self) -> None:
        if self._bench_op_progress_line_count > 0 and sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._bench_op_progress_line_count = 0

    def reset_progress(self) -> None:
        """在新一轮 bench 请求前调用, 重置进度条状态。"""
        self._end_progress_ui()
        self._bench_op_progress.clear()







@dataclass
class KernelTrace:
    trace_data: Dict[str, Any] = field(default_factory=dict)

    index: int = 0
    name: str = ""

    start_time: float = 0.0
    end_time: float = 0.0
    latency: float = 0.0

    stream_id: int = 0

    @classmethod
    def from_trace_event(cls, trace_event: Dict[str, Any]):
        instance = cls()

        instance.name = trace_event["name"]
        instance.latency = trace_event["dur"]
        instance.start_time = float(trace_event["ts"])
        instance.end_time = instance.start_time + instance.latency
        instance.stream_id = trace_event["tid"]
        instance.trace_data = trace_event
        return instance

    def is_same_stream(self, other: "KernelTrace") -> bool:
        return self.stream_id == other.stream_id

    def is_before(self, other: "KernelTrace") -> bool:
        return self.end_time < other.start_time

    def is_after(self, other: "KernelTrace") -> bool:
        return self.start_time > other.end_time


def _stream_id_sort_key(sid: Any) -> Tuple[Any, ...]:
    """用于稳定排序的 stream_id 键。"""
    try:
        return (0, int(sid))
    except (TypeError, ValueError):
        return (1, str(sid))


def _build_multi_stream_components(
    trace_list: List[KernelTrace],
) -> List[List[int]]:
    """构造跨 stream 重叠的连通分量，返回各分量在 ``trace_list`` 中的下标列表。"""
    n = len(trace_list)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    active: List[int] = []
    for i, cur in enumerate(trace_list):
        active = [j for j in active if trace_list[j].end_time > cur.start_time]
        for j in active:
            prev = trace_list[j]
            if prev.stream_id == cur.stream_id:
                continue
            if prev.end_time > cur.start_time and cur.end_time > prev.start_time:
                union(i, j)
        active.append(i)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    components: List[List[int]] = []
    for idxs in groups.values():
        if len(idxs) <= 1:
            continue
        stream_ids = {trace_list[i].stream_id for i in idxs}
        if len(stream_ids) <= 1:
            continue
        idxs.sort()
        components.append(idxs)

    components.sort(key=lambda idxs: idxs[0])
    return components


def reorder_kernel_traces_multi_stream(trace_list: List[KernelTrace]) -> List[KernelTrace]:
    """刷新 ``index``，并对多 stream 并行块做局部重排。

    先按全局时间序 ``(start_time, end_time, stream_id)`` 排序；再找出**跨 stream 有时间重叠**的连通分量。
    对每个多 stream 分量，仅在该局部块内按 ``(stream_id, start_time, end_time)`` 重排；块外 kernel 保持时间序。

    这样普通单流片段仍连续，只有真正并行的那一小段会按 stream 聚类，不会把别的无关 kernel 也卷进来。
    """
    if len(trace_list) <= 1:
        for i, k in enumerate(trace_list):
            k.index = i
        return trace_list

    time_sort_key = lambda k: (k.start_time, k.end_time, _stream_id_sort_key(k.stream_id))
    trace_list.sort(key=time_sort_key)

    # 基于原始 ts 排序后的整条 trace 固定 stream 顺序，避免每个并行块局部重算导致层内顺序漂移。
    global_stream_order: List[Any] = []
    seen_stream_ids = set()
    for k in trace_list:
        if k.stream_id in seen_stream_ids:
            continue
        seen_stream_ids.add(k.stream_id)
        global_stream_order.append(k.stream_id)
    global_stream_rank = {sid: rank for rank, sid in enumerate(global_stream_order)}

    components = _build_multi_stream_components(trace_list)
    if not components:
        for i, k in enumerate(trace_list):
            k.index = i
        return trace_list

    component_by_index: Dict[int, List[int]] = {}
    component_meta: Dict[int, Dict[str, Any]] = {}
    for idxs in components:
        first_idx = idxs[0]
        for idx in idxs:
            component_by_index[idx] = idxs

        kernels = [trace_list[idx] for idx in idxs]
        block = sorted(
            kernels,
            key=lambda k: (
                global_stream_rank[k.stream_id],
                k.start_time,
                k.end_time,
            ),
        )
        component_meta[first_idx] = {"block": block}

    final: List[KernelTrace] = []
    emitted_component_starts = set()
    for idx, kernel in enumerate(trace_list):
        idxs = component_by_index.get(idx)
        if idxs is None:
            final.append(kernel)
            continue

        first_idx = idxs[0]
        if first_idx in emitted_component_starts:
            continue
        emitted_component_starts.add(first_idx)
        final.extend(component_meta[first_idx]["block"])

    for i, k in enumerate(final):
        k.index = i

    return final


# xpu_gpt NPU profiling：kernel 名归一化（与 summary_npu / trace 化简共用）
XPU_HCCL_KERNEL_SET = frozenset(
    {
        "hcom_allGather",
        "hcom_alltoall",
        "hcom_reduceScatter",
        "hcom_allReduce",
    }
)


def normalize_xpu_gpt_kernel_name(kernel_name: str, *, is_graph: bool, is_sk: bool) -> str:
    if not is_graph and not is_sk:
        if kernel_name.startswith("hcom_"):
            kernel_name_split = kernel_name.split("__")
            if len(kernel_name_split) > 1 and kernel_name_split[0] in XPU_HCCL_KERNEL_SET:
                index_str_split = kernel_name_split[1].split("_")
                return f"{kernel_name_split[0]}_{index_str_split[0]}"
        return kernel_name

    if is_graph and not is_sk:
        kernel_name_splits = kernel_name.split("_")
        if len(kernel_name_splits) <= 1:
            return kernel_name

        try:
            int(kernel_name_splits[-1])
            kernel_name = "_".join(kernel_name_splits[:-1])
        except ValueError:
            pass

        try:
            int(kernel_name_splits[1])
            return "_".join(kernel_name_splits[:1] + kernel_name_splits[2:])
        except ValueError:
            return kernel_name_splits[0]

    if is_graph and is_sk and kernel_name.startswith("sk_sk_layer_"):
        return "sk_layer"

    return kernel_name



def valid_dir(dir_path_str: str) -> pathlib.Path:
    if dir_path_str is None:
        return None

    dir_path = pathlib.Path(dir_path_str).absolute()
    if not dir_path.exists():
        raise FileNotFoundError(f"Dir {dir_path} not found!")
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Dir {dir_path} is not a dir!")
    return dir_path


def valid_file(model_path_str: str, required_suffix: str) -> pathlib.Path:
    if model_path_str is None:
        return None

    model_path = pathlib.Path(model_path_str).absolute()
    if not model_path.exists():
        raise FileNotFoundError(f"File {model_path} not found!")
    if not model_path.is_file():
        raise FileNotFoundError(f"File {model_path} is not a file!")
    if not model_path.suffix == required_suffix:
        raise ValueError(f"File {model_path} is not a json file!")
    return model_path



def load_dir_as_module(dir_path, module_name):
    """
    将指定文件夹动态加载为 Python 模块
    :param dir_path: 文件夹的路径 (字符串或 Path 对象)
    :param module_name: 你希望在 Python 中调用该模块的名字
    """
    # 1. 统一转为绝对路径，避免相对路径带来的困扰
    folder = pathlib.Path(dir_path).resolve()
    init_file = folder / "__init__.py"
    
    if not init_file.exists():
        raise FileNotFoundError(f"在 {folder} 中找不到 __init__.py，无法作为包加载")

    # 2. 告诉 Python 这个模块的物理位置和加载方式 (Spec)
    spec = importlib.util.spec_from_file_location(module_name, str(init_file))
    
    # 3. 创建一个新的模块对象
    module = importlib.util.module_from_spec(spec)
    
    # 4. 【关键步骤】将模块注入缓存，确保模块内部的相对导入（.import）能正常工作
    sys.modules[module_name] = module
    
    # 5. 执行模块内容（如果不执行，模块只是个空壳）
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # 如果初始化失败，清理缓存防止后续引用报错
        del sys.modules[module_name]
        raise e

    return module




def get_func_from_file(file_path, func_name):
    file_path = pathlib.Path(file_path).resolve()
    module_name = file_path.stem  # 获取文件名作为模块名
    
    # 创建 Spec 和 Module
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    
    # 执行模块代码（必须执行，否则函数还没被定义）
    spec.loader.exec_module(module)
    
    # 获取并返回函数
    return getattr(module, func_name, None)


def parse_csv_file(csv_file_path: str) -> List[Dict]:
    test_cases = []
    with open(csv_file_path, "r") as f:
        reader = csv.DictReader(f)
        for data in reader:
            test_cases.append(data)
    return test_cases






@dataclass
class CommonModelConfig:
    # 为了应对像Deepseek-v3.1这样的模型, 前3层是dense层，后58层是moe层
    # 仅考虑对应的层数, 暂时不用考虑实际的顺序
    num_layers: List[int] = field(default_factory=list)

    # 为了应对 prefill 阶段不处理某些层且不吐字的情况
    num_mirror_layers: int = 0





def get_unique_id():
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    combined = dt_str + random_str
    hash_obj = hashlib.md5(combined.encode())  # 使用MD5哈希
    hash_str = hash_obj.hexdigest()[:8]  # 取前8位哈希值
    return f"{dt_str}_{hash_str}"


def print_server_info(info_dict):
    # common info
    common_pt = prettytable.PrettyTable()
    common_pt.field_names = ["attr", "value"]
    common_pt.align = "l"
    for attr, value in info_dict["common"].items():
        if attr == "numa_configs":
            continue
        else:
            common_pt.add_row([attr, value])    
    print(common_pt)

    # provider info
    provider_pt = prettytable.PrettyTable()
    provider_pt.field_names = ["provider", "version"]
    provider_pt.align = "l"
    for provider, version in info_dict["provider"].items():
        provider_pt.add_row([provider, version])
    print(provider_pt)

    # backend info
    backend_pt = prettytable.PrettyTable()
    backend_pt.field_names = ["attr", "value"]
    backend_pt.align = "l"
    for attr, value in info_dict["backend"].items():
        backend_pt.add_row([attr, value])
    print(backend_pt)

    # runtime info
    runtime_pt = prettytable.PrettyTable()
    runtime_pt.field_names = ["attr", "value"]
    runtime_pt.align = "l"
    for attr, value in info_dict["runtime"].items():
        runtime_pt.add_row([attr, value])
    print(runtime_pt)

    print("\n\n\n")




