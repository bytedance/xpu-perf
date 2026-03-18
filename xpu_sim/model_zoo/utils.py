from typing import List
from dataclasses import dataclass, field

@dataclass
class CommonModelConfig:
    # 为了应对像Deepseek-v3.1这样的模型, 前3层是dense层，后58层是moe层
    # 仅考虑对应的层数, 暂时不用考虑实际的顺序
    num_layers: List[int] = field(default_factory=list)

    # 为了应对 prefill 阶段不处理某些层且不吐字的情况
    num_mirror_layers: int = 0





from typing import Dict, Any


"""
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
    

    def get_dist_info_str(self):
        parallel_config_str = f"{self.device_num}D"
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



