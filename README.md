# xpu-perf

## 项目目标

最终目标是打通`芯片厂商 --> ai infra engine层 --> ai infra serving层 --> 业务层`由下到上的整条技术链路，制定更加合理、专业的评估方法论和评估工具，消除层间的信息隔离，优化价值判断以找到最核心的指标需求，最终实现`以最低的TCO成本提供更高的业务服务性能`。

为达成上述目标，我们拆解了以下需求：

1. 深入芯片底层架构研究、机型研究、互联拓扑研究，探索未来架构设计方向。

2. 提供芯片本身评测能力，包括精度、指令吞吐、SDC等领域。

3. 定义基础算子、特定应用场景（llm、dit等）专用算子，并提供高效的算子测试框架，用于评估衡量特定芯片、特定软件栈、特定算子库的综合性能表现，得到每个核心算子的MFU（算力利用率）、MBU（内存带宽利用率）、CBU（通信带宽利用率）。

4. 基于算子测试框架提供特定应用场景的性能仿真工具（llm、dit等），快速得到接近实际部署的性能数据，比如在分布式LLM模型推理中PD分离性能。

5. 提供特定应用场景（llm推理、llm训练、dit推理等）的实测要求，衡量具体部署下的实际性能表现，比如特性芯片、特定模型、特定部署形态（并行方式+精度）、特定框架（比如vllm、sglang）上的prefill/decode性能。

6. 提供业务视角的性能指标，提供具体业务场景的trace_gen能力，以评估具体部署形态下在特定业务场景下的实际性能表现。


## 子项目

### [micro_perf](./micro_perf/)
算子测试框架。

### [xpu_sim](./xpu_sim/)
基于算子测试框架的模型端到端性能和breakdown性能仿真工具。

### [trace_gen](./trace_gen/)
独立的llm模型推理的请求生成工具。

### [(old) infer_perf](./infer_perf/)
原有的小模型、llm模型测试框架，已过时，正在调整中，未来将聚焦成熟推理框架（vllm、sglang）的bench能力。

### [train_perf](./train_perf)
llm模型训练评估工程。



## 引用
**If you use this code or find our work valuable, please cite:**
```bibtex
@inproceedings{cai2026characterizing,
  title={Characterizing Cloud-Native LLM Inference at Bytedance and Exposing Optimization Challenges and Opportunities for Future AI Accelerators},
  author={Cai, Jingwei and Kong, Dehao and Huang, Hantao and Jiang, Zishan and Ma, Zixuan and Guo, Qingyu and Zhang, Zhenxing and Shi, Guiming and Gao, Mingyu and Ma, Kaisheng and others},
  booktitle={2026 IEEE International Symposium on High Performance Computer Architecture (HPCA)},
  pages={1--19},
  year={2026},
  organization={IEEE}
}
```
