<div align="center">
  <img src="Graphcore-Chinese-Wordmark-Horizontal.svg">
</div>

[ [English](README.md) ]

# Graphcore® C600

C600 是 Graphcore 为云和数据中心打造的高端推训一体加速卡，主打推理，兼做训练，可以支持各种主流的 AI 应用，在搜索和推荐等业务上别具优势。C600 在提供低延时、高吞吐量的同时不损失精度，帮助 AI 开发人员解决”精度与速度难两全”的痛点，为 AI 应用提供解锁 IPU 强大算力的新路径，以满足客户和机器智能从业者对于易用、高效以及更优 TCO 推理产品的强烈需求。

C600 是一张 PCIe Gen 4 双插槽卡，使用一个 IPU，每个 IPU 具有 1472 个处理核心，能够并行运行 8832 个独立程序线程。每个 IPU 都有 900MB 的片上 SRAM 存储。用户可以在单个机箱中直接连接多达 8 块卡，通过高带宽的 IPU-Links 进行桥接。C600 可搭配市场上主流的 AI 服务器使用，比如浪潮信息 NF5468M6 等。

## 产品规格

| 规格 | 说明 |
| :-----| :-----|
| **IPU 处理器** | 支持 FP8 的 Graphcore® MK2 IPU 处理器 |
| **IPU 核心**	| 1472 个 IPU 核心，每个核心都是一个高性能处理器，支持多线程和独立代码执行 |
| **处理器内存储** | 每个 IPU 核心都配有快速且紧密耦合的本地处理器内存储 <br> C600加速器包括 900MB 的处理器内存储 |
| **计算** | 高达 560 teraFLOPS 的 FP8 计算 <br> 高达 280 teraFLOPS 的 FP16 计算 |
| **系统接口** | 2 个分叉 16 位 PCIe 接口的 8 路端口 |
| **散热方案** | 被动散热 |
| **外形** | PCIe 全高/全长；双插槽 |
| **尺寸** | 长度：267 毫米（10.5 英寸）<br> 高度：111 毫米（4.37 英寸）<br> 宽度：27.6 毫米（1.09 英寸）|
| **重量** | 1.27 千克（2.8 磅) |
| **IPU-Link™ 支持** | 64 路，256GB/s 的双 IPU-Links |
| **电源** | 185 瓦 |
| **辅助电源** | 8 针 |
| **质量级别** | 服务器级别 |

关于 Graphcore® C600 的更多信息，请访问 [Graphcore 中文网站](https://www.graphcore.cn/c600-pcie%e5%8d%a1/)。

# Graphcore® PopRT
PopRT 是一个针对 IPU 处理器的高性能推理引擎，负责把训练完导出的模型，针对推理进行深度编译优化，生成能在 IPU 上运行的可执行程序 PopEF，并提供灵活的 Runtime，实现对 PopEF 进行低延时，高吞吐的推理。

PopRT 提供了易于集成的 Python 和 C++ API，ByteMLPerf 模型在 IPU 上的运行即通过 PopRT Python API 进行模型的优化，编译和运行。

更多关于 PopRT 的资料，请访问 [PopRT 用户指南](https://graphcore.github.io/PopRT/1.4.0/)。

获取 PopRT 的 Docker 镜像，请访问 [graphcorecn/poprt](https://hub.docker.com/r/graphcorecn/poprt)。

# 支持的模型

| Model name |  Precision | QPS | Dataset | Metric name | Metric value | report |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| albert-torch-fp32 | FP16 | 3,280 | Open Squad 1.1 | F1 Score | 87.69675 | [report](../../reports/IPU/albert-torch-fp32/) |
| bert-torch-fp32 | FP8 | 4,464 | Open Squad 1.1 | F1 Score | 85.71465 | [report](../../reports/IPU/bert-torch-fp32/) |
| bert-torch-fp32 | FP16 | 3,134 | Open Squad 1.1 | F1 Score | 85.85797 | [report](../../reports/IPU/bert-torch-fp32/) |
| clip-onnx-fp32 | FP16 | 7,305 | Fake Dataset | Mean Diff | 0.00426 | [report](../../reports/IPU/clip-onnx-fp32/) |
| conformer-encoder-onnx-fp32 | FP16 | 9,341 | Fake Dataset | Mean Diff | 0.00161 | [report](../../reports/IPU/conformer-encoder-onnx-fp32/) |
| deberta-torch-fp32 | FP16 | 1,702 | Open Squad 1.1 | F1 Score | 81.24629 | [report](../../reports/IPU/deberta-torch-fp32/) |
| resnet50-torch-fp32 | FP8 | 18,851 | Open Imagenet | Top-1 | 0.76824 | [report](../../reports/IPU/resnet50-torch-fp32/) |
| resnet50-torch-fp32 | FP16 | 13,499 | Open Imagenet | Top-1 | 0.76963 | [report](../../reports/IPU/resnet50-torch-fp32/) |
| roberta-torch-fp32 | FP16 | 3,088 | Open Squad 1.1 | F1 Score | 83.1606 | [report](../../reports/IPU/roberta-torch-fp32/) |
| roformer-tf-fp32 | FP16 | 2,520 | OPEN_CAIL2019 | Top-1 | 0.64323 | [report](../../reports/IPU/roformer-tf-fp32/) |
| swin-large-torch-fp32 | FP8 | 480 | Open Imagenet | Top-1 | 0.8552 | [report](../../reports/IPU/swin-large-torch-fp32/) |
| swin-large-torch-fp32 | FP16 | 315 | Open Imagenet | Top-1 | 0.8536 | [report](../../reports/IPU/swin-large-torch-fp32/) |
| videobert-onnx-fp32 | FP16 | 3,691 | OPEN_CIFAR | Top-1 | 0.6169 | [report](../../reports/IPU/videobert-onnx-fp32/) |
| widedeep-tf-fp32 | FP16 | 31,446,195 | Open Criteo Kaggle | Top-1 | 0.77392 | [report](../../reports/IPU/widedeep-tf-fp32/) |

# 如何运行

## 下载并安装 Poplar SDK

```
wget -O 'poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz' 'https://downloads.graphcore.ai/direct?package=poplar-poplar_sdk_ubuntu_20_04_3.3.0_208993bbb7-3.3.0&file=poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz'

tar xzf poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz

source poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/enable
```

## 启动 PopRT Docker 容器

```
docker pull graphcorecn/poprt:1.4.0

gc-docker -- -it \
              -v `pwd -P`:/workspace \
              -w /workspace \
              --entrypoint /bin/bash \
              graphcorecn/poprt:1.4.0
```

## 安装 ByteMLPerf 的依赖

```
apt-get update && \
apt-get install wget libglib2.0-0 -y
```

## 运行 ByteMLPerf 的任务

使用如下命令运行：

```
python3 launch.py --task widedeep-tf-fp32 --hardware IPU
```

更多关于 ByteMLPerf 运行命令的说明，请参考 [ByteMLPerf](../../../README.zh_CN.md#usage)。

