<div align="center">
  <img src="Graphcore-Chinese-Wordmark-Horizontal.svg">
</div>

[ [中文](README.zh_CN.md) ]

# Graphcore® C600

The Graphcore® C600 IPU-Processor PCIe Card is a high-performance acceleration server card targeted for machine learning inference and training. Powered by the Graphcore Mk2 IPU Processor with FP8 support, the C600 is a dual-slot, full height PCI Express Gen4 card designed for mounting in industry standard server chassis to accelerate machine intelligence workloads.

Up to eight C600 IPU-Processor PCIe Cards can be networked together using IPU-Link™ high-bandwidth interconnect cables, delivering enhanced IPU compute capability.

## Product Specs

| Name | Description |
| :-----| :-----|
| IPU Processor | Graphcore Mk2 IPU Processor with FP8 support |
| IPU-Cores™ | 1,472 IPU-Cores, each one a high-performance processor capable of multi-thread, independent code execution |
| In-Processor Memory™ | Each IPU-Core is paired with fast, local, tightly-coupled In-Processor Memory. The C600 accelerator includes 900MB of In-Processor Memory |
| Compute | Up to 560 teraFLOPS of FP8 compute <br> Up to 280 teraFLOPS of FP16 compute <br> Up to 70 teraFLOPS of FP32 compute |
| System Interface | Dual PCIe Gen4 8-lane interfaces |
| Thermal Solution | Passive |
| Form Factor | PCIe full-height/length; double-slot |
| System Dimensions |	Length: 267mm (10.50”); Height: 111mm (4.37”); Width: 27.6mm (1.09”); Mass: 1.27kg (2.8lbs) |
| IPU-Link™ | Support	32 lanes, 128 GB/s bandwidth (64 GB/s in each direction) IPU-Links |
| TDP |	185W |
| Auxiliary Power Supply | 8-pin |
| Quality Level | Server grade |

For more information of the Graphcore® C600, please refer to [C600 cards](https://docs.graphcore.ai/en/latest/hardware.html#c600-cards).

# PopRT

PopRT is a high-performance inference framework specifically for Graphcore IPUs. It is responsible for deeply optimizing the trained models, generating executable programs that can run on the Graphcore IPUs, and performing low-latency, high-throughput inference.

You can get PopRT and related documents from [graphcore/PopRT](https://graphcore.github.io/PopRT/1.4.0/).

Docker images are provided at [graphcorecn/poprt](https://hub.docker.com/r/graphcorecn/poprt).

# Models supported

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

# How to run

## Download and enable Poplar SDK

```
wget -O 'poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz' 'https://downloads.graphcore.ai/direct?package=poplar-poplar_sdk_ubuntu_20_04_3.3.0_208993bbb7-3.3.0&file=poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz'

tar xzf poplar_sdk-ubuntu_20_04-3.3.0-208993bbb7.tar.gz

source poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/enable
```

## Start PopRT docker container

```
docker pull graphcorecn/poprt:1.4.0

gc-docker -- -it \
              -v `pwd -P`:/workspace \
              -w /workspace \
              --entrypoint /bin/bash \
              graphcorecn/poprt:1.4.0
```

## Install dependencies in docker container

```
apt-get update && \
apt-get install wget libglib2.0-0 -y
```

## Run byte-mlperf task

For example,

```
python3 launch.py --task widedeep-tf-fp32 --hardware IPU
```

For more information of the command to run the task, please refer to [ByteMLPerf](../../../README.md#usage).
