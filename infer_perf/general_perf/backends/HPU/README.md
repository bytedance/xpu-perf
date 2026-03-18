<div align="center">
  <img src="habana-white_intel_logo.png">
</div>


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Habana](#habana)
  - [Product Specs](#product-specs)
- [Models supported](#models-supported)
- [How to run](#how-to-run)
    - [1. Create docker container](#1-create-docker-container)
    - [2. Environment initialization](#2-environment-initialization)
    - [3. Device basic information verification](#3-device-basic-information-verification)
    - [4.Run byte-mlperf task](#4run-byte-mlperf-task)

<!-- /code_chunk_output -->


# Habana

As enterprises and organizations look to seize the growing advantages of AI, the time has never been better for AI compute that’s faster yet efficient. Efficient on cost, power, and your time and resources. That’s why you’ll want to give Habana Gaudi processors a try.The Gaudi acceleration platform was conceived and architected to address training and inference demands of large-scale era AI, providing enterprises and organizations with high-performance, high-efficiency deep learning compute.

## Product Specs

- Gaudi

With Habana’s first-generation Gaudi deep learning processor, customers benefit from the most cost-effective, high-performance training and inference alternative to comparable GPUs. This is the deep learning architecture that enables AWS to deliver up to 40% better price/performance training with its Gaudi-based DL1 instances—as compared to comparable Nvidia GPU-based instances. Gaudi’s efficient architecture also enables Supermicro to provide customers with equally significant price performance advantage over GPU-based servers with the Supermicro X12 Gaudi Training Server.

<div align="center">
  <img src="gaudi.png">
</div>

- Gaudi2

Our Gaudi2 accelerator is driving improved deep learning price-performance
and operational efficiency for training and running state-of-the-art models, from the largest language and multi-modal models to more basic computer vision and NLP models. Designed for efficient scalability—whether in the cloud or in your data center, Gaudi2 brings the AI industry the choice it needs—now more than ever.

<div align="center">
  <img src="gaudi2.png">
</div>

# Models supported

| Model name |  Precision | QPS | Dataset | Metric name | Metric value | report |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| bert-torch-fp32 | BF16 | 1970 | Open Squad 1.1 | F1 Score | 85.8827 | [report](../../reports/HPU/bert-torch-fp32/) |
| albert-torch-fp32 | BF16 | 2030 | Open Squad 1.1 | F1 Score | 87.66915 | [report](../../reports/HPU/albert-torch-fp32/) |
| deberta-torch-fp32 | BF16 | 1970 | Open Squad 1.1 | F1 Score | 81.33603 | [report](../../reports/HPU/deberta-torch-fp32/) |
| resnet50-torch-fp32 | BF16 | 8279 | Open ImageNet | Top-1 | 0.7674 | [report](../../reports/HPU/resnet50-torch-fp32/) |
|  swin-large-torch-fp32 | BF16 |341 | Open ImageNet | Top-1 | 0.855 | [report](../../reports/HPU/swin-large-torch-fp32/) |

# How to run

### 1. Create docker container

```bash
docker run -itd --name test --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host   vault.habana.ai/gaudi-docker/1.12.0/ubuntu20.04/habanalabs/pytorch-installer-2.0.1:latest
```
### 2. Environment initialization
Environment initialization please operate in the container.
```bash=
docker exec -it test /bin/bash
```
### 3. Device basic information verification
hl-smi is a command line utility that can view various information of Gaudi, such as card number, usage, temperature, power consumption, etc.
After the driver is successfully installed, execute hl-smi to view the basic information of the device.
```bash
hl-smi
```

### 4.Run byte-mlperf task

For example,

```bash
python launch.py --task bert-torch-fp32 --hardware_type HPU
```

For more information of the command to run the task, please refer to [ByteMLPerf](../../../README.md#usage).
