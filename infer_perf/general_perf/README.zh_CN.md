<div align="center">
  <img src="../../docs/images/icon.png">
</div>


# Byte MLPerf Inference Benchmark Tool
Byte MLPerf（推理）是字节使用的一个基准套件，用于测量推理系统在各种部署场景中运行模型的速度。相比MLPerf，Byte MLPerf有如下特点：
- 模型和运行环境会更贴近真实业务；
- 对于新硬件，除了评估性能和精度之外，同时也会评估图编译的易用性、覆盖率等指标；
- 在开放Model Zoo上测试所得的性能和精度，会作为新硬件引入评估的参考；

厂商可以参考该文档接入测试：[ByteMLPerf厂商接入指南](https://bytedance.feishu.cn/docs/doccno9eLS3OseTA5aMBeeQf2cf) [[English Version](https://bytedance.us.feishu.cn/docx/L98Mdw3J6obMtJxeRBzuHeRbsof)]

## Usage
用户使用入口为launch.py, 在使用byte mlperf评估时，只需传入--task 、--hardware_type 两个参数，如下所示：
```bash
python3 launch.py --task xxx --hardware_type xxx
```

1. tasks
--task 参数为传入的workload 名字，需要指定评估workload，例如：若要评估 open_bert-tf-fp16.json 定义的 workload，则需指定   --task open_bert-tf-fp16 。
注：所有workload定义在general_perf/workloads下，传参时名字需要和文件名对齐。目前格式为model-framework-precision。

2. hardware_type
--hardware_type 参数为传入的hardware_type 名字，无默认值，必须用户指定。例如：若要评估 Habana Goya ，则需指定   --hardware_type GOYA 。
注：所有hardware type定义在general_perf/backends下，传参时名字需要和folder名对齐。

3. compile_only
--compile_only 参数将在模型编译完成后停止任务

4. show_task_list
--show_task_list 参数会打印所有任务名字

5. show_hardware_list
--show_hardware_list 参数会打印目前所有支持的硬件Backend名称

### Workload说明
一个workload定义需包含如下字段:
```javascript
{
    "model": "bert-torch-fp32",   //待评估模型的名字，需要和model_zoo名字对齐
    "test_perf": true,            //是否评估模型性能
    "test_accuracy": true,        //是否评估模型精度
    "test_numeric": true,         //精度：是否评估数值误差
    "clients": 3,                 //性能：提交数据的client threads
    "iterations": 100,            //性能：每个thread提交多少iteration
    "batch_sizes":[1,4,8,16,32],  //性能：每个thread提交数据时的bs
    "data_percent": 50,           //精度：使用百分多少数据集评估精度, [1-100]
    "compile_only": false,        //是否仅编译模型
}
```

## Model Zoo List
Model Zoo&Dataset
Model Zoo下收录了Byte MlPerf支持的模型，从访问权限上，目前分为内部模型、开放模型。随Byte MlPerf 发布的是对应版本收录的开放模型。
Dataset为模型需要用到数据集，对应的dataloader、accuracy_checker从结构上也归入Dataset。

开放模型收录原则：
- 基础模型：包含十分常见的Rn50、Bert和WnD；
- 业务类似：包含目前内部较多的、或结构相似的模型结构；
- 前沿模型：包含业务领域对应的SOTA模型；

此外，除了完整模型结构，Byte MlPerf还会加入一些典型模型子结构子图或OP（前提是开放模型无法找到合适的完整模型包含这类经典子结构），比如各不同序列长度的transformer encoder/decoder，各类常见conv op，如group conv、depwise-conv、point-wise conv，以及rnn 常见结构，如gru/lstm等。

| Model | Domain | Purpose | Framework | Dataset | Precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | cv | regular | tensorflow, pytorch | imagenet2012 | fp32 |
| bert-base | nlp | regular | tensorflow, pytorch | squad-1.1 | fp32 |
| wide&deep | rec | regular | tensorflow | criteo | fp32 |
| videobert | mm  |popular | onnx | cifar100 | fp32 |
| albert | nlp | popular | pytorch | squad-1.1 | fp32 |
| conformer | nlp | popular | onnx | none | fp32 |
| roformer | nlp | popular | tensorflow | cail2019 | fp32 |
| yolov5 | cv | popular | onnx | none | fp32 |
| roberta | nlp | popular | pytorch | squad-1.1 | fp32 |
| deberta | nlp | popular | pytorch | squad-1.1 | fp32 |
| swin-transformer | cv | popular | pytorch | imagenet2012 | fp32 |
| stable diffusion | cv | sota | onnx | none | fp32 |

### ByteIR

ByteIR项目是字节跳动的模型编译解决方案。ByteIR包括编译器、运行时和前端，并提供端到端的模型编译解决方案。 尽管所有的ByteIR组件（编译器/runtime/前端）一起提供端到端的解决方案，并且都在同一个代码库下，但每个组件在技术上都可以独立运行。

更多信息请查看[ByteIR](https://github.com/bytedance/byteir)

ByteIR 编译支持的模型列表:
| Model | Domain | Purpose | Framework | Dataset | Precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | cv | regular | [mhlo](https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/resnet50_mhlo.tar) | imagenet2012 | fp32 |
| bert-base | nlp | regular | [mhlo](https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/bert_mhlo.tar) | squad-1.1 | fp32 |


## Vendor List
目前支持的厂商Backend如下:

| Vendor |  SKU | Key Parameters | Supplement |
| :---- | :----| :---- | :---- |
| Intel | Xeon | - | - |
| Stream Computing | STC P920 | <li>Computation Power:128 TFLOPS@FP16 <li> Last Level Buffer: 8MB, 256GB/s <li>Level 1 Buffer: 1.25MB, 512GB/s   <li> Memory: 16GB, 119.4GB/S <li> Host Interface：PCIe 4, 16x, 32GB/s <li> TDP: 160W | [STC Introduction](byte_infer_perf/general_perf/backends/STC/README.md) |
| Graphcore | Graphcore® C600 | <li>Compute: 280 TFLOPS@FP16, 560 TFLOPS@FP8 <li> In Processor Memory: 900 MB, 52 TB/s <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 185W | [IPU Introduction](byte_infer_perf/general_perf/backends/IPU/README.zh_CN.md) |
| Moffett-AI | Moffett-AI S30 | <li>Compute: 1440 (32x-Sparse) TFLOPS@BF16, 2880 (32x-Sparse) TOPS@INT8, <li> Memory: 60 GB,  <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 250W                           | [SPU Introduction](byte_infer_perf/general_perf/backends/SPU/README.md) |
| Habana | Gaudi2 | <li>24 Tensor Processor Cores, Dual matrix multiplication engines <li> Memory: 96 GB HBM2E, 48MB SRAM                            | [HPU Introduction](byte_infer_perf/general_perf/backends/HPU/README.md) |
