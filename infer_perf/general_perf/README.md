<div align="center">
  <img src="../../docs/images/icon.png">
</div>


# Byte MLPerf Inference Benchmark Tool
Byte MLPerf(Inference) is an AI Accelerator Benchmark that focuses on evaluating AI Accelerators from practical production perspective, including the ease of use and versatility of software and hardware. Byte MLPerf has the following characteristics:
- Models and runtime environments are more closely aligned with practical business use cases.
- For ASIC hardware evaluation, besides evaluate performance and accuracy, it also measure metrics like compiler usability and coverage.
- Performance and accuracy results obtained from testing on the open Model Zoo serve as reference metrics for evaluating ASIC hardware integration.

Vendors can refer to this document for guidance on building backend: [ByteMLPerf Guide](https://bytedance.us.feishu.cn/docx/L98Mdw3J6obMtJxeRBzuHeRbsof) [[中文版](https://bytedance.feishu.cn/docs/doccno9eLS3OseTA5aMBeeQf2cf#TDK8of)]

## Usage
The user uses launch.py as the entry point. When using Byte MLPerf to evaluate the model, you only need to pass in two parameters --task and --hardware_type, as shown below:
```bash
python3 launch.py --task xxx --hardware_type xxx
```

1. task
--task parameter is the name of the incoming workload. You need to specify the workload. For example, if you would like to evaluate the workload: bert-tf-fp16.json, you need to specify --task bert-tf-fp16.
Note: All workloads are defined under general_perf/workloads, and the name needs to be aligned with the file name when passing parameters. The current format is model-framework-precision.

2. hardware_type
--hardware_type parameter is the incoming hardware_type name, there is no default value, it must be specified by the user. Example: To evaluate Habana Goya, specify --hardware_type GOYA .
Note: All hardware types are defined under general_perf/backends, and the name needs to be aligned with the folder name when passing parameters.

3. compile_only
--compile_only parameter will make task stoped once compilation is finished

4. show_task_list
--show_task_list parameter will print all task name

5. show_hardware_list
--show_hardware_list parameter will print all hardware backend

### Workload Description
A workload definition needs to contain the following fields:
```javascript
{
    "model": "bert-torch-fp32",   //The name of the model to be evaluated, which needs to be aligned with the model_zoo name
    "test_perf": true,            //Evaluate model performance
    "test_accuracy": true,        //Evaluate model accuracy
    "test_numeric": true,         //Accuracy：Evaluate model numeric
    "clients": 3,                 //Performance：Client threads that submit data
    "iterations": 100,            //Performance：How many iterations are submitted by each thread
    "batch_sizes":[1,4,8,16,32,64],//Performance：The batch size when each thread submits data
    "data_percent": 50,           //Accuracy：Ratio of data to assess accuracy, [1-100]
    "compile_only": false,           //Compile the model only
}
```

## Model Zoo List
Model Zoo&Dataset
The models supported by Byte MLPerf are collected under the Model Zoo. From the perspective of access rights, they are currently divided into internal models and open models. Released with Byte MLPerf is the open model included in the corresponding version.

Open model collection principles:
- Basic Model: including Resnet50, Bert and WnD;
- Popular Model：Includes models currently widely used in the industry;
- SOTA: including SOTA models corresponding to business domains;

In addition to the complete model structure, Byte MLPerf will also add some typical model substructure subgraphs or OPs (provided that the open model cannot find a suitable model containing such classic substructures), such as transformer encoder/decoder with different sequence lengths , all kinds of common conv ops, such as group conv, depwise-conv, point-wise conv, and rnn common structures, such as gru/lstm, etc.

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

The ByteIR Project is a ByteDance model compilation solution. ByteIR includes compiler, runtime, and frontends, and provides an end-to-end model compilation solution.

Although all ByteIR components (compiler/runtime/frontends) are together to provide an end-to-end solution, and all under the same umbrella of this repository, each component technically can perform independently.

For More Information, please refer to [ByteIR](https://github.com/bytedance/byteir)

Models Supported By ByteIR:
| Model | Domain | Purpose | Framework | Dataset | Precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | cv | regular | [mhlo](https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/resnet50_mhlo.tar) | imagenet2012 | fp32 |
| bert-base | nlp | regular | [mhlo](https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/bert_mhlo.tar) | squad-1.1 | fp32 |

## Vendor List
ByteMLPerf Vendor Backend List will be shown below

| Vendor |  SKU | Key Parameters | Supplement |
| :---- | :----| :---- | :---- |
| Intel | Xeon | - | - |
| Stream Computing | STC P920 | <li>Computation Power:128 TFLOPS@FP16 <li> Last Level Buffer: 8MB, 256GB/s <li>Level 1 Buffer: 1.25MB, 512GB/s   <li> Memory: 16GB, 119.4GB/S <li> Host Interface：PCIe 4, 16x, 32GB/s <li> TDP: 160W | [STC Introduction](general_perf/backends/STC/README.md) |
| Graphcore | Graphcore® C600 | <li>Compute: 280 TFLOPS@FP16, 560 TFLOPS@FP8 <li> In Processor Memory: 900 MB, 52 TB/s <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 185W | [IPU Introduction](general_perf/backends/IPU/README.md) |
| Moffett-AI | Moffett-AI S30 | <li>Compute: 1440 (32x-Sparse) TFLOPS@BF16, 2880 (32x-Sparse) TOPS@INT8, <li> Memory: 60 GB,  <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 250W                           | [SPU Introduction](general_perf/backends/SPU/README.md) |
| Habana | Gaudi2 | <li>24 Tensor Processor Cores, Dual matrix multiplication engines <li> Memory: 96 GB HBM2E, 48MB SRAM                            | [HPU Introduction](general_perf/backends/HPU/README.md) |

## Statement
[ASF Statement on Compliance with US Export Regulations and Entity List](https://news.apache.org/foundation/entry/statement-by-the-apache-software)
