<div align="center">
  <img src="STC.jpg">
</div>


# Supported model inference results
| Model name | QPS | Dataset | Metric name | Metric value |
| :-----:| :----: | :----: | :----: | :----: |
| albert-torch-fp32 | 824.49 | Open Squad 1.1 | F1 Score | 87.66 |
| bert-tf-fp32 | 822.38 | Open Squad 1.1 | F1 Score | 86.45 |
| bert-torch-fp32 | 813.86 | Open Squad 1.1 | F1 Score | 86.14 |
| resnet50-tf-fp32 | 8725.94 | Open ImageNet | Top-1 | 77.24% |
| robert-torch-fp32 | 800.7 | Open Squad 1.1 | F1 Score | 83.19 |
| widedeep-tf-fp32 | 2395899.9 | Open Criteo Kaggle | Top-1 | 77.39% |


For more detailed result information, see general_perf/reports/STC/. Models above are depolyed on a NPU (Neural-network Processing Unit) card "STCP920" which is designed and manufactured by Beijing Stream Computing Technology Co., LTD. Softwares associated with STCP920 are as following: 

| Software | Version | Description |B
| :-----:| :----: | :----: |
| HPE | 1.5.1 | Heterogeneous Programming Environment |
| TensorTurbo | 1.11.0 | An AI compiler for STCP920 developed based on TVM |
| STC_DDk | 1.1.0 | Deploy Development Kits for STCP920, which includes AI Convertor, AI Executor, and utilities used in model conversion. |


In addition, a variety of tools for monitoring status of NPU devices, debugging heterogeneous programs, and analyzing accuracy and performance of NPU programs are provieded.

| Software  | Description |
| :-----:| :----: |
| stc-smi | Stream Computing System Management Interface for managing and monitoring NPU devices, including viewing device information and resource usage |
| stc-gdb | Stream Computing Debugger for debugging heterogeneous NPU programs  |
| stc-prof | Stream Computing Profiler, for performance analysis and optimization of heterogeneous programs  |
| stc-hpaa | Stream Computing Half-Precision Accuracy Analysis, for locating the calculation error location and corresponding data  |


For more detailed software information, please refer to: https://docs.streamcomputing.com/_/sharing/vSxLMI20nalGphdpXdEVoDg6JkUcfEkT?next=/zh/latest/

# How to run
1. Prepare environment  
Prepare a machine with the STCP920 chip, install HPE, install -r general_perf/requirements.txt. Then create a virtual environment, install -r general_perf/backends/STC/requirements.txt, install Tensorturbo and STC_DDK. These installation packages can be obtained by visiting this link: https://docs.streamcomputing.com/_/sharing/vSxLMI20nalGphdpXdEVoDg6JkUcfEkT?next=/zh/latest/

```bash
export PYTHONPATH=$PYTHONPATH:ByteMLPerf:ByteMLPerf/general_perf/backends/STC
```

2. Prepare model and dataset  
Run general_perf/prepare_model_and_dataset.sh to get model and dataset.

3. Run 
```bash
python3 launch.py --tasks xxx --hardware_type STC  
```
--task parameter is the name of the incoming workload. You need to specify the workload. For example, if you would like to evaluate the workload: bert-tf-fp16.json, you need to specify --task bert-tf-fp16.


# Company introduction
Beijing Stream Computing Technology Co., LTD, is committed to providing cloud service manufacturers with high cost performance and high versatility of AI accelerated chips.

The first-generation chip achieves 128 TFLOPS in semi-precision floating-point operations, twice as big as T4. At present, the first-generation NPU card 'STCP920' is in mass production, and has completed a batch of shipments to users. The second-generation products are in schedule and will be coming soon in 2023.

# The technical specifications of the first-generation chip
| Name  | Value |
| :-----:| :----: |
| AI Computation power | 128 TFLOPS @ FP16 |
| Memory Type | LPDDR4X |
| Memory | 16GB, 119.4GB/S |
| Last Level Buffer | 8MB, 256GB/s |
| Level 1 Buffer | 1.25MB, 512GB/s |
| Host Interface | PCIe 4, 16x, 32GB/s, support Lane Reversal |
| Thermal Design Power | 160W |
| Structural Dimension | 268.44mm x 111.15mm, single slot |

# What we have done
We provide development kits to support converting any deep learning model into an stc engine deploying it on a CPU+NPU server.

An AI compiler(TensorTurbo) is developed to convert certain part of a deep learning model into an NPU-executable file. The AI compiler employs a series of transformations and optimizations in the process of model conversion, to ensure better inference performance of the outcome.

Using the associated softwares, we have supported over 150 open source models from four deep learning frameworks including tensorflow 1.x and 2.x, pytorch, onnx, paddlepaddle. The application fields include CV, NLP, recommendation, speech, OCR, multimodel. Most of the models achieve 2x inference performance compared to Nvidia GPU T4.


# Contact us
If you are interested in further information about the product, please contact the email: johnson@streamcomputing.com

