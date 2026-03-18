# Byte LLM Perf
## Requirements
* Python >= 3.8
* torch >= 2.1.0

## Installation
```shell
# modify according to torch version and hardware
pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# install required packages
pip3 install -r requirements.txt
```

## Quick Start (run accuracy and performance tests)
Please be sure to complete the installation steps before proceeding with the following steps: 
1. Modify task workload, for example, [chatglm2-torch-fp16-6b.json](https://github.com/bytedance/ByteMLPerf/blob/main/byte_infer_perf/llm_perf/workloads/chatglm2-torch-fp16-6b.json)
2. Download model weights using prepare_model.sh or huggingface_cli.
3. Download model output logits in specific input case(.npy files) using prepare_model.sh.
4. Start accuracy and performance tests.

You can run following command automate all steps with chatglm2 model on GPU backend
```shell
python3 byte_infer_perf/llm_perf/launch.py --hardware_type GPU --task chatglm2-torch-fp16-6b 
```

## Split model
Splitting model is needed if model is too large to fit into one GPU. Except for chatglm2-6b, other models should be splitted manually using `split_model.py` under `backends/GPU/model_impl`, such as `split_mixtral.py`. `chatglm2-6b` will be automatically splitted online.

After splitting model, you will find a subdirectory `TP8` (tp_size=8) under model directory.


## Test accuracy (single query with specify prompt)
Launch a server running mixtral-8x22b (tp_size=8, max_batch_size=8) with following command:
```shell
cd byte_infer_perf/llm_perf
python3 ./server/launch_server.py --hardware_type GPU --model_config ./model_zoo/mixtral-torch-bf16-8x22b.json --tp_size 8 --max_batch_size 8
```

Test server with single prompt, and you can get infer result, logits numpy file and model forward time. Output files will locate in `./reports/single_query/`
```shell
python3 ./script/single_query.py --prompt "What is 7 multiplied by 7?" --batch_size 8
```

## Test model_impl model forward performance
Only need to instantiate MpEngine running mixtral-8x22b (tp_size=8, max_batch_size=8) and feed proper inputs. Runing following command will get performance outputs. You can modify test cases in `./bench_model.py` currerntly.
```shell
python3 ./bench_model.py --hardware_type GPU --model_config ./model_zoo/mixtral-torch-bf16-8x22b.json --tp_size 8 --max_batch_size 8
```

The output will located in `./reports/{hardware_type}/{model_config}/bench_model`:
- **config.json**: perf config
- **context_perf.csv**: prefill, latency with specified {batch_size, seq_len}
- **decode_perf.csv**: decode, latency with specified {batch_size, seq_len}
- **output.txt**: raw latency data


## Demo Project
[GPU Backend](https://github.com/bytedance/ByteMLPerf/tree/main/byte_infer_perf/llm_perf/backends/GPU) provides a demo project that realizes llm inference of chatglm2-6b on A100 with following features: 
- Separate functional components:
    * Scheduler 
        - custom scheduling on tasks
    * Inferencer
        - transfer tasks to real inputs and get outputs
    * Mp Engine
        - deal with TP logic using multiple processes
    * Sampler
        - postprocess logic
    * Ckpt Loader
        - custom ckpt loader with split logic which matches TP logic.
    * Custom model implementation
        - custom model implementation using hardware backend torch realization
- Seperate scheduling logic
    * Context: one task, input_ids shape is [1, q_len]
    * Decode: multiple tasks, input_ids shape up to [max_batch_size, 1]
- Tensor parallelism
- kv cache

The demo project is intended to provide a reference implementation, and there's no guarantee of achieving optimal performance. More technical details will be provided later on [ByteMLPerf](https://bytemlperf.ai)


## Vendor Integration
Vendors can refer to this document for guidance on building backend: [Byte LLM Perf](https://bytemlperf.ai/zh/guide/inference_llm_vendor.html)

## Models
The following models are planned to be supported:
* [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
* [shenzhi-wang/Llama3-70B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3-70B-Chinese-Chat)
* [tiiuae/falcon-180B](https://huggingface.co/tiiuae/falcon-180B)
    - test_accuracy is unavailable temporarily.
* [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
    - test_accuracy is unavailable temporarily.
