# TensorRT-LLM Benchmark


## installation
Refer to [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) for details.

Follwing envs have been tested:
- Docker image: nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu20.04
- TensorRT-LLM: 0.13.0
- TensorRT: 10.4.0.26

## build engine and test
mixtral-8x22b, tp_size=8, moe_tp_size=8, dtype=float16
```
cd TensorRT-LLM/examples/mixtral

// convert model
python3 ../llama/convert_checkpoint.py \
    --model_dir ./mixtral-8x22b \
    --output_dir ./tllm_checkpoint_mixtral_8gpu \
    --dtype float16 \
    --tp_size 8

// build engine
trtllm-build \
    --checkpoint_dir ./tllm_checkpoint_mixtral_8gpu \
    --output_dir ./trt_engines/mixtral/tp8 \
    --max_batch_size 256 \
    --max_input_len 17408 \ 
    --max_seq_len 17408 \
    --max_num_tokens 17408

// run engine with given prompt
mpirun --allow-run-as-root -n 8 python3 ../run.py \
    --engine_dir ./trt_engines/mixtral/tp8/ \
    --tokenizer_dir ./mixtral-8x22b \
    --max_output_len 100 \
    --input_text "7 years ago, I was 6 times older than my son. My son is 12 years old now. How old am I now?"

// benchmark engine
python3 bench_engine.py \
    --engine_dir ./trt_engines/mixtral/tp8/ \
    --model_dir ./mixtral-8x22b \
    --batch_size_list 1,2,4,8,16,32,40,48,56,64,72,80,88,96,104,112,120,128 \
    --seq_len_list 1024,2048,4096,6144,8192
```




