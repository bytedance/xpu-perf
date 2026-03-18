#!/bin/bash

# Runs the "345M" parameter model
PRETRAIN_MODEL_PATH=/projs/platform/ckpt_345M/ckpt
CHECKPOINT_SAVE_PATH=/data02/huanghantao/Megatron-LM/examples/gpt3/ckpt_345M/ckpt
TENSORBOARD_LOGS_PATH=/data02/huanghantao/Megatron-LM/examples/gpt3/ckpt_345M/tensorboard
VOCAB_FILE=/projs/framework/jichen/aaa/ByteMLPerf/byte_train_perf/Megatron-LM_mlu/gpt2-vocab.json
MERGE_FILE=/projs/framework/jichen/aaa/ByteMLPerf/byte_train_perf/Megatron-LM_mlu/gpt2-merges.txt
DATA_PATH=/projs/platform/zhangshuanglin/datasets/gpt_test/my-gpt2_text_document
# OUTPUT path
LOG_ADDR_EXISTS=/projs/platform/a100_345M_variable_memory.log
MAX_TRAIN_STEPS=2000
OUTPUT_CMP_ADDR=./output/gpt3_345M_variable_memory_log
echo $LOG_ADDR_EXISTS
echo $MAX_TRAIN_STEPS
echo $OUTPUT_CMP_ADDR

# Check if there are any torch logs
if [ ! -f "$LOG_ADDR_EXISTS" ]; then
    echo "Error: The file $LOG_ADDR_EXISTS does not exist." >&2
    exit 1  # Exit with non-zero status code, indicating abnormal termination
else
    echo "The file $LOG_ADDR_EXISTS exists, continue execution..."
fi

# Check if the folder where the results are saved exists. If it does not exist, create it
if [ ! -d "$OUTPUT_CMP_ADDR" ]; then
    echo "The folder $OUTPUT_CMP_ADDR does not exist, creating it now..."
    mkdir "$OUTPUT_CMP_ADDR"
fi

# Define the log file path and create
LOG_FILE_PATH="${OUTPUT_CMP_ADDR}/mlu_345M_variable_memory.log"
# Check if there are any base logs
if [ ! -f "$LOG_FILE_PATH" ]; then
    echo "The file $LOG_FILE_PATH does not exist, creating it now..."
    touch $LOG_FILE_PATH
else
    echo "The file $LOG_FILE_PATH exists, continue execution..."
fi

# Runs the "345M" parameter model
bash examples/gpt3/train_gpt3_345M_distributed_variable_memory.sh $CHECKPOINT_SAVE_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH $PRETRAIN_MODEL_PATH $MAX_TRAIN_STEPS 2>&1 | tee $LOG_FILE_PATH
python3 memory_chart.py -b $LOG_ADDR_EXISTS -i $LOG_FILE_PATH -o memory_variable_max_allocated  -s 50 -t 0.002
python3 memory_chart.py -b $LOG_ADDR_EXISTS -i $LOG_FILE_PATH --reserved -o memory_variable_max_reserved  -s 50 -t 0.002
python3 memory_chart.py -b $LOG_ADDR_EXISTS -i $LOG_FILE_PATH --fragment -o memory_variable_fragment  -s 50 -t 0.002

