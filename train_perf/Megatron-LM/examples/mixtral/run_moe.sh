#!/bin/bash

# Runs the "345M" parameter model

PRETRAIN_MODEL_PATH=/data02/huanghantao/Megatron-LM/examples/mixtral/moe/ckpt
CHECKPOINT_SAVE_PATH=/data02/huanghantao/Megatron-LM/examples/mixtral/moe/ckpt
TENSORBOARD_LOGS_PATH=/data02/huanghantao/Megatron-LM/examples/mixtral/moe/tensorboard
# VOCAB_FILE=/data02/huanghantao/Megatron-LM/gpt2-vocab.json
# MERGE_FILE=/data02/huanghantao/Megatron-LM/gpt2-merges.txt
DATA_PATH=/data02/huanghantao/Megatron-LM/moe_test_text_document
TOKENIZER_MODEL=/data02/huanghantao/Megatron-LM/examples/mixtral/Llama2tokenizer.model

# OUTPUT path
LOG_ADDR_EXISTS=/workspace/logs/a100_moe_test_bf16.log
MAX_TRAIN_STEPS=500000
OUTPUT_CMP_ADDR=./output/moe_test_log
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
    mkdir -p "$OUTPUT_CMP_ADDR"
fi

# Define the log file path and create
LOG_FILE_PATH="${OUTPUT_CMP_ADDR}/mlu_moe_test_bf16.log"
# Check if there are any base logs
if [ ! -f "$LOG_FILE_PATH" ]; then
    echo "The file $LOG_FILE_PATH does not exist, creating it now..."
    touch $LOG_FILE_PATH
else
    echo "The file $LOG_FILE_PATH exists, continue execution..."
fi

# Runs moe test model
bash examples/mixtral/train_moe.sh $PRETRAIN_MODEL_PATH $CHECKPOINT_SAVE_PATH $TENSORBOARD_LOGS_PATH $DATA_PATH $TOKENIZER_MODEL $MAX_TRAIN_STEPS 2>&1 | tee $LOG_FILE_PATH 
python3 loss_chart.py -b $LOG_ADDR_EXISTS -i $LOG_FILE_PATH -o $OUTPUT_CMP_ADDR -s 50 -t 0.002 
