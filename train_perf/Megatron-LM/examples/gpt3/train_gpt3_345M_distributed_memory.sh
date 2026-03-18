#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CNCL_AVOID_RECORD_STREAMS=1
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_SAVE_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document
PRETRAIN_MODEL_PATH=$6
MAX_TRAIN_STEPS=$7

export NCCL_ALGO=Ring # for deterministic
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0  # for deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for deterministic

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --test-memory \
    --num-layers 12 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 32768 \
    --max-position-embeddings 32768
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 32 #1536 
    --train-iters $MAX_TRAIN_STEPS 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --deterministic-mode # for deterministic
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 2000 
    --save $CHECKPOINT_SAVE_PATH 
    --load $PRETRAIN_MODEL_PATH 
    --eval-iters 2000
    --data-cache-path ./cache 
    #--wandb-project benchmark_training
    #--wandb-exp-name gpt3_345M_deterministic
    #--tensorboard-dir $TENSORBOARD_LOGS_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
