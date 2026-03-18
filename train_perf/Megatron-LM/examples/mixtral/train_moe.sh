#!/bin/bash

# Runs MOE run

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

PRETRAIN_MODEL_PATH=$1 #<Specify path>
CHECKPOINT_SAVE_PATH=$2 #<Specify path>
TENSORBOARD_LOGS_PATH=$3 #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$4 #<Specify path and file prefix>_text_document
TOKENIZER_MODEL=$5  #tokenizer path
MAX_TRAIN_STEPS=$6 #train iters

export NCCL_ALGO=Ring # for deterministic
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0  # for deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for deterministic

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 6144
    --ffn-hidden-size 1024
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 8
    --moe-grouped-gemm
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --use-distributed-optimizer
    --moe-token-dispatcher-type alltoall
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters $MAX_TRAIN_STEPS
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

# # to update
DATA_ARGS=(
     --tokenizer-type Llama2Tokenizer
     --tokenizer-model ${TOKENIZER_MODEL}
     --data-path $DATA_PATH
     --split 99990,8,2
 )
#  DATA_ARGS=(
#      --data-path $DATA_PATH 
#      --vocab-file $VOCAB_FILE 
#      --merge-file $MERGE_FILE 
#      --split 949,50,1
#  )

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_SAVE_PATH 
    --load $PRETRAIN_MODEL_PATH 
    --eval-iters 10
    --data-cache-path ./cache
    # --wandb-project benchmark_training
    # --wandb-exp-name moe8x2-7B
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH
)




torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
