#!/bin/bash
# Runs MOE run

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=$7
MASTER_PORT=$8
NUM_NODES=$OMPI_COMM_WORLD_SIZE
NODE_RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

PRETRAIN_MODEL_PATH=$1   # <Specify path>
CHECKPOINT_SAVE_PATH=$2  # <Specify path>
TENSORBOARD_LOGS_PATH=$3 # <Specify path>
DATA_PATH=$4             # <Specify path and file prefix>_text_document
TOKENIZER_MODEL=$5       # tokenizer path
#MAX_TRAIN_STEPS=$6       # train iters
MAX_TRAIN_SAMPLES=$6      # train samples, eg. 15360000 (1w iter)
BATCH_WARMUP_SAMPLES=$(($MAX_TRAIN_SAMPLES / 31)) 
DECAY_SAMPLES=$(($MAX_TRAIN_SAMPLES / 2)) 
LR_WARMUP_SAMPLES=$(( 2000 * 1536 ))

export NCCL_ALGO=Ring                     # for deterministic
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 # for deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8    # for deterministic

echo "Dropout rate is set: $ATTN_DROPOUT, $HID_DROPOUT"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node-rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 16
    --hidden-size 5120
    --num-attention-heads 40
    --init-method-std 0.01
    --attention-dropout $ATTN_DROPOUT
    --hidden-dropout $HID_DROPOUT
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --attention-softmax-in-fp32
    #--use-cpu-initialization
)

MOE_ARGS=(
    --num-experts 128
    --moe-grouped-gemm
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
    --moe-router-topk 6
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-3
    --use-distributed-optimizer
    --moe-token-dispatcher-type alltoall
    --moe-ffn-hidden-size 1024
    --moe-shared-expert-intermediate-size 2048 # shared-experts 2
    --moe-expert-capacity-factor 1.2
)

TRAINING_ARGS=(
    --seed 1234
    --micro-batch-size 1
    --global-batch-size 1536
    --lr 1e-4
    --train-samples $MAX_TRAIN_SAMPLES
    --rampup-batch-size 128 32 $BATCH_WARMUP_SAMPLES
    --lr-decay-samples $DECAY_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr-decay-style multi-step
    --lr-decay-multi-step 0.6 0.3 0.1
    --min-lr 1e-5
    --lr-warmup-init 1.0e-7
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --norm-epsilon 1e-6
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --num-layers-per-virtual-pipeline-stage 1    
    --expert-model-parallel-size 16
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

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 10
    --data-cache-path /mnt/hdfs/training_data/node${NUM_NODES}/cache 
    --save $CHECKPOINT_SAVE_PATH
    --load $PRETRAIN_MODEL_PATH
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
