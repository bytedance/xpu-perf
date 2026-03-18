#!/bin/bash
# change directory to your own

HDFS_trainingdata=/mnt/hdfs/training_data

WORK_PATH=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM


cd $WORK_PATH


# python package , for your reference
pip install -e . 
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention 
git submodule update --init --recursive
git checkout v2.4.2
python3  setup.py install --user
pip3 install  transformer_engine[pytorch]==1.11.0 
pip3 install  transformer_engine[pytorch]==1.11.0 
#pip3 install /mnt/hdfs/training_data/transformer_engine_torch-1.11.0.tar.gz
cd /home/tiger/.local/lib/python3.9/site-packages
mv transformer_engine transformer_engine_old
cp /mnt/hdfs/training_data/transformer_engine . -rf
cd $WORK_PATH
git clone https://github.com/NVIDIA/apex.git
cd apex
NVCC_APPEND_FLAGS="--threads 4" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" ./

cd $WORK_PATH

pip uninstall -y nvidia-modelopt

export OMPI_COMM_WORLD_SIZE=$2
export OMPI_COMM_WORLD_RANK=$3

PRETRAIN_PATH=/mnt/hdfs/training_data/node4/moe_ckpt/
echo "PRETRAIN_PATH is : $PRETRAIN_PATH"


TENSORBOARD_LOGS_PATH=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/examples/ds_like/node2ckpt/tensorboard
VOCAB_FILE=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/gpt2-vocab.json
MERGE_FILE=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/gpt2-merges.txt
DATA_PATH=$1
TOKENIZER_MODEL=/home/tiger/ByteMLPerf/byte_train_perf/Megatron-LM/examples/ds_like/Llama2tokenizer.model


if [ -z "${CHECKPOINT_PATH}" ]; then

    CHECKPOINT_PATH=/mnt/hdfs/training_data/node${OMPI_COMM_WORLD_SIZE}/moe_ckpt_tmp
    echo "CKPT_PATH is not set and will set path: $CHECKPOINT_PATH"
else
    echo "CKPT_PATH is set: $CHECKPOINT_PATH"
fi

if [ -z "${ATTN_DROPOUT}" ]; then

    export ATTN_DROPOUT=0.1
    echo "ATTN_DROPOUT is not set and will set: $ATTN_DROPOUT"
else
    echo "ATTN_DROPOUT is set: $ATTN_DROPOUT"
fi

if [ -z "${HID_DROPOUT}" ]; then

    export HID_DROPOUT=0.1
    echo "HID_DROPOUT is not set and will set: $HID_DROPOUT"
else
    echo "HID_DROPOUT is set: $HID_DROPOUT"
fi



export OMPI_COMM_WORLD_SIZE=$2
export OMPI_COMM_WORLD_RANK=$3

MAX_TRAIN_SAMPLES=15360000

bash ./examples/ds_like/train_ds_like_4node.sh $PRETRAIN_PATH $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $DATA_PATH $TOKENIZER_MODEL $MAX_TRAIN_SAMPLES $4 $5

