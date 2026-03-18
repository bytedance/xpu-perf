#!/bin/bash

echo "******************* Downloading Model....  *******************"

mkdir -p general_perf/model_zoo/regular
mkdir -p general_perf/model_zoo/popular
mkdir -p general_perf/model_zoo/sota
mkdir -p general_perf/download

#--Basic Model--
# https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/bert_mhlo.tar
# https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/resnet50_mhlo.tar

if [ $1 == "bert-tf-fp32" -o $1 == "bert-torch-fp32" ]; then
    wget -O general_perf/download/open_bert.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_bert.tar
    tar xf general_perf/download/open_bert.tar -C general_perf/model_zoo/regular/
elif [ $1 == "resnet50-tf-fp32" -o $1 == "resnet50-torch-fp32" ]; then
    wget -O general_perf/download/resnet50.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/resnet50.tar
    tar xf general_perf/download/resnet50.tar -C general_perf/model_zoo/regular/
elif [ $1 == "widedeep-tf-fp32" ]; then
    wget -O general_perf/download/open_wide_deep.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_wide_deep_saved_model.tar
    tar xf general_perf/download/open_wide_deep.tar -C general_perf/model_zoo/regular/
#--Popular Model--
elif [ $1 == "albert-torch-fp32" ]; then
    wget -O general_perf/download/open_albert.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_albert.tar
    tar xf general_perf/download/open_albert.tar -C general_perf/model_zoo/popular/ 
elif [ $1 == "roformer-tf-fp32" ]; then
    wget -O general_perf/download/open_roformer.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_roformer.tar
    tar xf general_perf/download/open_roformer.tar -C general_perf/model_zoo/popular/
elif [ $1 == "videobert-onnx-fp32" ]; then
    wget -O general_perf/download/open_videobert.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_videobert.tar
    tar xf general_perf/download/open_videobert.tar -C general_perf/model_zoo/popular/
elif [ $1 == "yolov5-onnx-fp32" ]; then
    wget -O general_perf/download/open_yolov5.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_yolov5.tar
    tar xf general_perf/download/open_yolov5.tar -C general_perf/model_zoo/popular/
elif [ $1 == "conformer-encoder-onnx-fp32" ]; then
    wget -O general_perf/download/open_conformer.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_conformer.tar
    tar xf general_perf/download/open_conformer.tar -C general_perf/model_zoo/popular/
elif [ $1 == "roberta-torch-fp32" ]; then
    wget -O general_perf/download/open_roberta.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_roberta.tar
    tar xf general_perf/download/open_roberta.tar -C general_perf/model_zoo/popular/
elif [ $1 == "deberta-torch-fp32" ]; then
    wget -O general_perf/download/open_deberta.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_deberta.tar
    tar xf general_perf/download/open_deberta.tar -C general_perf/model_zoo/popular/
elif [ $1 == "swin-large-torch-fp32" ]; then
    wget -O general_perf/download/open-swin-large.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open-swin-large.tar
    tar xf general_perf/download/open-swin-large.tar -C general_perf/model_zoo/popular/
#--Sota Model--
elif [ $1 == "vae-encoder-onnx-fp32" -o $1 == "vae-decoder-onnx-fp32" -o $1 == "clip-onnx-fp32" ]; then
    wget -O general_perf/download/stable_diffusion.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/stable_diffusion.tar
    tar xf general_perf/download/stable_diffusion.tar -C general_perf/model_zoo/sota/
elif [ $1 == "unet-onnx-fp32" ]; then
    wget -O general_perf/download/unet.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/unet.tar
    tar xf general_perf/download/unet.tar -C general_perf/model_zoo/sota/
fi

# Download Datasets
if [ $2 == "open_imagenet" ] && [ ! -f "general_perf/download/open_imagenet.tar" ] ; then
    wget -O general_perf/download/open_imagenet.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_imagenet.tar
    tar xf general_perf/download/open_imagenet.tar -C general_perf/datasets/
elif [ $2 == "open_squad" ] && [ ! -f "general_perf/download/open_squad.tar" ]; then
    wget -O general_perf/download/open_squad.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_squad.tar
    tar xf general_perf/download/open_squad.tar -C general_perf/datasets/open_squad
elif [ $2 == "open_criteo_kaggle" ] && [ ! -f "general_perf/download/eval.csv" ]; then
    wget -O general_perf/download/eval.csv https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/eval.csv
    cp general_perf/download/eval.csv general_perf/datasets/open_criteo_kaggle/eval.csv
elif [ $2 == "open_cail2019" ] && [ ! -f "general_perf/download/open_cail2019.tar" ]; then
    wget -O general_perf/download/open_cail2019.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/open_cail2019.tar
    tar xf general_perf/download/open_cail2019.tar -C general_perf/datasets/open_cail2019 --strip-components 1
elif [ $2 == "open_cifar" ] && [ ! -f "general_perf/download/cifar-100-python.tar" ]; then
    wget -O general_perf/download/cifar-100-python.tar https://lf-xpu-perf.bytetos.com/obj/bytemlperf-zoo/cifar-100-python.tar
    tar xf general_perf/download/cifar-100-python.tar -C general_perf/datasets/open_cifar
fi

echo "Extract Done."
