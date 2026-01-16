

# How to run

## 1. Create docker container

```bash
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name TPUPerf -td -v /dev/:/dev/ -v /opt/:/opt/ -v <your path>:/workspace/ --entrypoint bash sophgo/tpuc_dev:latest
docker exec -it TPUPerf bash
```

## 2. Environment Initialization

```bash
pip3 install tpu_mlir
apt install unzip
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/BM1690/sophon-sail2.zip
unzip sophon-sail2.zip
# 依照sail2目录下的README，在当前环境编译出whl并安装
```

## 3. Run ByteMLPerf for TPU backend

```bash
python3  launch.py --task yolov5-onnx-fp32 --hardware_type TPU
python3  launch.py --task resnet50-torch-fp32 --hardware_type TPU
```

# Notes
> Support FP32 and INT8 quantization for resnet50-torch-fp32 now, .