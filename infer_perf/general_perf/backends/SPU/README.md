# Sparse Processing Unit (SPU)

The inventor of the dual sparsity algorithm, Moffett AI has the world's leading sparse computing techniques with more
than 30 patents worldwide. The company creates a new generation of AI computing platform with hardware and software
co-design to achieve order-of-magnitude acceleration of computing performance, reducing latency and low TCO.

The result in Byte MLPerf has demonstrated the potential of sparse computing in inference performance and energy
efficiency, which leads to a lower total cost of ownership (TCO).

For Byte MLPerf, Moffett has submitted performance results of the following models.

| Model                | Precision | Sparsity* |QPS   | Dataset        | Metric name | Metric value | 
|----------------------|-----------|----------|------|----------------|-------------|--------------|
| resnet50-torch-fp32     | INT8      | 16x      | 52423 | Open Imagenet  | Top-1       | 76.61%       |
| bert-torch-fp32         | INT8/BF16 | 16x      | 7738 | Open Squad 1.1 | F1 Score    | 86.09        |
| albert-torch-fp32       | INT8/BF16 | 16x      | 10824 | Open Squad 1.1 | F1 Score    | 87.66        |
| roberta-torch-fp32      | INT8/BF16 | 16x      | 8107 | Open Squad 1.1 | F1 Score    | 86.63        |
| conformer-encoder-onnx-fp32 | INT8/BF16 | 8x       | 8211 | Fake Dataset   | Mean Diff   | 1.50       |

\* The sparsity is determined by the ratio of time spent on Matmul operations compared to the overall time of model inference.

Besides the performance results, energy efficiency is another significant highlight of Moffett's devices. For example,
the peak power consumption of S30 is merely
250W.

The Antoum architecture through hardware and software co-design and Moffett's original sparsity algorithm are the
reasons to achieve great performance with high energy efficiency.

The accelerators for AI inference applications in data centers are equipped with Moffett's 1st generation Antoum
processor - the first commercial AI processor with 32x sparsity in the world.

Besides the sparse processing units (SPU) for native sparse convolution and matrix computing in Antoum, the processor
also integrates a Vector Processing Unit (VPU), which enables flexible programmability to keep up with the fast
evolution of AI models.

Also, the on-chip Video Codec, which supports 192-way 1080p video decoding at 30 FPS, and the JPEG decoder, which
supports 1080p image decoding up to 6960 FPS, provide an end-to-end capability for video and image inference workloads.

Moffett provides three SKUs of sparse computing devices, namely S4, S10, and S30. Based on the computing power of S4,
S10 and S30 are designed to be 2 times and 3 times the computing power of S4 respectively.

## How to run
### 1. Environmental preparation
#### Download offline image
```bash
wget moffett-oss-bucket01.oss-cn-shenzhen.aliyuncs.com/byte-perf/byte-perf-2.3.2-20230721.tar
```
#### Load offline image
```bash
docker load -i byte-perf-2.3.2-20230721.tar
```
#### Decompress the model data package
```bash
wget moffett-oss-bucket01.oss-cn-shenzhen.aliyuncs.com/byte-perf/byte-perf-data.tar.gz
tar -zxvf byte-perf-data.tar.gz
```
### 2. Create docker container
notes: --shm-size="300g" is recommended to be 95% of the total memory of the host.
```bash
cd byte-perf-data
sudo docker run -itd \
    --privileged \
    --cap-add=ALL \
    --net=host \
    -v /dev:/dev \
    -v /usr/src:/usr/src \
    -v /lib/modules:/lib/modules \
    -v $PWD/package:/home/moffett/workspace/package \
    -e ROOT_PASS=moffett \
    --shm-size="300g" \
    --name byte-perf-2023 \
    byte-perf:2.3.2-20230721
``` 
### 3. Environment initialization 
Environment initialization please operate in the container.
```bash=
docker exec -it byte-perf-2023 /bin/bash
```
#### Install drivers and load firmware
```bash
cd /usr/local/sola/driver/bin/
sudo ./setup.sh
```
#### Device basic information verification 
mf-smi is a command line utility that can view various information of S30, such as card number, usage, temperature, power consumption, etc.
After the driver is successfully installed, execute mf-smi to view the basic information of the device.
```bash
mf-smi 
```

### 4. Run byte-mlperf task in container

```bash=
cd /home/moffett/workspace/package/bytemlperf

# config spu-backend env 
export PYTHONPATH=$PYTHONPATH:/home/moffett/workspace/spu-backend-release/ubuntu18.04-gcc7.5.0-x86_64/lib/

# conformer
python3 launch.py --task conformer-encoder-onnx-fp32 --hardware_type SPU

# albert
python3 launch.py --task albert-torch-fp32 --hardware_type SPU

# bert
python3 launch.py --task bert-torch-fp32 --hardware_type SPU

# roberta
python3 launch.py --task roberta-torch-fp32 --hardware_type SPU

# resnet50
python3 launch.py --task resnet50-torch-fp32 --hardware_type SPU
```

## Contact us

If you are interested in further information about the products, please contact the email: sales@moffett.ai
