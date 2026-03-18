# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def my_calibration_input_fn():
    for _ in range(10):
        yield np.random.normal(size=(1, 224, 224, 3)).astype(np.uint8),
        # yield tf.random.normal((1, 224, 224, 3)).astype(np.uint8),


saved_model_path = 'byte_mlperf/model_zoo/resnet50_saved_model'
model_params = tf.experimental.tensorrt.ConversionParams(
    precision_mode="int8".upper(), max_batch_size=64, use_calibration=True)
model_trt = tf.experimental.tensorrt.Converter(
    input_saved_model_dir=saved_model_path, conversion_params=model_params)
model_trt.convert(calibration_input_fn=my_calibration_input_fn)
output_saved_model_dir = 'test'
model_trt.save(output_saved_model_dir)
