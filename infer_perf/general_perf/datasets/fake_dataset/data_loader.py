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

import logging
import numpy as np
from general_perf.datasets import data_loader

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool
}

log = logging.getLogger("FAKE_DATA")


class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        self.config = config
        self.cur_bs = 1

    def name(self):
        return 'fake_dataset'

    def get_batch_count(self):
        # always return 100
        return 100

    def generate_fake_data(self):
        input_shape = self.config["input_shape"]
        input_type = self.config["input_type"].split(',')

        return self.get_fake_samples_regular(self.cur_bs, input_shape,
                                             input_type)

    def rebatch(self, new_bs, skip=True):
        log.info("Rebatching batch size to: {} ...".format(new_bs))

        if self.cur_bs == new_bs and skip:
            return

        self.cur_bs = new_bs

    def get_samples(self, sample_id):
        if sample_id > 99 or sample_id < 0:
            raise ValueError("Your Input ID is out of range")

        np.random.seed(sample_id)
        return self.generate_fake_data()

    def get_fake_samples_regular(self, batch_size, shape, input_type):
        data = {}
        if not input_type:
            raise ValueError("Please provide input type")
        i = 0
        for key, val in shape.items():
            val = [batch_size] + val[1:]
            if 'LONG' in input_type[i] or 'INT' in input_type[i]:
                if "mask" in key or "segment" in key:
                    data[key] = np.random.randint(
                        low=0, high=2,
                        size=val).astype(INPUT_TYPE[input_type[i]])
                elif self.config[
                        "model"] == "internal_videobert01-onnx-fp32" and key == "1_input_1":
                    data[key] = np.random.ones(size=val).astype(
                        INPUT_TYPE[input_type[i]])
                else:
                    data[key] = np.random.randint(
                        low=0, high=1000,
                        size=val).astype(INPUT_TYPE[input_type[i]])

            elif 'STRING' in input_type[i]:
                data[key] = 'This is a test string.'
            elif 'BOOL' in input_type[i]:
                data[key] = np.zeros(shape=val, dtype=bool)
            else:
                sample_data = np.random.random(size=val) * 2 - 1
                data[key] = sample_data.astype(INPUT_TYPE[input_type[i]])
            i += 1

        return data

    def get_fake_samples_bert(self, batch_size, shape, input_type):
        data = {}

        avg_seq_len = 192
        max_seq_len = 384

        if not input_type:
            raise ValueError("Please provide input type")
        i = 0
        for key, val in shape.items():
            val = [val[0] * batch_size] + val[1:]
            if i == 0:
                # fake input id and mask
                input_ids = np.random.randint(low=0, high=30523,
                                              size=val).astype(
                                                  INPUT_TYPE[input_type[i]])
                data[key] = input_ids
            elif i == 1:
                # fake input array length
                input_len = np.random.randint(low=2 * avg_seq_len -
                                              max_seq_len,
                                              high=max_seq_len + 1,
                                              size=(batch_size),
                                              dtype=np.int32)

                input_mask = np.zeros(val).astype(INPUT_TYPE[input_type[i]])

                for b_idx, s_len in enumerate(input_len):
                    input_mask[b_idx][:s_len] = 1
                data[key] = input_mask
            else:
                data[key] = np.zeros(val).astype(INPUT_TYPE[input_type[i]])
            i += 1
        return data
