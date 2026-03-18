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

log = logging.getLogger("Dataset")

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "FLOAT16": np.float16,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool
}


class Dataset():
    def __init__(self, config):
        self.config = config
        self.cur_bs = 1
        self.batched_data = []
        self.labels = []
        self.items = 0
        self.batch_num = int(self.items / self.cur_bs)

    def name(self) -> str:
        """
        Return the name of dataset
        """
        raise NotImplementedError("Dataset:name")

    def get_item_count(self) -> int:
        """
        Return the number of data loaded
        """
        return self.items

    def get_batch_count(self) -> int:
        """
        Return the number of batched data
        """
        return self.batch_num

    def preprocess(self):
        """
        Data preprocess will happened here
        """
        return

    def get_samples(self, sample_id):
        """
        Query data with sample id
        """
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID is out of range")
        return self.batched_data[sample_id], self.labels[sample_id]

    def rebatch(self, new_bs, skip=True) -> None:
        """
        Rebatch Datasets to specified number
        """
        raise NotImplementedError("Dataset:rebatch")

    def get_fake_samples(self, batch_size, shape, input_type):
        """
        Generate fake data for testing
        """
        data = {}
        if not input_type:
            raise ValueError("Please provide input type")
        i = 0
        for key, val in shape.items():
            val = [val[0] * batch_size] + val[1:]
            data[key] = np.random.random(size=val).astype(
                INPUT_TYPE[input_type[i]])
            i += 1
        return data
