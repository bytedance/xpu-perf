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
from general_perf.datasets import test_accuracy
from tqdm import tqdm

log = logging.getLogger("TestAccuracy")


class AccuracyChecker(test_accuracy.AccuracyChecker):
    def calculate_acc(self, data_percent=10):
        log.info("Start to calculate accuracy...")
        num = int((data_percent / 100) * self.dataloader.get_batch_count()
                  ) if data_percent else self.dataloader.get_batch_count()

        diffs = []
        for i in tqdm(range(num)):
            test_data = self.dataloader.get_samples(i)

            results = self.runtime_backend.predict(test_data)
            if isinstance(results, dict):
                list_key = list(results.keys())
                list_key.sort()
                for key in list_key:
                    diffs.extend(results[key].flatten())
            elif isinstance(results, list):
                for out in results:
                    diffs.extend(out.flatten())
            else:
                diffs.extend(results)

        log.info('Batch size is {}, Accuracy: {}'.format(
            self.dataloader.cur_bs, 0.0))
        np.save(self.output_dir + "/{}.npy".format(self.dataloader.name()),
                np.array(diffs),
                allow_pickle=True)
        return {"Fake Dataset Accuracy": 0}
