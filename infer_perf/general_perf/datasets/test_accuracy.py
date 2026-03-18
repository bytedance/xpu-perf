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

import os
import logging
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger("TestAccuracy")


def draw_all_diff(ori_outs, cur_outs, file_name) -> Dict[str, Any]:
    ori_data = ori_outs.flatten()
    cur_data = cur_outs.flatten()
    '''
    Nan & Inf is not compareable, replece with 0 
    '''
    ori_data[np.isnan(ori_data)] = 0.0
    ori_data[np.isinf(ori_data)] = 0.0

    cur_data[np.isnan(cur_data)] = 0.0
    cur_data[np.isinf(cur_data)] = 0.0

    length = min(ori_data.shape[0], 300)
    diff = ori_data - cur_data

    ori_data = np.where(ori_data == 0, 1, ori_data)
    rel_diff = np.divide(diff, ori_data)
    rel_diff = np.nan_to_num(rel_diff)

    log.info('Mean Diff: {}, Std Diff: {}, Max Diff: {}, Max Rel-Diff: {}, Mean Rel-Diff: {}'.format(
        np.mean(abs(diff)), np.std(abs(diff)),
        abs(diff).max(), abs(rel_diff).max(), np.mean(abs(rel_diff))))

    result = {}
    result["Mean Diff"] = round(float(np.mean(abs(diff))), 5)
    result["Std Diff"] = round(float(np.std(abs(diff))), 5)
    result["Max Diff"] = round(float(abs(diff).max()), 5)
    result["Max Rel-Diff"] = round(float(abs(rel_diff).max()), 5)
    result["Mean Rel-Diff"] = round(float(np.mean(abs(rel_diff))), 5)

    plt.figure(figsize=(16, 8))

    plt.cla()

    plt.subplot(1, 3, 1)
    plt.yscale('log')
    plt.hist(diff,
             bins=length,
             alpha=0.5,
             label='Diff',
             range=(diff.min(), diff.max()))
    plt.xlabel("Diff Distribute")

    plt.subplot(1, 3, 2)
    plt.yscale('log')
    plt.hist(ori_data,
             bins=length,
             alpha=0.5,
             label='CPU',
             range=(ori_data.min(), ori_data.max()))
    plt.xlabel("CPU Result")

    plt.subplot(1, 3, 3)
    plt.yscale('log')
    plt.hist(cur_data,
             bins=length,
             alpha=0.5,
             label='Backend',
             range=(cur_data.min(), cur_data.max()))
    plt.xlabel("Backend Result")

    plt.savefig(file_name, dpi=300)
    return result


class AccuracyChecker():
    def __init__(self):
        self.configs = None
        self.dataloader = None
        self.runtime_backend = None
        self.output_dir = ""

    def calculate_diff(self) -> Dict[str, float]:
        """
        Return a dictionary of Mean Diff, Std Diff and Max Diff

        Args: None

        Returns: Dict[str, float]
        """
        cpu_data_path = os.path.abspath('general_perf/reports/CPU/' +
                                        self.configs["model"])
        if not os.path.exists(cpu_data_path):
            log.info("Fetch CPU Data Failed")
            return {}
        vendor_data = np.load(self.output_dir +
                              "/{}.npy".format(self.dataloader.name()))
        cpu_data = np.load(cpu_data_path +
                           "/{}.npy".format(self.dataloader.name()))
        return draw_all_diff(
            cpu_data, vendor_data,
            self.output_dir + "/" + self.configs["model"] + '-to-' + self.configs['compile_precision'].lower() + '.png')

    def calculate_acc(self, data_percent) -> Dict[str, Any]:
        raise NotImplementedError("Dataset: caculate_acc")
