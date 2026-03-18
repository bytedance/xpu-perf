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

import importlib
import logging
from typing import Any, Dict
import os
import sys
from general_perf.datasets.data_loader import Dataset

log = logging.getLogger("DatasetStore")


def load_dataset(config: Dict[str, Any]) -> Dataset:
    """
    Load related dataset class with config file
    Args: Dict

    Returns: Dataloader()
    """
    if config['dataset_name']:
        dataset_name = config['dataset_name']
        log.info("Loading Dataset: {}".format(config['dataset_name']))
    else:
        dataset_name = 'fake_dataset'
        log.info("Loading Dataset: Dataset does not exist, using fake data")

    data_loader = importlib.import_module('general_perf.datasets.' +
                                          dataset_name + ".data_loader")
    data_loader = getattr(data_loader, 'DataLoader')
    dataset = data_loader(config)
    return dataset
