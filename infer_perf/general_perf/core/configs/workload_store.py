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

import json
import os
import logging
from typing import Any, List, Dict

log = logging.getLogger("WorkloadStore")


def load_workload(task: str) -> Dict[str, Any]:
    """
    Return a list of dictionary with model Configuration

    Args: List[str]

    Returns: List[dic]
    """
    modules_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))) + '/workloads'

    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)
        if (not file.startswith('_') and not file.startswith('.')
                and (file.endswith('.json') or os.path.isdir(path))
                and file[:file.find('.json')] == task):
            module_name = file
            with open("general_perf/workloads/" + module_name, 'r') as f:
                workload_dict = json.load(f)
            return workload_dict
    else:
        log.error(
            "Task name: [ {} ] was not found, please check your task name".
            format(task))