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

from typing import Any, Dict


class RuntimeBackend(object):
    def __init__(self):
        self.hardware_type = 'UnKnown'
        self.need_reload = False
        self.need_quant = False

    def version(self) -> str:
        """
        Return runtime backend version details
        """
        raise NotImplementedError("RuntimeBackend:version")

    def load(self, batch_size) -> str:
        """
        Return runtime backend version details
        """
        raise NotImplementedError("RuntimeBackend:load")

    def get_loaded_batch_size(self) -> int:
        """
        Get Currect batch size
        """
        raise NotImplementedError("RuntimeBackend:get_loaded_batch_size")

    def predict(self, data):
        """
        Run the compiled model and return the model output corresponding to the data.
        """
        raise NotImplementedError("RuntimeBackend:predict")

    def is_qs_mode_supported(self) -> bool:
        """
        Used to check whether QSv2 Runtime is enabled
        """
        return False

    def generate_qs_config(self) -> Dict[str, Any]:
        """
        Used only when is_qs_ported return True. Generate QS Config
        File for QSv2 Runtime
        """
        return None

    def benchmark(self, dataloader):
        """
        Performance Testing when qs mode is not enabled.
        """
        raise NotImplementedError("RuntimeBackend:benchmark")
