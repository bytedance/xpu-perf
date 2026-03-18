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

from typing import Any, Dict, List, Optional


class CompileBackend(object):
    def __init__(self):
        self.hardware_type = 'UnKnown'
        self.need_reload = False
        self.need_quant = False

    def version(self) -> str:
        """
        Return compile backend version details
        """
        raise NotImplementedError("CompileBackend:version")

    def pre_optimize(self, configs: Dict[str, Any]):
        """
        Model pre-optimization interface. Requirements: Model pre-optimization
        cannot change the model format. Torch model export to ONNX is allowed.
        """
        return configs

    def compile(self,
                configs: Dict[str, Any],
                dataloader=None) -> Dict[str, Any]:
        """
        Model compilation interface. Model conversion and compilation 
        can be performed here. The model format can be changed here.

        Arguments:
            configs (list of ``str``s, optional): model configs.
        """
        raise NotImplementedError("CompileBackend:compile")

    def tuning(self, configs: Dict[str, Any]):
        """
        This interface is reserved for the future. The purpose is
        that some compilation optimization needs to be improved
        according to the results of the first compilation and operation.
        The tuning interface provides such a window for tuning.
        """
        return

    def segment(self, configs: Dict[str, Any]):
        """
        This interface is reserved for the future. The purpose is
        to better adapt to the scene of subgraph compilation in the future.
        For manufacturers who place segment and compile in the same stage,
        this interface can be ignored.
        """
        return

    def get_interact_profile(self, config: Dict[str, Any]):
        """
        Load the interactive configuration interface. If the vendor needs
        the user to provide some additional information, you can load the
        json file you added here and return a list of dict. mlperf will 
        display the content of the profile to the user and is responsible
        for collecting feedback about the profile. If the user does not need
        to provide additional information, return None here. get_interact_profile
        can already get some workload info and model info, and the vendor can
        also generate some options other than json under this API.
        """
        raise NotImplementedError("CompileBackend:get_interact_profile")

    def get_best_batch_size(self) -> Optional[List[int]]:
        """
        Get Best Batch Size for the model
        """
        raise NotImplementedError("CompileBackend:get_best_batch_size")
