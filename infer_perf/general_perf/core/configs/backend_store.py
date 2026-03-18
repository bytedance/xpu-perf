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
from general_perf.backends.compile_backend import CompileBackend
from general_perf.backends.runtime_backend import RuntimeBackend

log = logging.getLogger("BackendStore")

__all__ = [
    "CompileBackend",
]


def init_compile_backend(hardware_type: str) -> CompileBackend:
    """
    Load related compile backend with input hardware type

    Arguments: str

    Returns: CompileBackend()
    """
    log.info("Loading Compile Backend: {}".format(hardware_type))

    compile_backend = importlib.import_module('general_perf.backends.' +
                                              hardware_type +
                                              ".compile_backend_" +
                                              hardware_type.lower())
    compile_backend = getattr(compile_backend,
                              "CompileBackend" + hardware_type)
    return compile_backend()


def init_runtime_backend(hardware_type: str) -> RuntimeBackend:
    """
    Load related compile backend with input hardware type

    Arguments: str

    Returns: RuntimeBackend()
    """
    log.info("Loading Runtime Backend: {}".format(hardware_type))

    runtime_backend = importlib.import_module('general_perf.backends.' +
                                              hardware_type +
                                              ".runtime_backend_" +
                                              hardware_type.lower())
    runtime_backend = getattr(runtime_backend,
                              "RuntimeBackend" + hardware_type)
    return runtime_backend()
