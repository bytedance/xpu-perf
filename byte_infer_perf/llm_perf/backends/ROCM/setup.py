import torch
import importlib
from typing import Any, Dict

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.ROCM.gpu_inferencer import GpuInferencer
from llm_perf.backends.GPU.gpu_sampler import GpuSampler
from llm_perf.backends.GPU.gpu_scheduler import GpuScheduler
from llm_perf.backends.ROCM.gpu_mp_engine import GpuMpEngine
from llm_perf.backends.ROCM.gpu_mp_engine import GpuMpEngineWithGraph
from llm_perf.utils.logger import logger
import os

def get_device_name():
    return torch.cuda.get_device_name(0)


def get_engine(xpu_cfg) -> CoreScheduler:
    # get model impl
    hardware_type = xpu_cfg["hardware_type"]
    model_config = xpu_cfg["model_config"]
    model_name = model_config["model_name"]

    vendor_model_path = f"llm_perf/backends/{hardware_type}/model_impl"
    vendor_model_impl = importlib.import_module(
        ".", package=vendor_model_path.replace("/", ".")
    )
    vendor_model = vendor_model_impl.__all__[model_name]

    is_graph = int(os.environ.get("ENABLE_GRAPH", "0"))

    if is_graph:
        mp_engine = GpuMpEngineWithGraph(
            world_size=xpu_cfg["tp_size"],
            model_impl=vendor_model,
            xpu_cfg=xpu_cfg
        )
        return mp_engine
    else:
        mp_engine = GpuMpEngine(
            world_size=xpu_cfg["tp_size"],
            model_impl=vendor_model,
            xpu_cfg=xpu_cfg
        )
        return mp_engine


def setup_scheduler(xpu_cfg) -> CoreScheduler:

    # get model impl
    hardware_type = xpu_cfg["hardware_type"]
    model_config = xpu_cfg["model_config"]
    model_name = model_config["model_name"]

    vendor_model_path = f"llm_perf/backends/{hardware_type}/model_impl"
    vendor_model_impl = importlib.import_module(
        ".", package=vendor_model_path.replace("/", ".")
    )
    vendor_model = vendor_model_impl.__all__[model_name]

    # create inferencer
    inferencer = GpuInferencer(vendor_model, xpu_cfg)

    # create sampler
    sampler = GpuSampler()

    # create scheduler
    scheduler = GpuScheduler(
        inferencer=inferencer, 
        sampler=sampler, 
        xpu_cfg=xpu_cfg
    )

    return scheduler
