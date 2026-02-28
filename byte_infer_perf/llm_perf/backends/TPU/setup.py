import torch
import importlib
from typing import Any, Dict

from llm_perf.core.scheduler import CoreScheduler
from llm_perf.backends.TPU.tpu_inferencer import TpuInferencer
from llm_perf.backends.TPU.tpu_sampler import TpuSampler
from llm_perf.backends.TPU.tpu_scheduler import TpuScheduler
from llm_perf.backends.TPU.tpu_mp_engine import TpuMpEngine
from llm_perf.utils.logger import logger



def get_engine(xpu_cfg) -> TpuMpEngine:
    # get model impl
    hardware_type = xpu_cfg["hardware_type"]
    model_config = xpu_cfg["model_config"]
    model_name = model_config["model_name"]
    
    vendor_model_path = f"llm_perf/backends/{hardware_type}/model_impl"
    vendor_model_impl = importlib.import_module(
        ".", package=vendor_model_path.replace("/", ".")
    )
    vendor_model = vendor_model_impl.__all__[model_name]
    
    mp_engine = TpuMpEngine(
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
    inferencer = TpuInferencer(vendor_model, xpu_cfg)

    # create sampler
    sampler = TpuSampler()

    # create scheduler
    scheduler = TpuScheduler(
        inferencer=inferencer, 
        sampler=sampler, 
        xpu_cfg=xpu_cfg
    )

    return scheduler