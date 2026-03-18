import gc
import os

import psutil
import torch

from llm_perf.utils.logger import logger

def check_memory_usage(tag):

    # dist config
    mp_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    vm_stats = psutil.virtual_memory()
    used_GB = round((vm_stats.total - vm_stats.available) / (1024**3), 2)

    dev_mem_reserved = 0
    dev_mem_allocated = 0
    if torch.cuda.is_available():
        dev_mem_reserved = round(torch.cuda.memory_reserved() / (1024**3), 2)
        dev_mem_allocated = round(torch.cuda.memory_allocated() / (1024**3), 2)
    else:
        pass

    msg = f"<<{tag}>> CPU VM State: Used = {used_GB} GB, Percent = {vm_stats.percent}% | "\
          f"DEV MEM State(Rank{local_rank}): Used = {dev_mem_allocated} GB, Reserved = {dev_mem_reserved} GB"

    if local_rank == 0:
        print(msg)
    # logger.info(msg)

