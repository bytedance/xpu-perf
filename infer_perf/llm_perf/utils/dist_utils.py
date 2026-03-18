import os
import torch
import torch.distributed as dist


def check_dist():
    mp_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    buffer = torch.zeros([1], dtype=torch.int32).cuda()
    if local_rank == 0:
        buffer = buffer + 1

    print(f"rank={local_rank}, before, {buffer}")
    dist.broadcast(buffer, 0)
    print(f"rank={local_rank}, after, {buffer}")
