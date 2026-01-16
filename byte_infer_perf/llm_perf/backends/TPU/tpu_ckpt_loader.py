import torch
import torch_tpu
import torch.distributed as dist

from llm_perf.core.ckpt_loader import CoreCkptLoader

class TpuCkptLoader(CoreCkptLoader):
    def __init__(
        self,
        prefix, model,
        mp_size=1, mp_rank=0,
        ckpt_path: str=""
    ):
        super().__init__(prefix, model, mp_size, mp_rank, ckpt_path)
        
    def weight_to_device(self, weight: torch.Tensor, non_blocking=False):
        if self.mp_rank == 0:
            weight = weight.tpu(non_blocking=non_blocking)
        else:
            cur_device = torch.tpu.current_device()
            weight = torch.emtpy_like(weight, device=f"tpu:{cur_device}")
        return weight
    
    
    def broadcast_weight(self, key, device='cpu', non_blocking=False):
        if self.mp_rank != 0:
            tensor_shape = self.state_dict[key]["shape"]
            tensor_dtype = self.state_dict[key]["dtype"]
            tensor = torch.empty(tensor_shape, dtype=tensor_dtype)
        else:
            tensor = self.state_dict[key].cpu()
        tensor_tpu = self.weight_to_device(tensor, non_blocking=non_blocking)
        dist.broadcast(tensor_tpu, src=0)
        self.state_dict[key] = tensor_tpu
        
    def scatter_weight(self, key, dim, split_mode='default', outter=1, device='cpu', non_blocking=False):
        self.broadcast_weight(key, non_blocking=non_blocking)
        weight = self.state_dict[key]

        if split_mode == 'default':
            weight_split = self.split(weight, dim)
        elif split_mode == 'with_outter':
            weight_split = self.with_outter_split(weight, dim, outter)
        elif split_mode == 'split_outter':
            weight_split = self.split(weight, dim, outter)
        else:
            assert False, f"unknown split mode {split_mode}"

        weight_split = [x.contiguous() for x in weight_split]
        self.state_dict[key] = weight_split[self.mp_rank]