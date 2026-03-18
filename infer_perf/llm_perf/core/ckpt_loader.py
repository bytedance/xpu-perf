import json
import pathlib
from tqdm import tqdm
from safetensors import safe_open

import torch
import torch.nn as nn
import torch.distributed as dist

from llm_perf.utils.logger import logger

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from typing import Union, List, Dict


class CoreCkptLoader(ABC):
    def __init__(
        self, 
        prefix, model, 
        mp_size=1, mp_rank=0, 
        ckpt_path: str = ""
    ):
        self.prefix = prefix
        self.model = model

        self.mp_size = mp_size
        self.mp_rank = mp_rank

        self.ckpt_path = ckpt_path
        
        self.state_dict = None


    def to_parameter(
        self, 
        data : torch.Tensor, 
        dtype : torch.dtype =None
    ):
        if dtype is not None:
            data = data.to(dtype)
        return nn.Parameter(data, requires_grad=False)


    def to_contiguous(self, num_layers, param_suffixes, prefix, state_dict): 
        result = {}

        with ThreadPoolExecutor() as executor:
            for i in range(num_layers):
                for suffix in param_suffixes:
                    # for example: 
                    # "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight"
                    name = f"{prefix}.{i}.{suffix}"
                    if name in state_dict:
                        result[name] = executor.submit(lambda t : t.contiguous(), state_dict[name])
        
        for i in range(num_layers):
            for suffix in param_suffixes:
                name = f"{prefix}.{i}.{suffix}"
                if name in state_dict:
                    state_dict[name] = result[name].result


    def gqa_split(self, src, dim):
        qkv_head_num = src.shape[dim] // self.head_dim
        src_split = src.chunk(qkv_head_num, dim=dim)
        qkv_cat = []
        for i in range(self.mp_size):
            qkv_cat.append(
                torch.cat(
                    [src_split[i * self.mp_size + self.mp_rank] for i in range(qkv_head_num // self.mp_size)],
                    axis=dim,
                )
            )

        return qkv_cat


    def qkv_split(self, src, dim):
        src_split = torch.split(src.data, src.shape[dim] // 3, dim=dim)
        qkv_split = [torch.split(src_s, src_s.shape[dim] // self.mp_size, dim=dim) for src_s in src_split]
        qkv_cat = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=dim) for i in range(len(qkv_split[0]))]
        return qkv_cat
        

    def with_outter_split(
        self, 
        src : torch.Tensor, 
        dim : int, 
        outter : int
    ):
        src_split = torch.split(src.data, src.shape[dim] // outter, dim=dim)
        output_split = [torch.split(src_s, src_s.shape[dim] // self.mp_size, dim=dim) for src_s in src_split]
        output_tensors = [
            torch.cat(
                [output_s[i] for output_s in output_split], 
                axis=dim
            ) for i in range(len(output_split[0]))
        ]

        output_tensors = [tensor.contiguous() for tensor in output_tensors]
        return output_tensors
        

    def split(
        self, 
        src : torch.Tensor, 
        dim : int, 
        chunks : List [int]=[]
    ):
        if len(chunks) == 0:
            split_arg = src.shape[dim] // self.mp_size
            output_tensors = torch.split(src, split_arg, dim=dim)
        else:
            # for example
            #   chunks = [32, 2, 2], sum_chunks = 36, src.shape[dim] = (32 + 2 + 2) * 128, other_dim = 128
            #   mp_size = 8
            #   new_chunks = [4, 1, 1]
            sum_chunks = sum(chunks)
            other_dim_size = src.shape[dim] // sum_chunks

            split_arg = [i * other_dim_size for i in chunks]            
            split_tensors = torch.split(src, split_arg, dim=dim)

            output_split = []
            for i, tensor in enumerate(split_tensors):
                if self.mp_size > chunks[i]:
                    tensor_shape = tensor.size()[:dim] + (chunks[i], 1, other_dim_size) + tensor.size()[dim+1:]
                    new_tensor_shape = tensor.size()[:dim] + (chunks[i], self.mp_size // chunks[i], other_dim_size) + tensor.size()[dim+1:]
                    output_tensor_shape = tensor.size()[:dim] + (self.mp_size * other_dim_size,) + tensor.size()[dim+1:]

                    tensor = tensor.view(tensor_shape)
                    tensor = tensor.expand(*new_tensor_shape)
                    tensor = tensor.contiguous()
                    tensor = tensor.view(output_tensor_shape)

                cur_split = torch.split(tensor, tensor.shape[dim] // self.mp_size, dim=dim)
                output_split.append(cur_split)

            output_tensors = []
            for i in range(self.mp_size):
                temp_tensors = [output_split[j][i] for j in range(len(chunks))]    
                tp_tensors = torch.concat(temp_tensors, dim=dim)
                output_tensors.append(tp_tensors)

        output_tensors = [tensor.contiguous() for tensor in output_tensors]
        return output_tensors


    def broadcast_meta(self):
        meta = [
            {k: {"shape": v.shape, "dtype": v.dtype} for k, v in self.state_dict.items()}
        ] if self.mp_rank == 0 else [None]
        dist.broadcast_object_list(meta, src=0)

        if self.mp_rank != 0:
            self.state_dict = meta[0]

    @abstractmethod
    def broadcast_weight(self, key, device='cpu', non_blocking=False):
        raise NotImplementedError


    # split_mode
    #   default
    #   with_outter
    #   split_outter
    @abstractmethod
    def scatter_weight(self, key, dim, split_mode='default', outter=1, non_blocking=False):
        raise NotImplementedError


    @abstractmethod
    def parallel_loader(self):
        raise NotImplementedError


    @abstractmethod
    def infusion_to_model(self):
        raise NotImplementedError





class ModelLoader():
    def __init__(
        self, 
        model_dir : pathlib.Path, 
        total_size : int, 
        weight_map: Dict[str, str]
    ) -> None:
        self.model_dir = model_dir
        self.total_size = total_size
        # {tensor_name: file_name} map
        self.weight_map = weight_map
        
        weight_set = set()
        for weight_name in weight_map:
            weight_set.add(weight_map[weight_name])

        self.file_num = len(weight_set)

        # loaded bytes
        self.loaded_bytes = 0

        # {tensor_name: tensor} map
        self.weight_dict = {}

        # {file_name: {tensor_name: tensor}} map
        self.file_cache = {}


    def load_tensor(
        self, 
        tensor_name: str
    ):
        if not tensor_name in self.weight_map:
            logger.error(f"tensor_name {tensor_name} not in weight_map")
            return

        if not self.file_cache:
            self.p_bar = tqdm(total=self.file_num, desc="loading model")

        file_name = self.weight_map[tensor_name]
        if not file_name in self.file_cache:
            if file_name.endswith(".safetensors"):
                with safe_open(
                    self.model_dir.joinpath(file_name), 
                    framework="pt", 
                    device="cpu"
                ) as f:
                    self.file_cache[file_name] = {}
                    for key in f.keys():
                        self.file_cache[file_name][key] = f.get_tensor(key)
                        self.loaded_bytes += self.file_cache[file_name][key].numel() * self.file_cache[file_name][key].element_size()
            elif file_name.endswith(".bin"):
                self.file_cache[file_name] = torch.load(
                    self.model_dir.joinpath(file_name),
                    map_location="cpu"
                )
                for key in self.file_cache[file_name]:
                    self.loaded_bytes += self.file_cache[file_name][key].numel() * self.file_cache[file_name][key].element_size()
            else:
                logger.error(f"file_name {file_name} not supported")
                return
            self.p_bar.update(1)
            if self.p_bar.n == self.file_num:
                self.p_bar.close()
                self.p_bar = None
        self.weight_dict[tensor_name] = self.file_cache[file_name][tensor_name]


class ChatGLM2_ModelLoader(ModelLoader):
    def __init__(
        self,
        model_dir : pathlib.Path,
        model_config,
        weight_index_config: Dict,
    ) -> None:
        # parent class
        super().__init__(
            model_dir,
            weight_index_config["metadata"]["total_size"],
            weight_index_config["weight_map"]
        )
        self.model_config = model_config


    def load_weight(self):
        self.loaded_bytes = 0
        self.weight_dict = {}

        self.load_tensor("transformer.embedding.word_embeddings.weight")
        self.load_tensor("transformer.rotary_pos_emb.inv_freq")
        for i in range(self.model_config.num_layers):
            self.load_tensor(f"transformer.encoder.layers.{i}.input_layernorm.weight")
            self.load_tensor(f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight")
            self.load_tensor(f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight")
            self.load_tensor(f"transformer.encoder.layers.{i}.post_attention_layernorm.weight")
            self.load_tensor(f"transformer.encoder.layers.{i}.self_attention.dense.weight")
            self.load_tensor(f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias")
            self.load_tensor(f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight")
        self.load_tensor("transformer.encoder.final_layernorm.weight")
        self.load_tensor("transformer.output_layer.weight")

        weight_bytes = 0
        for tensor_name in self.weight_dict:
            tensor = self.weight_dict[tensor_name]
            weight_bytes += tensor.numel() * tensor.element_size()
        
        logger.info(f"total_size: {self.total_size}, loaded_bytes: {self.loaded_bytes}, weight_bytes: {weight_bytes}")
        assert self.loaded_bytes == self.total_size
        assert weight_bytes == self.total_size

        return self.weight_dict


from transformers import LlamaConfig
class Llama_ModelLoader(ModelLoader):
    def __init__(self, model_dir : pathlib.Path):
        model_config = LlamaConfig.from_pretrained(model_dir)
        weight_index_config = {}
        for child in model_dir.iterdir():
            if child.name.endswith(".index.json"):
                with open(child, "r") as f:
                    weight_index_config = json.load(f)
                break
        
        self.layer_num = model_config.num_hidden_layers

        super().__init__(
            model_dir,
            weight_index_config["metadata"]["total_size"],
            weight_index_config["weight_map"]
        )

    def load_weight(self):
        self.loaded_bytes = 0
        self.weight_dict = {}

        self.load_tensor("model.embed_tokens.weight")
        for i in range(self.layer_num):
            self.load_tensor(f"model.layers.{i}.input_layernorm.weight")

            self.load_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
            self.load_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
            self.load_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
            self.load_tensor(f"model.layers.{i}.self_attn.o_proj.weight")

            self.load_tensor(f"model.layers.{i}.post_attention_layernorm.weight")

            self.load_tensor(f"model.layers.{i}.mlp.gate_proj.weight")
            self.load_tensor(f"model.layers.{i}.mlp.up_proj.weight")
            self.load_tensor(f"model.layers.{i}.mlp.down_proj.weight")

        self.load_tensor("model.norm.weight")
        self.load_tensor("lm_head.weight")


        weight_bytes = 0
        for tensor_name in self.weight_dict:
            tensor = self.weight_dict[tensor_name]
            weight_bytes += tensor.numel() * tensor.element_size()
        
        logger.info(f"total_size: {self.total_size}, loaded_bytes: {self.loaded_bytes}, weight_bytes: {weight_bytes}")
        assert self.loaded_bytes == self.total_size
        assert weight_bytes == self.total_size

        return self.weight_dict






from transformers import MixtralConfig

class Mixtral_ModelLoader(ModelLoader):
    def __init__(
        self, 
        model_dir : pathlib.Path
    ) -> None:
        model_config = MixtralConfig.from_pretrained(model_dir)
        weight_index_config = {}
        for child in model_dir.iterdir():
            if child.name.endswith(".index.json"):
                with open(child, "r") as f:
                    weight_index_config = json.load(f)
                break

        self.layer_num = model_config.num_hidden_layers
        self.expert_num = model_config.num_local_experts

        # parent class
        super().__init__(
            model_dir,
            weight_index_config["metadata"]["total_size"],
            weight_index_config["weight_map"]
        )


    def load_weight(self):
        self.loaded_bytes = 0
        self.weight_dict = {}

        self.load_tensor("model.embed_tokens.weight")
        for i in range(self.layer_num):
            self.load_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
            self.load_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
            self.load_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
            self.load_tensor(f"model.layers.{i}.self_attn.o_proj.weight")
            self.load_tensor(f"model.layers.{i}.block_sparse_moe.gate.weight")
            for j in range(self.expert_num):
                self.load_tensor(f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight")
                self.load_tensor(f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight")
                self.load_tensor(f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight")
            self.load_tensor(f"model.layers.{i}.input_layernorm.weight")
            self.load_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
        self.load_tensor("model.norm.weight")
        self.load_tensor("lm_head.weight")


        weight_bytes = 0
        for tensor_name in self.weight_dict:
            tensor = self.weight_dict[tensor_name]
            weight_bytes += tensor.numel() * tensor.element_size()
        
        logger.info(f"total_size: {self.total_size}, loaded_bytes: {self.loaded_bytes}, weight_bytes: {weight_bytes}")
        assert self.loaded_bytes == self.total_size
        assert weight_bytes == self.total_size

        return self.weight_dict
            



from transformers import FalconConfig

class Falcon_ModelLoader(ModelLoader):
    def __init__(    
        self,
        model_dir : pathlib.Path
    ) -> None:
        model_config = FalconConfig.from_pretrained(model_dir)
        weight_index_config = {}
        for child in model_dir.iterdir():
            if child.name.endswith(".index.json"):
                with open(child, "r") as f:
                    weight_index_config = json.load(f)
                break

        # model config
        self.layer_num = model_config.num_hidden_layers


        super().__init__(
            model_dir,
            weight_index_config["metadata"]["total_size"],
            weight_index_config["weight_map"]
        )


    def load_weight(self):
        self.loaded_bytes = 0
        self.weight_dict = {}

        self.load_tensor("transformer.word_embeddings.weight")
        for i in range(self.layer_num):
            self.load_tensor(f"transformer.h.{i}.self_attention.query_key_value.weight")
            self.load_tensor(f"transformer.h.{i}.self_attention.dense.weight")
            self.load_tensor(f"transformer.h.{i}.mlp.dense_h_to_4h.weight")
            self.load_tensor(f"transformer.h.{i}.mlp.dense_4h_to_h.weight")

            self.load_tensor(f"transformer.h.{i}.ln_attn.weight")
            self.load_tensor(f"transformer.h.{i}.ln_attn.bias")
            self.load_tensor(f"transformer.h.{i}.ln_mlp.weight")
            self.load_tensor(f"transformer.h.{i}.ln_mlp.bias")
        self.load_tensor("transformer.ln_f.weight")
        self.load_tensor("transformer.ln_f.bias")
        self.load_tensor("lm_head.weight")

        weight_bytes = 0
        for tensor_name in self.weight_dict:
            tensor = self.weight_dict[tensor_name]
            weight_bytes += tensor.numel() * tensor.element_size()
        
        logger.info(f"total_size: {self.total_size}, loaded_bytes: {self.loaded_bytes}, weight_bytes: {weight_bytes}")
        assert self.loaded_bytes == self.total_size
        assert weight_bytes == self.total_size

        return self.weight_dict

