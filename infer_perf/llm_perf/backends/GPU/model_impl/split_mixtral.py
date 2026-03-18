import os
import sys
import pathlib
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple

from accelerate import init_empty_weights
from transformers import MixtralConfig

FILE_DIR = pathlib.Path(__file__).parent.absolute()

sys.path.insert(0, str(FILE_DIR.parents[3]))
from llm_perf.backends.GPU.model_impl.modeling_mixtral import MixtralForCausalLM
from llm_perf.core.ckpt_loader import Mixtral_ModelLoader


def to_parameter(
    data : torch.Tensor, 
    dtype : torch.dtype = None
):
    if dtype is not None:
        data = data.to(dtype)
    return nn.Parameter(data, requires_grad=False)


def split(
    src : torch.Tensor, 
    mp_size : int, 
    dim : int, 
    chunks : List [int]=[]
):
    if len(chunks) == 0:
        split_arg = src.shape[dim] // mp_size
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
            if mp_size > chunks[i]:
                tensor_shape = tensor.size()[:dim] + (chunks[i], 1, other_dim_size) + tensor.size()[dim+1:]
                new_tensor_shape = tensor.size()[:dim] + (chunks[i], mp_size // chunks[i], other_dim_size) + tensor.size()[dim+1:]
                output_tensor_shape = tensor.size()[:dim] + (mp_size * other_dim_size,) + tensor.size()[dim+1:]

                tensor = tensor.view(tensor_shape)
                tensor = tensor.expand(*new_tensor_shape)
                tensor = tensor.contiguous()
                tensor = tensor.view(output_tensor_shape)

            cur_split = torch.split(tensor, tensor.shape[dim] // mp_size, dim=dim)
            output_split.append(cur_split)

        output_tensors = []
        for i in range(mp_size):
            temp_tensors = [output_split[j][i] for j in range(len(chunks))]    
            tp_tensors = torch.concat(temp_tensors, dim=dim)
            output_tensors.append(tp_tensors)

    output_tensors = [tensor.contiguous() for tensor in output_tensors]

    return output_tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mp_size", type=int, default=8, choices=[2, 4, 8])
    args = parser.parse_args()

    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(args.mp_size)

    model_path = pathlib.Path(args.model_path).absolute()
    model_config : MixtralConfig = MixtralConfig.from_pretrained(str(model_path))
    print(model_config)

    model_loader = Mixtral_ModelLoader(model_path)
    state_dict = model_loader.load_weight()

    # model_config.num_hidden_layers = 4

    p_bar = tqdm(total=model_config.num_hidden_layers, desc="split model")
    for i in range(model_config.num_hidden_layers):
        q = f"model.layers.{i}.self_attn.q_proj.weight"
        k = f"model.layers.{i}.self_attn.k_proj.weight"
        v = f"model.layers.{i}.self_attn.v_proj.weight"
        o = f"model.layers.{i}.self_attn.o_proj.weight"

        state_dict[q] = split(state_dict[q], args.mp_size, 0)
        state_dict[k] = split(state_dict[k], args.mp_size, 0)
        state_dict[v] = split(state_dict[v], args.mp_size, 0)
        state_dict[o] = split(state_dict[o], args.mp_size, 1)
        
        for j in range(model_config.num_local_experts):
            w1 = f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"
            w2 = f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"
            w3 = f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"

            state_dict[w1] = split(state_dict[w1], args.mp_size, 0)
            state_dict[w2] = split(state_dict[w2], args.mp_size, 1)
            state_dict[w3] = split(state_dict[w3], args.mp_size, 0)

        p_bar.update(1)
    p_bar.close()

    split_model_path = model_path / f"TP{args.mp_size}"
    split_model_path.mkdir(parents=True, exist_ok=True)

    with init_empty_weights():
        model = MixtralForCausalLM(model_config)
        model.eval()

    p_bar = tqdm(total=args.mp_size, desc="save model")
    for rank in range(args.mp_size):
        output_dir = split_model_path / f"device_{rank}"
        output_dir.mkdir(parents=True, exist_ok=True)

        model.model.embed_tokens.weight = to_parameter(state_dict["model.embed_tokens.weight"])
        for i in range(model_config.num_hidden_layers):
            model.model.layers[i].self_attn.q_proj.weight = to_parameter(state_dict[f"model.layers.{i}.self_attn.q_proj.weight"][rank])
            model.model.layers[i].self_attn.k_proj.weight = to_parameter(state_dict[f"model.layers.{i}.self_attn.k_proj.weight"][rank])
            model.model.layers[i].self_attn.v_proj.weight = to_parameter(state_dict[f"model.layers.{i}.self_attn.v_proj.weight"][rank])
            model.model.layers[i].self_attn.o_proj.weight = to_parameter(state_dict[f"model.layers.{i}.self_attn.o_proj.weight"][rank])
            model.model.layers[i].block_sparse_moe.gate.weight = to_parameter(state_dict[f"model.layers.{i}.block_sparse_moe.gate.weight"])
            for j in range(model_config.num_local_experts):
                model.model.layers[i].block_sparse_moe.experts[j].w1.weight = to_parameter(state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"][rank])
                model.model.layers[i].block_sparse_moe.experts[j].w2.weight = to_parameter(state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"][rank])
                model.model.layers[i].block_sparse_moe.experts[j].w3.weight = to_parameter(state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"][rank])
            model.model.layers[i].input_layernorm.weight = to_parameter(state_dict[f"model.layers.{i}.input_layernorm.weight"])
            model.model.layers[i].post_attention_layernorm.weight = to_parameter(state_dict[f"model.layers.{i}.post_attention_layernorm.weight"])
        model.model.norm.weight = to_parameter(state_dict["model.norm.weight"])
        model.lm_head.weight = to_parameter(state_dict["lm_head.weight"])

        model.save_pretrained(str(output_dir))
        p_bar.update(1)
    p_bar.close()
