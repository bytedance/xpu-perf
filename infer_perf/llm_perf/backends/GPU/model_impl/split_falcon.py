import os
import sys
import pathlib
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple

from accelerate import init_empty_weights
from transformers import FalconConfig

FILE_DIR = pathlib.Path(__file__).parent.absolute()

sys.path.insert(0, str(FILE_DIR.parents[3]))
from llm_perf.backends.GPU.model_impl.modeling_falcon import FalconForCausalLM
from llm_perf.core.ckpt_loader import Falcon_ModelLoader


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
    model_config = FalconConfig.from_pretrained(str(model_path))
    print(model_config)

    model_loader = Falcon_ModelLoader(model_path)
    state_dict = model_loader.load_weight()

    # model_config.num_hidden_layers = 4

    p_bar = tqdm(total=model_config.num_hidden_layers, desc="split model")
    for i in range(model_config.num_hidden_layers):
        attn_qkv = f"transformer.h.{i}.self_attention.query_key_value.weight"
        attn_dense = f"transformer.h.{i}.self_attention.dense.weight"

        dense_h_to_4h = f"transformer.h.{i}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h = f"transformer.h.{i}.mlp.dense_4h_to_h.weight"

        state_dict[attn_qkv] = split(
            state_dict[attn_qkv], args.mp_size, 
            dim=0
        )
        state_dict[attn_dense] = split(
            state_dict[attn_dense], args.mp_size, 
            dim=1
        )
        state_dict[dense_h_to_4h] = split(
            state_dict[dense_h_to_4h], args.mp_size, 
            dim=0
        )
        state_dict[dense_4h_to_h] = split(
            state_dict[dense_4h_to_h], args.mp_size, 
            dim=1
        )

        p_bar.update(1)
    p_bar.close()

    split_model_path = model_path / f"TP{args.mp_size}"
    split_model_path.mkdir(parents=True, exist_ok=True)

    with init_empty_weights():
        model = FalconForCausalLM(model_config)
        model.eval()


    p_bar = tqdm(total=args.mp_size, desc="save model")
    for i in range(args.mp_size):    
        output_dir = split_model_path / f"device_{i}"
        output_dir.mkdir(parents=True, exist_ok=True)

        model.transformer.word_embeddings.weight = to_parameter(state_dict["transformer.word_embeddings.weight"])
        for j in range(model_config.num_hidden_layers):
            model.transformer.h[j].self_attention.query_key_value.weight = to_parameter(state_dict[f"transformer.h.{j}.self_attention.query_key_value.weight"][i])
            model.transformer.h[j].self_attention.dense.weight = to_parameter(state_dict[f"transformer.h.{j}.self_attention.dense.weight"][i])
            model.transformer.h[j].mlp.dense_h_to_4h.weight = to_parameter(state_dict[f"transformer.h.{j}.mlp.dense_h_to_4h.weight"][i])
            model.transformer.h[j].mlp.dense_4h_to_h.weight = to_parameter(state_dict[f"transformer.h.{j}.mlp.dense_4h_to_h.weight"][i])

            model.transformer.h[j].ln_attn.weight = to_parameter(state_dict[f"transformer.h.{j}.ln_attn.weight"])
            model.transformer.h[j].ln_attn.bias = to_parameter(state_dict[f"transformer.h.{j}.ln_attn.bias"])
            model.transformer.h[j].ln_mlp.weight = to_parameter(state_dict[f"transformer.h.{j}.ln_mlp.weight"])
            model.transformer.h[j].ln_mlp.bias = to_parameter(state_dict[f"transformer.h.{j}.ln_mlp.bias"])
        model.transformer.ln_f.weight = to_parameter(state_dict["transformer.ln_f.weight"])
        model.transformer.ln_f.bias = to_parameter(state_dict["transformer.ln_f.bias"])
        model.lm_head.weight = to_parameter(state_dict["lm_head.weight"])

        model.save_pretrained(str(output_dir))
        p_bar.update(1)
    p_bar.close()

