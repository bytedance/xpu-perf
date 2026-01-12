# LLM 常用算子

## 1. 通信算子

### 1.1 all_reduce 
- [num_tokens, hidden_size] --> [num_tokens, hidden_size]
- support bfloat16, float16, float32

### 1.2 reduce_scatter
- [num_tokens, hidden_size] --> [num_tokens // world_size, hidden_size]
- support bfloat16, float16, float32

### 1.3 all_gather
- [num_tokens // world_size, hidden_size] --> [num_tokens, hidden_size]
- support int8, int32, bfloat16, float16, float32

### 1.4 all_to_all
- [world_size, num_tokens * hidden_size // world_size] --> [world_size, num_tokens * hidden_size // world_size]
- support int8, int32, bfloat16, float16, float32


## 2. Norm & Quant 算子

### 2.1 scale_dynamic_quant
Given **hidden_states**, per token dynamic quant with smooth_scale.

---

- (in) hidden_states
    - [num_tokens // sp_size, hidden_size]
    - bfloat16

- (in) smooth_scale
    - [hidden_size, ]
    - float32
---

- (out) quant_tokens
    - [num_tokens // sp_size, hidden_size]
    - int8, float8

- (out) per_token_scale
    - [num_tokens // sp_size, ]
    - float32

---

- (attr) dtype, support {bfloat16}
- (attr) dst_dtype, support {int8, float8}
- (attr) num_tokens
- (attr) hidden_size
- (attr) sp_size




### 2.2 head_rms_norm
For information about rms_norm, refer to [rms_norm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html#torch.nn.RMSNorm). 

head_rms_norm is a variant of rms_norm, which only norm the heads specified by **norm_head_start**, totally **norm_head_num** heads per token will be normed, all heads use the same weight.

The operation is in-place.

--- 

- (in) token_data:
    - [num_tokens // sp_size, total_head_num, head_dim]
    - bfloat16

- (in) norm_weight:
    - [head_dim, ]
    - float32

--- 

- (attr) dtype, support {bfloat16}
- (attr) num_tokens
- (attr) head_dim
- (attr) sp_size
- (attr) total_head_num, depends on model structure
- (attr) norm_head_start, depends on model structure
- (attr) norm_head_num, depends on model structure


### 2.3 head_rms_norm_dynamic_quant
Norm on all heads, and then per token dynamic quant.

The operation is out-place.

--- 

- (in) token_data:
    - [num_tokens, head_num, head_dim]
    - bfloat16

- (in) norm_weight:
    - [head_dim, ]
    - float32

- (in) smooth_scale:
    - [head_num, head_dim, ]
    - float32

--- 

- (out) quant_tokens
    - [num_tokens, head_num * head_dim]
    - int8, float8

- (out) per_token_scale
    - [num_tokens, ]
    - float32

---

- (attr) dtype, support {bfloat16}
- (attr) dst_dtype, support {int8, float8}
- (attr) num_tokens
- (attr) head_num
- (attr) head_dim
- (attr) sp_size




### 2.4 add_rms_norm_dynamic_quant
Given **hidden_states** and **residual (optional)**, add **residual** (if exists), rms_norm and dynamic_quant on **hidden_size** dim.

--- 

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16

- (in) smooth_scale
    - [hidden_size, ]
    - float32
    - fuse norm_weight and smooth_scale

---
**optional**

- (in) residual
    - [num_tokens, hidden_size]
    - bfloat16

--- 

- (out) quant_tokens
    - [num_tokens, hidden_size]
    - int8, float8

- (out) per_token_scale
    - [num_tokens, ]
    - float32

---
**optional**

- (out) output
    - [num_tokens, hidden_size]
    - bfloat16
    - "res": hidden_states + residual
    - "norm": rms_norm(hidden_states + residual)

--- 

- (attr) dtype, support {bfloat16}
- (attr) dst_dtype, support {int8, float8}
- (attr) num_tokens
- (attr) hidden_size
- (attr) sp_size
- (attr) add_residual, support {True, False}
- (attr) output_mode, support {"none", "res", "norm"}



## 3. Attention & rope & kvcache 算子

### fundamental concepts
1. packed_qkv: 
    - multi seqs are packed in one tensor, shape is [num_tokens, q_head_num + 2 * kv_head_num, head_dim], including q/k/v.

2. Attention Test Mode allows different test modes to be implemented with different kernels.
    - "prefill": single or mutiple seqs with different q_len/cache_len.
    - "decode": multiple seqs with same q_len and different cache_len.

3. how to get q_lens/cache_lens
    - For testing convenience, we predefine batch_size/q_len/cache_len to generate q_lens/cache_lens, which are further used to produce q_lens/cache_lens/kv_lens as well as accum_q_lens/accum_kv_lens. These parameters enable different vendors to read the corresponding data based on their respective implementations.
    - To further facilitate online traffic replay, we have also defined a parameter type called batch_llm, which directly specifies the q_len and cache_len for each sequence within a batch.
    

--- 

### 3.1 rotary_embedding
Give **packed_qkv**, rotary_embedding on **rope_dim** dim, which is part of **head_dim**.

Only **Q/K** need rotary_embedding, **V** not.

This operator is in-place.

--- 

- (in) packed_qkv
    - [num_tokens, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16

- (in) q_lens
    - [num_tokens, ]
    - int32

- (in) accum_q_lens
    - [num_tokens + 1, ]
    - int32

- (in) cache_lens
    - [num_tokens, ]
    - int32

- (in) cos
    - [max_kv_len, rope_dim]
    - bfloat16

- (in) sin
    - [max_kv_len, rope_dim]
    - bfloat16



---
for **llm** arg_type, used to generate q_lens/cache_lens/kv_lens/accum_q_lens/accum_kv_lens.

- (attr) batch_size
- (attr) q_len
- (attr) cache_len
- (attr) random_seed, default is 42
- (attr) unbalance_q, default is 0
- (attr) unbalance_cache, default is 0

---
for **batch_llm** arg_type, used to generate q_lens/cache_lens/kv_lens/accum_q_lens/accum_kv_lens.

- (attr) q_lens
- (attr) cache_lens

---

- (attr) dtype, support {bfloat16}
- (attr) q_head_num
- (attr) kv_head_num
- (attr) head_dim
- (attr) rope_offset, 0 <= rope_offset < head_dim
- (attr) rope_dim



### 3.2 store_kv_cache
Given **packed_qkv**, store k/v to kv_cache, kv_cache is **linear** or **paged**, depending on block_size.

When q_cache/v_cache is int8 or float8, we use k_scale/v_scale to quantize k/v respectively, currently support static quantization, where k_scale/v_scale is precomputed and passed as input.

This operator is inplace.

--- 

- (in) packed_qkv
    - [num_tokens, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16

- (in) kv_ids
    - [batch_size, ]
    - int32

- (in) q_lens
    - [num_tokens, ]
    - int32

- (in) accum_q_lens
    - [num_tokens + 1, ]
    - int32

- (in) cache_lens
    - [num_tokens, ]
    - int32

- (in) k_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - bfloat16 / int8 / float8

- (in) v_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - bfloat16 / int8 / float8

---
**optional** only necessary for paged kv_cache

- (in) block_table
    - [max_batch_size, max_seq_len]
    - int32

---
**optional** for static quant

- (in) k_scale
    - [kv_head_num, head_dim]
    - float32

- (in) v_scale
    - [kv_head_num, head_dim]
    - float32


---
for **llm** arg_type, used to generate q_lens/cache_lens/kv_lens/accum_q_lens/accum_kv_lens.

- (attr) batch_size
- (attr) q_len
- (attr) cache_len
- (attr) random_seed, default is 42
- (attr) q_unbalance_rate, default is 0
- (attr) kv_unbalance_rate, default is 0

---
for **batch_llm** arg_type, used to generate q_lens/cache_lens/kv_lens/accum_q_lens/accum_kv_lens.

- (attr) q_lens
- (attr) cache_lens

---

- (attr) dtype, support {bfloat16}
- (attr) cache_dtype, support {int8, float8}
- (attr) q_head_num
- (attr) kv_head_num
- (attr) head_dim
- (attr) block_size, default is 0, which means linear kv_cache, otherwise paged kv_cache




### 3.3 dequant_kv_cache
Given quantized **k_cache/v_cache**, dequantize them to bfloat16 while preserving the original format (linear/paged).

This operator is out-place.

---

- (in) k_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - int8 / float8

- (in) v_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - int8 / float8

- (in) k_scale
    - [kv_head_num, head_dim]
    - float32

- (in) v_scale
    - [kv_head_num, head_dim]
    - float32

---
**optional** only necessary for paged kv_cache

- (in) block_table
    - [max_batch_size, max_seq_len]
    - int32


---

- (in) dequant_k_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - bfloat16

- (in) dequant_v_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - bfloat16

---
for **llm** arg_type, used to generate q_lens/cache_lens/kv_lens/accum_q_lens/accum_kv_lens.

- (attr) batch_size
- (attr) q_len
- (attr) cache_len
- (attr) random_seed, default is 42
- (attr) q_unbalance_rate, default is 0
- (attr) kv_unbalance_rate, default is 0

---
for **batch_llm** arg_type, used to generate q_lens/cache_lens/kv_lens/accum_q_lens/accum_kv_lens.

- (attr) q_lens
- (attr) cache_lens

---

- (attr) dtype, support {bfloat16}
- (attr) cache_dtype, support {int8, float8}
- (attr) kv_head_num
- (attr) head_dim
- (attr) block_size, default is 0, which means linear kv_cache, otherwise paged kv_cache





### 3.4 flash_attention
Given q, k_cache, v_cache, do flash_attention, The following cases need to be considered: 
1. q should be bfloat16. if quantization needed, use a separate kernel to quantize q, or integrate the quantization process into the flash_attention kernel.
2. The data types of k_cache and v_cache are determined by cache_dtype, while their format may be either linear or paged. The specific input of quantization parameters depends on the vendor's quantization algorithm.
3. **compute_dtype** determines the expected data type for **q\*k** and **p\*v** computations in flash_attention, and serves as the basis for calculating MFU using the corresponding nominal computing power. However, vendors can flexibly adjust the actual data types used for q*k and p*v based on implementation requirements.
4. Since the implementation of flash_attention is well-known and varies significantly across vendors, we only specify the reference inputs, outputs, and expected configurations. Vendors are free to adopt their own implementation methods, such as combining multiple kernels or using different kernels for different parameters.

---

- (in) q
    - [num_tokens, q_head_num, head_dim]  
    - bfloat16
    - **sliced from packed_qkv**

- (in) k_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - bfloat16 / int8 / float8

- (in) v_cache
    - **linear**: [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - **paged**: [max_block_size, kv_head_num, block_size, head_dim]
    - bfloat16 / int8 / float8

- (in) q_lens
    - [num_tokens, ]
    - int32

- (in) accum_q_lens
    - [num_tokens + 1, ]
    - int32

- (in) kv_lens
    - [num_tokens, ]
    - int32

- (in) accum_kv_lens
    - [num_tokens + 1, ]
    - int32

---
**optional** only necessary for paged kv_cache

- (in) block_table
    - [max_batch_size, max_seq_len]
    - int32

---

**optional** for static quant

- (in) key_scale
    - [kv_head_num, head_dim]
    - float32

- (in) value_scale
    - [kv_head_num, head_dim]
    - float32

---
**optional** for dynamic quant
- (in) k_scale
    - [max_batch_size, max_seq_len]
    - float32

- (in) v_scale
    - [max_batch_size, max_seq_len]
    - float32


---

- (out) out
    - [num_tokens, q_head_num, head_dim]
    - bfloat16

---

for **llm** arg_type
- (attr) batch_size
- (attr) q_len
- (attr) cache_len
- (attr) random_seed
- (attr) q_unbalance_rate
- (attr) kv_unbalance_rate

---

for **batch_llm** arg_type
- (attr) q_lens
- (attr) cache_lens

--- 

- (attr) dtype, support {bfloat16}
- (attr) cache_dtype, support {bfloat16, int8, float8}
- (attr) compute_dtype, support {bfloat16, int8, float8}
- (attr) quant_mode, support {"static", "dynamic_per_token", "dynamic_per_block"}
    - static: [kv_head_num, head_dim], **[max_seq_len, ]** share 1 scale.
    - dynamic_per_token: [max_batch_size, max_seq_len], **[kv_head_num, head_dim]** share 1 scale.
    - dynamic_per_block: [kv_head_num, kv_len // seq_quant_block_size, head_dim // head_dim_quant_block_size], **[quant_block_size, quant_block_size]** share 1 scale
- (attr) q_head_num
- (attr) kv_head_num
- (attr) head_dim
- (attr) block_size, default is 0, which means linear kv_cache, otherwise paged kv_cache
- (attr) **optional**, seq_quant_block_size, only used for **dynamic_per_block** quant, for example 16
- (attr) **optional**, head_dim_quant_block_size, only used for **dynamic_per_block** quant, for example 16


## 4. gemm & group_gemm & moe_ops

### fundamental concepts
1. num_tokens/sp_size:
    - Considering that the parameters of different operators may change under different parallel methods, all operators uniformly adopt the unified num_tokens parameter, with only the impact of sp_size on num_tokens taken into account.

2. ep_size/
    - split num_experts on multiple devices



### 4.1 moe_gating_gemm
Gemm kernel specialized for moe gating, small N, which used to be 32/64/128/256, need to split K, and output dtype is float32.

---

- (in) hidden_states
    - [num_tokens // sp_size, hidden_size]
    - bfloat16

- (in) gating_weight
    - [hidden_size, num_experts]
    - bfloat16

---

- (out) gating_output
    - [num_tokens // sp_size, num_experts]
    - float32

--- 

- (attr) dtype, supports {bfloat16, float32}
- (attr) dst_dtype, supports {float32}
- (attr) num_tokens
- (attr) hidden_size
- (attr) num_experts
- (attr) trans_w, supports {True, False}, default is False
- (attr) sp_size


### 4.2 quant_matmul

Support:

- M = num_tokens
- K = hidden_size
- N = new_hidden_size

act: [M, K], per token quant, dynamic quant with scale [M]

weight: [K, N], per channel quant, with scale [N]

--- 

- (in) hidden_states
    - [num_tokens // sp_size, hidden_size]
    - int8, float8

- (in) per_token_scale
    - [num_tokens, ]
    - float32

- (in) expert_weight
    - [hidden_size, new_hidden_size]
    - int8, float8

- (in) expert_scale
    - [new_hidden_size, ]
    - float32

---

- (out) y
    - [num_tokens // sp_size, new_hidden_size]
    - bfloat16



### 4.3 moe_quant_group_gemm
In fact, it is quant_group_gemm. Considering practical MoE scenarios, we compute the M dimension for each problem in the current group_gemm by leveraging a series of parameters such as num_tokens and ep_size.

For num_experts, use EP (experts parallel)
- Each rank will pre-allocated num_experts // ep_size experts. 
- We assume that the num_tokens * topk tokens are evenly distributed across all experts.

---

- (in) scatter_tokens
    - [num_tokens // sp_size * topk // ep_size, hidden_size]
    - int8

- (in) scatter_per_token_scale
    - [num_tokens // sp_size * topk // ep_size, ]
    - float32

- (in) experts_weight
    - [num_experts // ep_size, hidden_size, new_hidden_size]
    - int8

- (in) experts_scale
    - [num_experts // ep_size, new_hidden_size]
    - float32

- (in) experts_token_offset
    - [num_experts // ep_size, ]
    - int32

- (in) experts_token_count
    - [num_experts // ep_size, ]
    - int32

---

- (out) y
    - [num_tokens * topk // ep_size, new_hidden_size]
    - bfloat16






### 4.4 moe_softmax_topk
Select topk experts for each token, expert_weights need to be normalized if softmax first.

--- 

- (in) gating_output
    - [num_tokens // sp_size, num_experts]
    - float32

---

- (out) selected_experts
    - [num_tokens // sp_size, topk]
    - int32

- (out) moe_weights
    - [num_tokens // sp_size, topk]
    - float32

---

- (attr) dtype, supports {float32}
- (attr) num_tokens
- (attr) num_experts
- (attr) topk
- (attr) compute_mode, supports {pre-softmax, post-softmax}
- (attr) sp_size



### 4.5 moe_scatter_dynamic_quant
For M experts, use EP (experts parallel)
- Each rank will pre-allocated num_experts // ep_size experts. 
- We assume that the num_tokens * topk tokens are evenly distributed across all experts.
- Exclude sp_size scenarios.
- Per token dynamic quant for each problem.

---

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16

- (in) experts_smooth_scale
    - [num_experts // ep_size, hidden_size]

- (in) selected_experts
    - [num_tokens, topk]
    - int32

- (in) moe_weights
    - [num_tokens, topk]
    - float32

---

- (out) scatter_tokens
    - [num_tokens * topk // ep_size, new_hidden_size]
    - int8, float8

- (out) scatter_per_token_scale
    - [num_tokens * topk // ep_size, ]
    - float32

- (out) scatter_token_id
    - [num_tokens * topk // ep_size, ]
    - int32

- (out) scatter_token_weight
    - [num_tokens * topk // ep_size, ]
    - float32

- (out) experts_token_count
    - [num_experts // ep_size, ]
    - int32

- (out) experts_token_offset
    - [num_experts // ep_size, ]
    - int32

--- 

- (attr) dtype, supports {bfloat16}
- (attr) dst_dtype, supports {int8, float8}
- (attr) num_tokens
- (attr) hidden_size
- (attr) num_experts
- (attr) topk
- (attr) ep_size/ep_rank, used to calc expert_start/expert_end



### 4.6 moe_swiglu_dynamic_quant
For M experts, use EP (experts parallel)
- Each rank will pre-allocated num_experts // ep_size experts. 
- We assume that the num_tokens * topk tokens are evenly distributed across all experts.
- Exclude sp_size scenarios.
- Per token dynamic quant for each problem.

---

- (in) scatter_tokens
    - [num_tokens * topk // ep_size, hidden_size * 2]
    - bfloat16

- (in) experts_smooth_scale
    - [num_experts // ep_size, hidden_size]
    - float32

- (in) experts_token_count
    - [num_experts // ep_size, ]
    - int32

- (in) experts_token_offset
    - [num_experts // ep_size, ]
    - int32

---

- (out) quant_tokens
    - [num_tokens * topk // ep_size, hidden_size]
    - int8, float8

- (out) per_token_scale
    - [num_tokens * topk // ep_size, ]
    - float32

---

- (attr) dtype, supports {bfloat16}
- (attr) dst_dtype, supports {int8, float8}
- (attr) num_tokens
- (attr) hidden_size
- (attr) num_experts
- (attr) topk
- (attr) ep_size/ep_rank, used to calc expert_start/expert_end



### 4.7 swiglu_dynamic_quant
Given hidden_states [num_tokens // sp_size, hidden_size * 2], 
output is [num_tokens // sp_size, hidden_size]

---

- (in) hidden_states
    - [num_tokens // sp_size, hidden_size * 2]
    - bfloat16

- (in) smooth_scale
    - [hidden_size, ]
    - float32

--- 

- (out) quant_tokens
    - [num_tokens // sp_size, hidden_size]
    - int8, float8

- (out) per_token_scale
    - [num_tokens // sp_size, ]
    - float32

---

- (attr) dtype, supports {bfloat16}
- (attr) dst_dtype, supports {int8, float8}
- (attr) num_tokens
- (attr) hidden_size





### 4.8 moe_gather
For M experts, use EP (experts parallel)
- Each rank will pre-allocated num_experts // ep_size experts. 
- We assume that the num_tokens * topk tokens are evenly distributed across all experts.
- sp_size will only be used on shared_experts residual_tokens.

**gathered_tokens** tensor is pre-allocated, and filled with 0.0. The gather operation (actually index_add) is done in-place.

---

- (in) scatter_tokens
    - [num_tokens * topk // ep_size, hidden_size]
    - bfloat16

- (in) scatter_token_id
    - [num_tokens * topk // ep_size, ]
    - int32

- (in) scatter_token_weight
    - [num_tokens * topk // ep_size, ]
    - float32

--- 
optional

- (in) residual_tokens
    - [num_tokens // sp_size, hidden_size]
    - bfloat16
    - muls res_scale and add to **gathered_tokens** depending on sp_size/sp_rank

---

- (out) gathered_tokens
    - [num_tokens, hidden_size]
    - bfloat16

---

- (attr) dtype, supports {bfloat16}
- (attr) num_tokens
- (attr) hidden_size
- (attr) num_experts
- (attr) topk
- (attr) ep_size/ep_rank, used to calc expert_start/expert_end
- (attr) sp_size/sp_rank, used to calc residual_start/residual_end
- (attr) res_scale

