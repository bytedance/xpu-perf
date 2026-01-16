#pragma once

#include <torch/extension.h>

void topk_softmax(torch::Tensor &topk_weights, torch::Tensor &topk_indices,
                  torch::Tensor &token_expert_indices,
                  torch::Tensor &gating_output,
                  bool need_renorm);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor token_nums,
                          torch::Tensor num_tokens_post_pad);

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);

void rms_norm(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor &input, torch::Tensor &residual,
                        torch::Tensor &weight, double epsilon);

// ck kernel
void layernorm2d(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight, torch::Tensor &bias,
                 double epsilon);

void wvSpltK(at::Tensor &in_a, at::Tensor &in_b, at::Tensor &out_c,
             const int64_t N_in, const int64_t CuCount);

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                      torch::Tensor &key, int64_t head_size,
                      torch::Tensor &cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                              torch::Tensor &key, int64_t head_size,
                              torch::Tensor &cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor &cos_sin_cache_offsets);

void moe_sum(torch::Tensor &input, torch::Tensor &output);

// all reduce
using fptr_t = int64_t;
fptr_t init_custom_ar(torch::Tensor &meta, torch::Tensor &rank_data,
                      const std::vector<std::string> &handles,
                      const std::vector<int64_t> &offsets, int64_t rank,
                      bool full_nvlink);
void all_reduce_reg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out);
void all_reduce_unreg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &reg_buffer,
                      torch::Tensor &out);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, torch::Tensor &t,
                     const std::vector<std::string> &handles,
                     const std::vector<int64_t> &offsets);
std::tuple<torch::Tensor, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::string> &handles,
                            const std::vector<std::vector<int64_t>> &offsets);
#ifdef USE_ROCM
torch::Tensor allocate_meta_buffer(int64_t size);
torch::Tensor get_meta_buffer_ipc_handle(torch::Tensor &inp);
#endif