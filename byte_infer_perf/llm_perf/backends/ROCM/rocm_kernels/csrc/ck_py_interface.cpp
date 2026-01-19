/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: ck_py_interface.cpp
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-10-24 12:11:58
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-10-24 14:39:16
 * @Description: This is description.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(FIND_CK)
#include "layernorm2d_fwd.hpp"

// common utility functions
#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", torch::kFloat)             \
    F("fp16", torch::kHalf)              \
    F("bf16", torch::kBFloat16)

inline std::string torchDTypeToStr(caffe2::TypeMeta dtype)
{
#define TYPE_CASE(type, torch_type) \
    case torch_type:                \
    {                               \
        return type;                \
    }

    switch (dtype.toScalarType())
    {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        throw std::runtime_error("Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}

void layernorm2d(torch::Tensor &out,    // [hidden_size]
                 torch::Tensor &input,  // [hidden_size]
                 torch::Tensor &weight, // [hidden_size]
                 torch::Tensor &bias,   // [hidden_size]
                 double epsilon)
{
    // auto dtype = input.dtype();
    // TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
    //             "FlashAttention only support fp16 and bf16 data type");

    std::string dtype_str = torchDTypeToStr(input.dtype());
    int n = input.size(-1);
    int m = input.numel() / n;
    int stride = n;
    bool SaveMeanVar = false;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layernorm2d_fwd({dtype_str, SaveMeanVar},
                    {input.data_ptr(),
                     weight.data_ptr(),
                     bias.data_ptr(),
                     out.data_ptr(),
                     nullptr,
                     nullptr,
                     static_cast<float>(epsilon),
                     m,
                     n,
                     stride},
                    {stream});
}
#endif //
