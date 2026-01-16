#pragma once

#include <torch/all.h>
void dynamic_scaled_int8_quant(
		torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,c10::optional<torch::Tensor> const& azp);
