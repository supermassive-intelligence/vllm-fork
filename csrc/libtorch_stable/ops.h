#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#ifndef USE_ROCM
// Note: tensor params are passed by value (not by reference) because the
// PyTorch 2.10 stable ABI's `torch::stable::detail::to<T>` requires `T` to be
// trivially copyable, which excludes both `Tensor&` and `const Tensor&`.
// Schema-level mutability is still conveyed via the `Tensor!` marker in the
// op definition string.
torch::stable::Tensor permute_cols(torch::stable::Tensor A,
                                   torch::stable::Tensor perm);

void per_token_group_quant_fp8(torch::stable::Tensor input,
                               torch::stable::Tensor output_q,
                               torch::stable::Tensor output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0,
                               bool dummy_is_scale_transposed,
                               bool dummy_is_tma_aligned);

// Fused activation quantisation + DeepGEMM-compatible UE8M0-packed scales.
void per_token_group_quant_8bit_packed(torch::stable::Tensor input,
                                       torch::stable::Tensor output_q,
                                       torch::stable::Tensor output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit);

void per_token_group_quant_int8(torch::stable::Tensor input,
                                torch::stable::Tensor output_q,
                                torch::stable::Tensor output_s,
                                int64_t group_size, double eps, double int8_min,
                                double int8_max);
#endif
