#ifndef _NO_SORT_KNN_GPU_H
#define _NO_SORT_KNN_GPU_H

#include <torch/extension.h>
#include <THC/THC.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void torch_NoSortKnnLauncher(
    torch::Tensor xyz1_tensor,
    torch::Tensor xyz2_tensor,
    torch::Tensor idx_n2_tensor,
    torch::Tensor random_hw_tensor,
    int H,
    int W,
    int npoints,
    int kernel_size_H,
    int kernel_size_W,
    int K,
    bool flag_copy,
    float distance,
    int stride_h,
    int stride_w,
    torch::Tensor selected_b_idx_tensor,
    torch::Tensor selected_h_idx_tensor,
    torch::Tensor selected_w_idx_tensor,
    torch::Tensor selected_mask_tensor);

void NoSortKnnLauncher(
    int batch_size, int H, int W, int npoints, int kernel_size_H, int kernel_size_W, int K, int flag_copy, float distance, int stride_h, int stride_w, const float *xyz1, const float *xyz2, const int *idx_n2, const int *random_hw, long *selected_b_idx, long *selected_h_idx, long *selected_w_idx, float *selected_mask, cudaStream_t stream);



#endif