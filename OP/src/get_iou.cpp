#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "get_iou.h"

extern THCState *state;

//fp
int get_iou_wrapper_fast(int b, int n, at::Tensor proposals_idx_tensor, int proposal_num, at::Tensor instance_labels_tensor,
                         at::Tensor instance_nums_tensor, int max_instance_num, at::Tensor proposals_iou_tensor){
    const int *proposals_idx = proposals_idx_tensor.data<int>();
    const int *instance_labels = instance_labels_tensor.data<int>();
    const int *instance_nums = instance_nums_tensor.data<int>();
    float *proposals_iou = proposals_iou_tensor.data<float>();

    get_iou_kernel_launcher_fast(b, n, proposals_idx, proposal_num, instance_labels, instance_nums, max_instance_num,
                                 proposals_iou);
    return 1;
}

