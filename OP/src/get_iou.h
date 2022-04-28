#ifndef _GET_IOU_H
#define _GET_IOU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int get_iou_wrapper_fast(int b, int n, at::Tensor proposals_idx_tensor, int proposal_num, at::Tensor instance_labels_tensor,
                         at::Tensor instance_nums_tensor, int max_instance_num, at::Tensor proposals_iou_tensor);  //封装来说python接口的数据类型


void get_iou_kernel_launcher_fast(int b, int n, const int * proposals_idx, int proposal_num, const int * instance_labels,
                                  const int * instance_nums, int max_instance_num, float * proposals_iou);  //调用.cu方法

#endif
