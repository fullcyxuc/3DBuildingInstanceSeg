#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "roi_maxpool.h"

extern THCState *state;

//#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
//#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
//#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

//fp
int roi_maxpool_fp_wrapper_fast(int b, int n, int nProposal, int C,
	at::Tensor in_feats_tensor, at::Tensor in_instlable_tensor, at::Tensor out_feats_tensor, at::Tensor out_maxidx_tensor){
//    CHECK_INPUT(in_feats_tensor);
//    CHECK_INPUT(in_instlable_tensor);
    const float *in_feats = in_feats_tensor.data<float>();
    const int *in_instlable = in_instlable_tensor.data<int>();
    float *out_feats = out_feats_tensor.data<float>();
    int *out_maxidx = out_maxidx_tensor.data<int>();

    roi_maxpool_fp_kernel_launcher_fast(b, n, nProposal, C, in_feats, in_instlable, out_feats, out_maxidx);
    return 1;
}

//bp
int roi_maxpool_bp_wrapper_fast(int b, int n, int nProposal, int C,
	at::Tensor d_feats_tensor, at::Tensor out_maxidx_tensor, at::Tensor d_out_feats_tensor){
//    CHECK_INPUT(out_maxidx_tensor);
//    CHECK_INPUT(d_out_feats_tensor);
    float *d_feats = d_feats_tensor.data<float>();
    const int *out_maxidx = out_maxidx_tensor.data<int>();
    const float *d_out_feats = d_out_feats_tensor.data<float>();

    roi_maxpool_bp_kernel_launcher_fast(b, n, nProposal, C, d_feats, out_maxidx, d_out_feats);
    return 1;
}