#ifndef _ROI_MAXPOOL_H
#define _ROI_MAXPOOL_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

// fp
int roi_maxpool_fp_wrapper_fast(int b, int n, int nProposal, int C,
	at::Tensor in_feats, at::Tensor in_instlable, at::Tensor out_feats, at::Tensor out_maxidx);  //封装来说python接口的数据类型


void roi_maxpool_fp_kernel_launcher_fast(int b, int n, int nProposal, int C,
	const float *in_feats, const int *in_instlable, float *out_feats, int *out_maxidx);  //调用cu方法

// bp
int roi_maxpool_bp_wrapper_fast(int b, int n, int nProposal, int C,
	at::Tensor d_feats, at::Tensor out_maxidx, at::Tensor d_out_feats);

void roi_maxpool_bp_kernel_launcher_fast(int b, int n, int nProposal, int C,
	float *d_feats, const int *out_maxidx, const float *d_out_feats);


#endif
