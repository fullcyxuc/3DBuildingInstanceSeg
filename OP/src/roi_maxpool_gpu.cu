#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "roi_maxpool.h"
#include "cuda_utils.h"


__global__ void roi_maxpool_fp_kernel_fast(int b, int n, int nProposal, int C, const float *__restrict__ in_feats,
    const int *__restrict__ in_instlable, float *__restrict__ out_feats, int *__restrict__ out_maxidx) {
    int bs_idx = blockIdx.y;  // base idx of batch
    int ft_idx = bs_idx * n * C;  // base idx of feature in current batch
    int lb_idx = bs_idx * n;  // base idx of instance label in current batch
    int o_idx = bs_idx * nProposal * C; // base idx of output

    for(int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x){
        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            int argmax_idx = -1;
            float max_val = -1e50;


            for(int i = 0; i < n; i++){
                if(in_instlable[lb_idx + i] == pp_id){  //判断每个点是不是当前线程处理的实例点
                    if(in_feats[ft_idx + i * C + plane] > max_val){
                        max_val = in_feats[ft_idx + i * C + plane];
                        argmax_idx = i;
                    }
                }
            }
            out_feats[o_idx + pp_id * C + plane] = max_val;
            out_maxidx[o_idx + pp_id * C + plane] = argmax_idx;
        }
    }
}

//input: in_feats (B, N, C) float
//input: in_inslable (B, N) int
//output: out_feats (B, nProposal, C) float
//output: out_maxidx (B, nProposal, C) int
void roi_maxpool_fp_kernel_launcher_fast(int b, int n, int nProposal, int C, const float *__restrict__ in_feats,
    const int *__restrict__ in_instlable, float *__restrict__ out_feats, int *__restrict__ out_maxidx) {

    cudaError_t err;

    dim3 blocks(std::min(nProposal, (int)1024), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(std::min(THREADS_PER_BLOCK, 32));

    roi_maxpool_fp_kernel_fast<<<blocks, threads>>>(b, n, nProposal, C, in_feats, in_instlable, out_feats, out_maxidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void roi_maxpool_bp_kernel_fast(int b, int n, int nProposal, int C, float *__restrict__ d_feats,
    const int *__restrict__ out_maxidx, const float *__restrict__ d_out_feats) {
        int bs_idx = blockIdx.y;
        int df_idx = bs_idx * n * C;  // the base idx of the output derivative feature
        int pf_idx = bs_idx * nProposal * C;  // the base idx of the input proposal feature

        for(int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x){
            for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
                int argmax_idx = out_maxidx[pf_idx + pp_id * C + plane];
                atomicAdd(&d_feats[df_idx + argmax_idx * C + plane], d_out_feats[pf_idx + pp_id * C + plane]);
            }
        }
}


void roi_maxpool_bp_kernel_launcher_fast(int b, int n, int nProposal, int C, float *__restrict__ d_feats,
    const int *__restrict__ out_maxidx, const float *__restrict__ d_out_feats) {

    cudaError_t err;

    dim3 blocks(std::min(nProposal, (int)1024), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(std::min(THREADS_PER_BLOCK, 32));


    roi_maxpool_bp_kernel_fast<<<blocks, threads>>>(b, n, nProposal, C, d_feats, out_maxidx, d_out_feats);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}