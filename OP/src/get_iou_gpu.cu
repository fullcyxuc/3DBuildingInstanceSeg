#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "get_iou.h"
#include "cuda_utils.h"


__global__ void get_iou_kernel_fast(int b, int n, const int * proposals_idx, int proposal_num, const int * instance_labels,
                                    const int * instance_nums, int max_instance_num, float * proposals_iou) {
    int bs_idx = blockIdx.y;  // base idx of batch

    for(int pp_id = blockIdx.x; pp_id < proposal_num; pp_id += gridDim.x){
        for(int inst_id = threadIdx.x; inst_id < instance_nums[bs_idx]; inst_id += blockDim.x){
            int cur_proposal_pnum = 0;  // point number of current proposal
            int cur_instance_pnum = 0;  // point number of current instance
            int cur_intersection = 0;   // the intersection between current proposal and instance
            int cur_idx = bs_idx * n;   // the current base idx

            for(int i = 0; i < n; i++){
                int cur_propsal_label = proposals_idx[cur_idx + i];
                int cur_instance_label = instance_labels[cur_idx + i];
                if(cur_propsal_label == pp_id)
                    cur_proposal_pnum++;
                if(cur_instance_label == inst_id)
                    cur_instance_pnum++;
                if(cur_propsal_label == pp_id && cur_instance_label == inst_id)
                    cur_intersection++;
            }

            float cur_iou = 1.0 * cur_intersection / (cur_proposal_pnum + cur_instance_pnum - cur_intersection + 1e-5);
            proposals_iou[bs_idx * proposal_num * max_instance_num + pp_id * max_instance_num + inst_id] = cur_iou;
        }
    }
}

//input: proposals_idx (B, N) int
//input: proposal_num  int
//input: instance_labels (B, N) int
//input: instance_nums (B, ) int
//input: max_instance_num int
//output: proposals_iou (B, Np, Ninst)
void get_iou_kernel_launcher_fast(int b, int n, const int * proposals_idx, int proposal_num, const int * instance_labels,
                                  const int * instance_nums, int max_instance_num, float * proposals_iou) {

    cudaError_t err;

    dim3 blocks(std::min(proposal_num, (int)1024), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(std::min(max_instance_num, 32));

    get_iou_kernel_fast<<<blocks, threads>>>(b, n, proposals_idx, proposal_num, instance_labels, instance_nums,
                                                        max_instance_num, proposals_iou);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
