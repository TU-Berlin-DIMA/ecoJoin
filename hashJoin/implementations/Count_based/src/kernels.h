#ifndef KERNELS_H
#define KERNELS_H

#include "worker.h"
#include "master.h"
#include "data.h"

/*
 * Simplified Version of Hells Join Kernel
 */
__global__
void compare_kernel_new_s(int *output_buffer, int* a, float* b, int* x, float* y, int s_first, int s_end, int r_first, int r_end);
__global__
void compare_kernel_new_r(int *output_buffer, int* a, float* b, int* x, float* y,int s_first, int s_end, int r_first, int r_end);

#endif /* KERNELS_H */
