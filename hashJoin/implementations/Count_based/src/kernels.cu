#include "master.h"
#include "worker.h"

/*
 * Simplified Version of Hells Join Kernel
 */
__global__
void compare_kernel_new_s(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y, int s_first, int s_end, int r_first, int r_end) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	long  global_threads = blockDim.x * gridDim.x;
	
	for (int r = idx + r_first; r + global_threads < r_end; r += global_threads){
		for (int s = s_first; s < s_end; s++){
		  	//printf("r:%d s:%d rf:%d sf:%d re:%d se:%d\n",r, s, r_first, s_first, r_end, s_end);
                        const a_t a_ = a[s] - x[r];
                        const b_t b_ = b[s] - y[r];
                        if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
		  	    //printf("k:%d %d\n",r,s);
                            output_buffer[r] = s;
			}
		}
	}
}

__global__
void compare_kernel_new_r(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y,int s_first, int s_end, int r_first, int r_end) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	long  global_threads = blockDim.x * gridDim.x;
	
	for (int r = idx + r_first; r + global_threads < r_end; r += global_threads){
		for (int s = s_first; s < s_end; s++){
		  	//printf("r:%d s:%d rf:%d sf:%d re:%d se:%d\n",r, s, r_first, s_first, r_end, s_end);
                        const a_t a_ = a[s] - x[r];
                        const b_t b_ = b[s] - y[r];
                        if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
		  	    //printf("k:%d %d\n",r,s);
                            output_buffer[r] = s;
			}
		}
	}
}

