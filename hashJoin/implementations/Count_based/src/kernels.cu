#include "master.h"
#include "worker.h"

/*
 * Simplified Version of Hells Join Kernel
 * Encodes the result as Integer
 */
__global__
void compare_kernel_new_s_legacy(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y, int s_first, int s_end, int r_first, int r_end) {
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
void compare_kernel_new_r_legacy(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y,int s_first, int s_end, int r_first, int r_end) {
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

/*
 * Compare kernel of hells join for new s tuples (batched version)
 *
 * Computes a nested loop join for s_count and r_count window elements on the gpu. Optimized for the 
 * case s_count << r_count. 
 * R column divided onto cuda threads. The degree of paralellism can be modified by the number of launch
 * threads. Every thread computes the result of multiple (r_count / 32) sets
 * of r tuples. In each thread the S column is run through per r tuple.
 * The result is encoded as bitmap of 32 bit integer.
 * 
 * @Params:
 * - The _output_buffer_ size should be (bytes): s_count * ((r_count/32)+1)
 * - The _data arrays_ (a,b,x,y) should be gives as input moved by s_first / r_first eg. &(S.a[s_first]).
 * - _r_count_ and _s_count_ give the number of tuples to process for each stream.
 *
 */
__global__ 
void compare_kernel_new_s(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
    	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;
	const int r_res_integers = r_count / 32;

	for (int r = idx; r < r_res_integers; r += global_threads){
		for (int s = 0; s < s_count; s++){
			int z = 0;
			if (r+2 < r_res_integers){
#pragma unroll
				for (int i = 0; i < 32; i++) {
					const a_t a_ = a[s] - x[r*32+i];
					const b_t b_ = b[s] - y[r*32+i];
					if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
						z = z | 1 << i;
					}
				}
			} else if (r+1 < r_res_integers){
				for (int i = 0; i < r_res_integers - r; i++) {
					const a_t a_ = a[s] - x[r*32+i];
					const b_t b_ = b[s] - y[r*32+i];
					if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
						z = z | 1 << i;
					}
				}
			}
			output_buffer[s*r_res_integers+r] = z;
		}
	}
}

/*
 * Compare kernel of hells join for new r tuples (batched version)
 *
 * SEE compare_kernel_new_s.
 */
__global__ 
void compare_kernel_new_r(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
    	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;
	const int s_res_integers = s_count / 32;

	for (int s = idx; s < s_res_integers; s += global_threads){
		for (int r = 0; r < r_count; r++){
			int z = 0;
			if (s+2 < s_res_integers){
#pragma unroll
				for (int i = 0; i < 32; i++) {
					const a_t a_ = a[s*32+i] - x[r];
					const b_t b_ = b[s*32+i] - y[r];
					if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
						z = z | 1 << i;
					}
				}
			} else if (s+1 < s_res_integers){
				for (int i = 0; i < s_res_integers - s; i++) {
					const a_t a_ = a[s*32+i] - x[r];
					const b_t b_ = b[s*32+i] - y[r];
					if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
						z = z | 1 << i;
					}
				}
			}
			output_buffer[r*s_res_integers+s] = z;
		}
	}
}
