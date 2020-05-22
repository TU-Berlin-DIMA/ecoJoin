#include "master.h"
#include "worker.h"



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

__global__ 
void compare_kernel_new_s_range(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
    	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;
	const int r_res_integers = r_count / 32;

	for (int r = idx; r < r_res_integers; r += global_threads){
		for (int s = 0; s < s_count; s++){
			int z = 0;
			if (r+2 < r_res_integers){
#pragma unroll
				for (int i = 0; i < 32; i++) {
					if (a[s] + b[s] == x[r*32+i] + y[r*32+i]){
						z = z | 1 << i;
					}
				}
			} else if (r+1 < r_res_integers){
				for (int i = 0; i < r_res_integers - r; i++) {
					if (a[s] + b[s] == x[r*32+i] + y[r*32+i]){
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
void compare_kernel_new_r_range(int *output_buffer, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
    	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;
	const int s_res_integers = s_count / 32;

	for (int s = idx; s < s_res_integers; s += global_threads){
		for (int r = 0; r < r_count; r++){
			int z = 0;
			if (s+2 < s_res_integers){
#pragma unroll
				for (int i = 0; i < 32; i++) {
					if (a[s*32+i] + b[s*32+i] == x[r] + y[r]){
						z = z | 1 << i;
					}
				}
			} else if (s+1 < s_res_integers){
				for (int i = 0; i < s_res_integers - s; i++) {
					if (a[s*32+i] + b[s*32+i] == x[r] + y[r]){
						z = z | 1 << i;
					}
				}
			}
			output_buffer[r*s_res_integers+s] = z;
		}
	}
}


__device__ bool update(int location, int s, int r, int *out) {
    int current_r = out[location*2];
    if (current_r == 0) {
        // Current location is empty; We can insert.
        int old = atomicCAS(&out[location*2], 0, r);

        // if old is r we already inserted
        // if old is 0 we swapped
        if (old != 0) {
            if (old != r) {
                return false;
            }
        } 
    } else {//if (current_r != r) {
        return false;
    }
    out[location*2+1] = s;
    return true;
}

__device__ int loc;
__global__ 
void compare_kernel_new_r_atomics_range(int *output_buffer, int outsize, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
   	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;

	int *location = &loc;
	*location = 0;
	for (int s = idx; s < s_count; s += global_threads){
#pragma unroll
		for (int r = 0; r < r_count; r++){
			const a_t a_ = a[s] - x[r];
			const b_t b_ = b[s] - y[r];
			if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
				for (unsigned j = 0;  1999 > j; j++) {
					if (update(*location, s, r, output_buffer)) 
						break;
					atomicAdd(location, 1);
					if (*location == outsize) {
						printf(" output_buffer full!!\n");
					}
				}
			}
		}
	}
}

__global__ 
void compare_kernel_new_s_atomics_range(int *output_buffer, int outsize, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
   	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;

	int *location = &loc;
	*location = 0;
	for (int r = idx; r < r_count; r += global_threads){
#pragma unroll
		for (int s = 0; s < s_count; s++){
			const a_t a_ = a[s] - x[r];
			const b_t b_ = b[s] - y[r];
			if ((a_ > -10) & (a_ < 10) & (b_ > -10.) & (b_ < 10.)){
				for (unsigned j = 0;  1999 > j; j++) {
					if (update(*location, s, r, output_buffer)) 
						break;
					atomicAdd(location, 1);
					if (*location == outsize) {
						printf(" output_buffer full!!\n");
					}
				}
			}
		}
	}
}

__global__ 
void compare_kernel_new_r_atomics(int *output_buffer, int outsize, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
   	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;

	int *location = &loc;
	*location = 0;
	for (int s = idx; s < s_count; s += global_threads){
#pragma unroll
		for (int r = 0; r < r_count; r++){
			if (a[s] + b[s] == x[r] + y[r]){
				for (unsigned j = 0;  1999 > j; j++) {
					if (update(*location, s, r, output_buffer)) 
						break;
					atomicAdd(location, 1);
					if (*location == outsize) {
						printf(" output_buffer full!!\n");
					}
				}
			}
		}
	}
}

__global__ 
void compare_kernel_new_s_atomics(int *output_buffer, int outsize, a_t* a, b_t* b, x_t* x, y_t* y, int s_count, int r_count) {
   	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;

	int *location = &loc;
	*location = 0;
	for (int r = idx; r < r_count; r += global_threads){
#pragma unroll
		for (int s = 0; s < s_count; s++){
			if (a[s] + b[s] == x[r] + y[r]){
				for (unsigned j = 0;  1999 > j; j++) {
					if (update(*location, s, r, output_buffer)) 
						break;
					atomicAdd(location, 1);
					if (*location == outsize) {
						printf(" output_buffer full!!\n");
					}
				}
			}
		}
	}
}
