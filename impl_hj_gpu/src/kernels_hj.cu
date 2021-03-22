#include <inttypes.h>
#include <assert.h>

#include "master.h"
#include "worker.h"
#include "MurmurHash.h"


__device__ inline 
int bindex(int b) { return b / (sizeof(unsigned)*8); }

__device__ inline 
int boffset(int b) { return b % (sizeof(unsigned)*8); }
//int boffset(int b) { return b >> 5; }

// FIXME: atomicOr ???
__device__ inline
void set_bit(int b, unsigned *array) { 
    array[bindex(b)] |= (1 << boffset(b));
}

__device__ inline
void clear_bit(int b, unsigned *array) { 
    array[bindex(b)] &= ~(1 << (boffset(b)));
}

__device__  inline
bool get_bit(int b, unsigned *array) { 
    return array[bindex(b)] & (1 << boffset(b));
}

// Struct definitions
struct chunk_R{
	long t_ns; /* timestamp */
	int x; /* key */
	int y; /* value */
	int r; /* index */
}; // 32 Byte

struct chunk_S{
	long t_ns; /* timestamp */
	int a; /* key */
	int b; /* value */
	int s; /* index */
}; // 32 Byte

struct ht{
	int counter;
	uint64_t address;
}; // 16 Byte

static const long n_sec = 1000000000L;

/* 
 * Calculate current timestamp
 */
__device__ inline
long r_get_tns(int generate_tuples_R, int r_iterations, int rate_R, int r, long *r_t_ns){
	if (r_iterations > 1) {
		return r_t_ns[r & (generate_tuples_R-1)]
			+ (r_iterations-1) * generate_tuples_R * (long)(1.e9 / rate_R);
	} else {
		return r_t_ns[r];
	}
}

__device__ inline
long s_get_tns(int generate_tuples_S, int s_iterations, int rate_S, int s, long *s_t_ns){
	if (s_iterations > 1) {
		return s_t_ns[s & (generate_tuples_S-1)]
			+ (s_iterations-1) * generate_tuples_S * (long)(1.e9 / rate_S);
	} else {
		return s_t_ns[s];
	}
}

/*
 * Includes: 
 *  1) Insertion of Tuple Block into HT S, 
 *  2) Comparision of Tuple Block with HT R, 
 *
 * Cleanup is launched in seperate kernel
 */
__global__ 
void compare_kernel_new_s_hj(
		int s_processed,
		int *output_buffer, int outsize_mask, 
		int *a, int *b, int *x, int *y, 
		long *s_ts, long *r_ts,
		int generate_tuples_S, int generate_tuples_R,
		int r_iterations, int s_iterations,
		int window_size_R, int window_size_S,
		int r_rate, int s_rate,
		ht *hmR, ht *hmS,
		unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
		int ht_size_r, int ht_size_s,
		int count, int* output_location,
		int *invalid_count_out) {
	unsigned int tid = threadIdx.x;
	/* const size_t idx = threadIdx.x + blockIdx.x * blockDim.x; */
	/* const long global_threads = blockDim.x * gridDim.x; */

    const int lane_id = threadIdx.x % warpSize;
    constexpr int leader_id = 0;
    const int warp_id = threadIdx.x / warpSize;
    const int num_warps = blockDim.x / warpSize;
    constexpr uint32_t warp_mask = 0xFFFFFFFFu;

    const int global_warp_id = blockIdx.x * num_warps + warp_id;
    const int global_warps = gridDim.x * num_warps;

	int invalid_count = 0;
	//if (idx == 0)
	//	invalid_count_out[0] = 0;
	extern __shared__ int sdata[];
        for (int s = global_warp_id + s_processed; s < count + s_processed; s += global_warps){

		const int k = a[s];
		//printf("add s: %d %d %d\n",k, s_processed, ht_size_s);

        // FIXME: parallelize with warp
        int hash = 0;
        if (lane_id == leader_id) {
            /* 
             * Build
             */

            /* get hash */
            MurmurHash_x86_32((void*)&k, sizeof(int), 0, &hash);

            hash = hash & (ht_size_s-1);
            int tpl_cntr = atomicAdd(&(hmS[hash].counter), 1);
            
            if (tpl_cntr >= 64) {
                printf("Chunk full at index: %d in S, hash: %d, s: %d \n", tpl_cntr, hash, s);
                __threadfence();
                assert(0);
            }

            chunk_S *chunk = (chunk_S*) hmS[hash].address;
            chunk[tpl_cntr].a = k;
            chunk[tpl_cntr].t_ns = s_get_tns(generate_tuples_S, s_iterations, s_rate, s, s_ts);
            chunk[tpl_cntr].b = b[s];
            chunk[tpl_cntr].s = s;
        }

		/* 
		 * Probe
		 */
		hash = __shfl_sync(warp_mask, hash, leader_id);
		int tpl_cntr = hmR[hash].counter;

		if (tpl_cntr != 0){
			const chunk_R *chunk = (chunk_R*) hmR[hash].address; // head
			for (int j = lane_id; j < tpl_cntr; j += warpSize){
				/*if (hash == 203630) {
					printf("%ld %ld\n", (chunk[j].t_ns + window_size_R * n_sec),
							s_get_tns(generate_tuples_S, s_iterations, s_rate, s, s_ts));
				}*/
				if ((chunk[j].t_ns + window_size_R * n_sec)
						> s_get_tns(generate_tuples_S, s_iterations, s_rate, s, s_ts)) { // Valid
					if (chunk[j].x == k) { // match
						int i = atomicAdd(output_location, 1) & outsize_mask;
						//printf("from s: %d %d\n",chunk[j].r, s);

						// Write into output buffer
						output_buffer[i*2]   = chunk[j].r;
						output_buffer[i*2+1] = s;
						
						//atomicAdd(num_out_tuples, 1);
					}
				} else { // Invalid
					set_bit(hash, cleanup_bitmap_R);
					atomicAdd(invalid_count_out, 1);
					//invalid_count_out[0]++;
					//invalid_count++;
				}
			}
		} 
	}

	/*sdata[tid] = invalid_count;
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) { 
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s]; 
		}
		__syncthreads(); 
	}
	// write result for this block to global mem
	if (tid == 0) 
		invalid_count_out[blockIdx.x] = sdata[0];*/
}

/*
 * Cleanup Kernel
 * launched in host code if cleanup threshold was reached
 * used threadnumber == ht_size
 */
__global__ 
void cleanup_r( int s_processed,
		int *output_buffer, int outsize_mask, 
		int *a, int *b, int *x, int *y, 
		long *s_ts, long *r_ts,
		int generate_tuples_S, int generate_tuples_R,
		int r_iterations, int s_iterations,
		int window_size_R, int window_size_S,
		int r_rate, int s_rate,
		ht *hmR, ht *hmS,
		unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
		int count) {
	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;

	for (int i = idx; i < count; i += global_threads){
		if (get_bit(i, cleanup_bitmap_R)){
			uint32_t tpl_cnt = hmR[i].counter;
			chunk_R *chunk = (chunk_R*) hmR[i].address; // head
			for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
				// find first correct entry
				if ((chunk[j].t_ns + window_size_S * n_sec)
						> s_get_tns(generate_tuples_S, s_iterations, s_rate, s_processed, s_ts)) {
					// Remove + Move
					int l = 0;
					for(size_t u = j; u < tpl_cnt; u++, l++) { // non-empty
						chunk[l].t_ns = chunk[u].t_ns;
                                                chunk[l].x = chunk[u].x;
                                                chunk[l].y = chunk[u].y;
                                                chunk[l].r = chunk[u].r;
					}
					hmR[i].counter -= j;
					break;
				}

				// all entires are invalid
				if (tpl_cnt == j-1)
					hmR[i].counter = 0;
			}
			
		}
	}
}



__global__ 
void compare_kernel_new_r_hj(
		int r_processed,
		int *output_buffer, int outsize_mask, 
		int *a, int *b, int *x, int *y, 
		long *s_ts, long *r_ts,
		int generate_tuples_S, int generate_tuples_R,
		int r_iterations, int s_iterations,
		int window_size_R, int window_size_S,
		int r_rate, int s_rate,
		ht *hmR, ht *hmS,
		unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
		int ht_size_r, int ht_size_s,
		int count, int* output_location,
		int *invalid_count_out) {
	unsigned int tid = threadIdx.x;
	/* const size_t idx = threadIdx.x + blockIdx.x * blockDim.x; */
	/* const long global_threads = blockDim.x * gridDim.x; */

    const int lane_id = threadIdx.x % warpSize;
    constexpr int leader_id = 0;
    const int warp_id = threadIdx.x / warpSize;
    const int num_warps = blockDim.x / warpSize;
    constexpr uint32_t warp_mask = 0xFFFFFFFFu;

    const int global_warp_id = blockIdx.x * num_warps + warp_id;
    const int global_warps = gridDim.x * num_warps;

	//if (idx == 0)
	//	invalid_count_out[0] = 0;

	int invalid_count = 0;
	extern __shared__ int sdata[];
	for (int r = global_warp_id + r_processed; r < count + r_processed; r += global_warps){

		const int k = x[r];

        // FIXME: parallelize with warp
        int hash = 0;
        if (lane_id == leader_id) {
            /* 
             * Build
             */ 

            /* get hash */
            MurmurHash_x86_32((void*)&k, sizeof(int), 0, &hash);

            hash = hash & (ht_size_r-1);
            int tpl_cntr = atomicAdd(&(hmR[hash].counter), 1);
            
            if (tpl_cntr >= 64) {
                printf("%d\n", window_size_R);
                printf("Chunk full at index: %d in R, hash: %d, r: %d\n", tpl_cntr, hash, r);
                __threadfence();
                assert(0);
            }

            chunk_R *chunk = (chunk_R*) hmR[hash].address;
            chunk[tpl_cntr].x = k;
            chunk[tpl_cntr].t_ns = r_get_tns(generate_tuples_R, r_iterations, r_rate, r, r_ts);
            chunk[tpl_cntr].y = y[r];
            chunk[tpl_cntr].r = r;
        }

		/* 
		 * Probe
		 */
		hash = __shfl_sync(warp_mask, hash, leader_id);
        int tpl_cntr = hmS[hash].counter;

		if (tpl_cntr != 0){
			const chunk_S *chunk = (chunk_S*) hmS[hash].address; // head
			for (int j = lane_id; j < tpl_cntr; j += warpSize){
				if ((chunk[j].t_ns + window_size_S * n_sec)
						> r_get_tns(generate_tuples_R, r_iterations, r_rate, r, r_ts)) { // Valid
					if (chunk[j].a == k) { // match
						int i = atomicAdd(output_location, 1) & outsize_mask;

						// Write into output buffer 
						output_buffer[i*2]   = r;
						output_buffer[i*2+1] = chunk[j].s;
						
						//printf("from r: %d %d\n",r, chunk[j].s);
						//atomicAdd(num_out_tuples, 1);
					}
				} else { // Invalid
					set_bit(hash, cleanup_bitmap_S);
					atomicAdd(invalid_count_out, 1);
					//invalid_count_out[0]++;
					//invalid_count++;
				}
			}
		}
	}

	/*sdata[tid] = invalid_count;
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) { 
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s]; 
		}
		__syncthreads(); 
	}
	// write result for this block to global mem
	if (tid == 0) 
		invalid_count_out[blockIdx.x] = sdata[0];*/
		
}

/*
 * Cleanup Kernel
 * launched in host code if cleanup threshold was reached
 * used threadnumber == ht_size
 */
__global__ 
void cleanup_s(	int r_processed,
		int *output_buffer, int outsize_mask, 
		int *a, int *b, int *x, int *y, 
		long *s_ts, long *r_ts,
		int generate_tuples_S, int generate_tuples_R,
		int r_iterations, int s_iterations,
		int window_size_R, int window_size_S,
		int r_rate, int s_rate,
		ht *hmR, ht *hmS,
		unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
		int count) {
	const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long global_threads = blockDim.x * gridDim.x;

	for (int i = idx; i < count; i += global_threads){
		if (get_bit(i, cleanup_bitmap_S)){
			uint32_t tpl_cnt = hmS[i].counter;
			chunk_S *chunk = (chunk_S*) hmS[i].address; // head
			for(size_t j = 0; j < tpl_cnt; j++) { // non-empty
				// find first correct entry
				if ((chunk[j].t_ns + window_size_S * n_sec)
						> r_get_tns(generate_tuples_R, r_iterations, r_rate, r_processed, r_ts)) {
					// Remove + Move
					int l = 0;
					for(size_t u = j; u < tpl_cnt; u++,  l++) { // non-empty
						chunk[l].t_ns = chunk[u].t_ns;
                                                chunk[l].a = chunk[u].a;
                                                chunk[l].b = chunk[u].b;
                                                chunk[l].s = chunk[u].s;
					}
					hmS[i].counter -= j;
					//printf("%d\n",hmS[i].counter);
					break;
				}

				// all entires are invalid
				if (tpl_cnt == j-1)
					hmS[i].counter = 0;
			}
		}
	}
}
