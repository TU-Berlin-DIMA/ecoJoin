#ifndef KERNELS_H
#define KERNELS_H

#include "worker.h"
#include "master.h"
#include "data.h"

struct ht{
        int counter;
        uint64_t address;
}; // 16 Byte

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

__global__
void compare_kernel_new_s_hj(
		int s_processed,
		int *output_buffer, int outsize,
                int* a, int* b, int* x, int* y,
                long *s_ts, long *r_ts,
                int generate_tuples_S, int generate_tuples_R,
                int r_iterations, int s_iterations,
                int window_size_R, int window_size_S,
                int r_rate, int s_rate,
                ht *hmR, ht *hmS,
                unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
		int ht_size_r, int ht_size_s,
                int count, int* output_location,
		int *invalid_count_out,
		int tup_per_chunk);


__global__
void cleanup_r( int s_processed,
		int *output_buffer, int outsize, 
		int* a, int* b, int* x, int* y,
                long *s_ts, long *r_ts,
                int generate_tuples_S, int generate_tuples_R,
                int r_iterations, int s_iterations,
                int window_size_R, int window_size_S,
                int r_rate, int s_rate,
                ht *hmR, ht *hmS,
                unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
                int count);



__global__
void compare_kernel_new_r_hj(
		int r_processed,
		int *output_buffer, int outsize,
                int* a, int* b, int* x, int* y,
                long *s_ts, long *r_ts,
                int generate_tuples_S, int generate_tuples_R,
                int r_iterations, int s_iterations,
                int window_size_R, int window_size_S,
                int r_rate, int s_rate,
                ht *hmR, ht *hmS,
                unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
		int ht_size_r, int ht_size_s,
                int count, int* output_location,
		int *invalid_count_out,
		int tup_per_chunk);


__global__
void cleanup_s( int r_processed,
		int *output_buffer, int outsize, 
		int* a, int* b, int* x, int* y,
                long *s_ts, long *r_ts,
                int generate_tuples_S, int generate_tuples_R,
                int r_iterations, int s_iterations,
                int window_size_R, int window_size_S,
                int r_rate, int s_rate,
                ht *hmR, ht *hmS,
                unsigned *cleanup_bitmap_S, unsigned *cleanup_bitmap_R,
		int count);

__global__
void checkR(unsigned *cleanup_bitmap_R);

__device__ 
uint32_t location = 0;

#endif /* KERNELS_H */
