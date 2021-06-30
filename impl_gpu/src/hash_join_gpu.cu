#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "cuda_helper.h"
#include "kernels_hj.h"
#include "worker.h"

using namespace std;

namespace hj_gpu {

// Constants & Setup vars
//static const int tpl_per_chunk  = 64;

static unsigned ht_size = 0; /* set in init */
static unsigned ht_mask = 0; /* set in init */
static unsigned output_mask = 0; /* set in init */
static unsigned output_buffersize = 0;

// Parameters HT Generation
static const double load_factor = 0.2;
static const unsigned output_size = 2097152; /* 2^21 */

ht* hmR;
ht* hmS;

// Cleanup bitmap
unsigned* cleanup_bitmap_S;
unsigned* cleanup_bitmap_R;
unsigned cleanup_size; /* set in init */

int* output_location;
int* invalid_count_out;

// Output Tuples Buffer
int* output;

/* return p with the smallest k for that holds: p = 2^k > n */
unsigned next_largest_power2(unsigned n)
{
    n--;
    n |= n >> 1; // Divide by 2^k for consecutive doublings of k up to 32,
    n |= n >> 2; // and then 'or' the results.
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void init(master_ctx_t* ctx)
{

    // Init hash table
    cout << "# Use a hast table load factor of " << load_factor << "\n";

    if (ctx->num_tuples_R != ctx->num_tuples_S) {
        cout << "# WARNING: ctx->num_tuples_R != ctx->num_tuples_S. May lead to problems in hash table generation.\n";
    }

    unsigned tpl_per_window = ctx->window_size_R * ctx->rate_R;
    cout << "# Expected number of tuples per window: " << tpl_per_window << "\n";

    ht_size = next_largest_power2(tpl_per_window / ctx->tpl_per_chunk);
    if (ht_size < (tpl_per_window / ctx->tpl_per_chunk) / load_factor) {
        ht_size = ht_size << 1;
        //ht_size = ht_size << 2;
    }
    ht_mask = ht_size - 1;

    cout << "# Use a hash table size of " << ht_size << " chunks with " << ctx->tpl_per_chunk << " tuples\n";

    unsigned chunk_size = ctx->tpl_per_chunk * sizeof(chunk_R);
    cout << "# The chunk size is " << chunk_size << " B\n";
    cout << "# Total hash table size is "
         << (long)ht_size * chunk_size / 1048576 /*MB*/ << " MB\n";

    assert(ht_size != 0);
    assert(ht_size % 2 == 0);
    assert(ht_mask != 0);

    CUDA_SAFE(cudaMalloc(&hmR, ht_size * sizeof(ht)))
    CUDA_SAFE(cudaMalloc(&hmS, ht_size * sizeof(ht)))

    //CUDA_SAFE(cudaHostAlloc(&hmR, ht_size*sizeof(ht),0))
    //CUDA_SAFE(cudaHostAlloc(&hmS, ht_size*sizeof(ht),0))

    ht* hmR_h = (ht*)malloc(ht_size * sizeof(ht));
    CUDA_SAFE(cudaMalloc((void**)&hmR, ht_size * sizeof(ht)))
    for (int i = 0; i < ht_size; i++) {
        chunk_R* chunk;
        CUDA_SAFE(cudaMalloc(&chunk, chunk_size))
        hmR_h[i].address = (uint64_t)chunk;
    }
    CUDA_SAFE(cudaMemcpy(hmR, hmR_h, ht_size * sizeof(ht), cudaMemcpyHostToDevice))

    ht* hmS_h = (ht*)malloc(ht_size * sizeof(ht));
    CUDA_SAFE(cudaMalloc((void**)&hmS, ht_size * sizeof(ht)))
    for (int i = 0; i < ht_size; i++) {
        chunk_S* chunk;
        CUDA_SAFE(cudaMalloc(&chunk, chunk_size))
        hmS_h[i].address = (uint64_t)chunk;
    }
    CUDA_SAFE(cudaMemcpy(hmS, hmS_h, ht_size * sizeof(ht), cudaMemcpyHostToDevice))

    /*chunk_R *chunks_r[];
	chunk_S *chunks_s[];
	for (unsigned i = 0; i < ht_size; i++){
		CUDA_SAFE(cudaMalloc(&(chunks_r[i]), chunk_size))
		CUDA_SAFE(cudaMalloc(&(chunks_s[i]), chunk_size))
	}*/

    /*for (unsigned i = 0; i < ht_size; i++){
		chunk_R *chunk;
		//CUDA_SAFE(cudaMalloc(&chunk, chunk_size))
		CUDA_SAFE(cudaHostAlloc(&chunk, chunk_size,0))
		hmR[i].address = (uint64_t)chunk;
		hmR[i].counter = 0;
	}
	for (unsigned i = 0; i < ht_size; i++){
		chunk_S *chunk;
		//CUDA_SAFE(cudaMalloc(&chunk, chunk_size))
		CUDA_SAFE(cudaHostAlloc(&chunk, chunk_size,0))
		hmS[i].address = (uint64_t)chunk;
		hmS[i].counter = 0;
	}*/

    // Init output tuple buffer
    output_mask = output_size - 1;
    assert(output_size % 2 == 0);
    assert(output_mask != 0);

    output_buffersize = output_size * sizeof(int) * 2;
    cout << "# Total output buffer size is "
         << output_buffersize / 1048576 /*MB*/ << " MB\n";
    CUDA_SAFE(cudaHostAlloc(&output, output_buffersize, 0))

    CUDA_SAFE(cudaHostAlloc(&output_location, sizeof(int), 0))
    *output_location = 0;

    CUDA_SAFE(cudaHostAlloc(&invalid_count_out, sizeof(int), 0))
    *invalid_count_out = 0;

    // Cleanup Bitmap
    cleanup_size = ht_size * sizeof(unsigned);

    CUDA_SAFE(cudaMalloc(&cleanup_bitmap_S, cleanup_size))
    CUDA_SAFE(cudaMemset(cleanup_bitmap_S, 0, cleanup_size))

    CUDA_SAFE(cudaMalloc(&cleanup_bitmap_R, cleanup_size))
    CUDA_SAFE(cudaMemset(cleanup_bitmap_R, 0, cleanup_size))
}

void process_r_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx)
{

    Timer::Timer timer = Timer::Timer();
    auto start_time = timer.now();

    compare_kernel_new_r_hj<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize, w_ctx->gpu_blocksize * sizeof(int)>>>(
        //compare_kernel_new_r_hj<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(
        *(w_ctx->r_processed),
        output, output_size / 2,
        w_ctx->S.a, w_ctx->S.b, w_ctx->R.x, w_ctx->R.y,
        w_ctx->S.t_ns, w_ctx->R.t_ns,
        ctx->generate_tuples_S, ctx->generate_tuples_R,
        ctx->r_iterations, ctx->s_iterations,
        ctx->window_size_R, ctx->window_size_S,
        ctx->rate_R, ctx->rate_S,
        hmR, hmS,
        cleanup_bitmap_S, cleanup_bitmap_R,
        ht_size, ht_size,
        ctx->r_batch_size, output_location,
        invalid_count_out,
        ctx->tpl_per_chunk);

    CUDA_SAFE(cudaDeviceSynchronize())

    int sum = invalid_count_out[0];

    if (sum > ctx->cleanup_threshold) {
        auto end_time = timer.now();
        w_ctx->stats.runtime_probe += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        start_time = timer.now();

        cleanup_s<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(
            *(w_ctx->r_processed),
            output, output_size / 2,
            w_ctx->S.a, w_ctx->S.b, w_ctx->R.x, w_ctx->R.y,
            w_ctx->S.t_ns, w_ctx->R.t_ns,
            ctx->generate_tuples_S, ctx->generate_tuples_R,
            ctx->r_iterations, ctx->s_iterations,
            ctx->window_size_R, ctx->window_size_S,
            ctx->rate_R, ctx->rate_S,
            hmR, hmS,
            cleanup_bitmap_S, cleanup_bitmap_R,
            ht_size);
        CUDA_SAFE(cudaDeviceSynchronize())
        CUDA_SAFE(cudaMemset(cleanup_bitmap_S, 0, cleanup_size))

        end_time = timer.now();
        w_ctx->stats.runtime_cleanup += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    } else {
        auto end_time = timer.now();
        w_ctx->stats.runtime_probe += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }

    *(w_ctx->r_processed) += ctx->r_batch_size;
}

void process_s_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx)
{

    Timer::Timer timer = Timer::Timer();
    auto start_time = timer.now();

    compare_kernel_new_s_hj<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize, w_ctx->gpu_blocksize * sizeof(int)>>>(
        //compare_kernel_new_s_hj<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(
        *(w_ctx->s_processed),
        output, output_size / 2,
        w_ctx->S.a, w_ctx->S.b, w_ctx->R.x, w_ctx->R.y,
        w_ctx->S.t_ns, w_ctx->R.t_ns,
        ctx->generate_tuples_S, ctx->generate_tuples_R,
        ctx->r_iterations, ctx->s_iterations,
        ctx->window_size_R, ctx->window_size_S,
        ctx->rate_R, ctx->rate_S,
        hmR, hmS,
        cleanup_bitmap_S, cleanup_bitmap_R,
        ht_size, ht_size,
        ctx->s_batch_size, output_location,
        invalid_count_out,
        ctx->tpl_per_chunk);

    CUDA_SAFE(cudaDeviceSynchronize())

    int sum = invalid_count_out[0];

    if (sum > ctx->cleanup_threshold) {
        auto end_time = timer.now();
        w_ctx->stats.runtime_probe += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        start_time = timer.now();

        cleanup_r<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(
            *(w_ctx->s_processed),
            output, output_size / 2,
            w_ctx->S.a, w_ctx->S.b, w_ctx->R.x, w_ctx->R.y,
            w_ctx->S.t_ns, w_ctx->R.t_ns,
            ctx->generate_tuples_S, ctx->generate_tuples_R,
            ctx->r_iterations, ctx->s_iterations,
            ctx->window_size_R, ctx->window_size_S,
            ctx->rate_R, ctx->rate_S,
            hmR, hmS,
            cleanup_bitmap_S, cleanup_bitmap_R,
            ht_size);
        CUDA_SAFE(cudaDeviceSynchronize())
        CUDA_SAFE(cudaMemset(cleanup_bitmap_R, 0, cleanup_size))

        end_time = timer.now();
        w_ctx->stats.runtime_cleanup += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    } else {
        auto end_time = timer.now();
        w_ctx->stats.runtime_probe += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        w_ctx->stats.runtime_proc += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }

    *(w_ctx->s_processed) += ctx->s_batch_size;
}

void end_processing(worker_ctx_t* w_ctx)
{
    w_ctx->stats.processed_output_tuples = output_location[0];
}
}
