#include "MurmurHash.h"
#include "master.h"
#include "worker.h"

// Forward decl.
__device__ bool update(int location, int s, int r, int* out);

/*
 * Includes: 
 *  1) Insertion of Tuple Block into HT S, 
 *  2) Comparision of Tuple Block with HT R, 
 *
 * Cleanup is launched in seperate kernel
 */
__global__ void compare_kernel_new_s_hj(int* output_buffer, int outsize, a_t* a, b_t* b, x_t* x, y_t* y,
    int s_count, int r_count,
    int* cleanup_bitmap)
{
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    const long global_threads = blockDim.x * gridDim.x;

    const s = idx;

    /* 
	 * Insertion 
	 */
    int hash;

    /* get hash */
    MurmurHash_x86_32(a[s], sizeof(int), 0, &hash);
    atomicAdd(hmS[hash].counter, 1);

    ((chunk_S*)hmS[hash].address)[tpl_cntr].a = k;
    /* others .. */

    /* 
	 * Comparison 
	 */

    /* Get hash */
    int tpl_cntr = hmR[hash].counter;

    if (tpl_cntr != 0) {
        const chunk_R* chunk = (chunk_R*)hmR[hash].address; // head
        for (unsigned j = 0; j < tpl_cntr; j++) {
            if ((chunk[j].t_ns.count() + w_ctx->window_size_R * n_sec)
                > s_get_tns(ctx, s).count()) { // Valid
                if (chunk[j].x == k) { // match
                    for (unsigned j = 0; 1999 /* max iterations */ > j; j++) {
                        /* Write into output buffer */
                        if (update(*location, s, r, output_buffer))
                            break;
                        atomicAdd(location, 1);
                        if (*location == outsize) {
                            printf("output_buffer full!\n");
                        }
                    }
                }
            } else { // Invalid
                to_delete_bitmap_R[hash] = 1;
                to_delete_tuples++;
            }
        }
    }
}
}

/*
 * Cleanup Kernel
 * launched in host code if cleanup threshold was reached
 * used threadnumber == ht_size
 */
__global__ void cleanup_kernel_new_s_hj(int* output_buffer, int outsize, a_t* a, b_t* b, x_t* x, y_t* y,
    int s_count, int r_count)
{
    if (to_delete_bitmap_R.test(i)) {
        uint32_t tpl_cnt = hmR[i].counter.load(std::memory_order_relaxed);
        chunk_R* chunk = (chunk_R*)hmR[i].address; // head
        for (size_t j = 0; j < tpl_cnt; j++) { // non-empty
            if ((chunk[j].t_ns.count() + w_ctx->window_size_R * n_sec)
                < s_get_tns(ctx, *(w_ctx->s_processed)).count()) {
                // Remove + Move
                for (int u = 0, l = 0; u < tpl_cnt; u++) {
                    if ((u != j || u != j + 1) && u != l) {
                        chunk[l].t_ns = chunk[u].t_ns;
                        chunk[l].x = chunk[u].x;
                        chunk[l].y = chunk[u].y;
                        chunk[l].r = chunk[u].r;
                        l++;
                    }
                }
                tpl_cnt--;
                hmR[i].counter--; // Update tpl counter
            }
        }
    }
}

__device__ bool update(int location, int s, int r, int* out)
{
    int current_r = out[location * 2];
    if (current_r == 0) {
        // Current location is empty; We can insert.
        int old = atomicCAS(&out[location * 2], 0, r);

        // if old is r we already inserted
        // if old is 0 we swapped
        if (old != 0) {
            if (old != r) {
                return false;
            }
        }
    } else { //if (current_r != r) {
        return false;
    }
    out[location * 2 + 1] = s;
    return true;
}
