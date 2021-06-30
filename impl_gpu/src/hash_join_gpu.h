#ifndef HASH_JOIN_GPU_H
#define HASH_JOIN_GPU_H

namespace hj_gpu {

void init(master_ctx_t* ctx);

void process_r_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx);
void process_s_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx);

void print_ht(worker_ctx_t* w_ctx);

void end_processing(worker_ctx_t* w_ctx);

}
#endif /* HASH_JOIN_GPU_H */
