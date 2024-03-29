#ifndef HASH_JOIN_MT_AT_CHUNK_WORKER_H
#define HASH_JOIN_MT_AT_CHUNK_WORKER_H

namespace mt_atomic_chunk {

extern std::atomic<uint32_t> processed_tuples;

void init(master_ctx_t* ctx);

void process_r_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx);
void process_s_ht_cpu(master_ctx_t* ctx, worker_ctx_t* w_ctx);

void print_ht(worker_ctx_t* w_ctx);
}
#endif /* HASH_JOIN_MT_AT_CHUNK_WORKER_H */
