#ifndef HASH_JOIN_MT_AT_CHUNK_WORKER_H
#define HASH_JOIN_MT_AT_CHUNK_WORKER_H

namespace mt_atomic_chunk {

extern std::atomic<uint32_t> processed_tuples;

void init_ht();

void process_r_ht_cpu(worker_ctx_t *w_ctx);
void process_s_ht_cpu(worker_ctx_t *w_ctx);

void print_ht(worker_ctx_t *w_ctx);
}
#endif  /* HASH_JOIN_MT_AT_CHUNK_WORKER_H */
