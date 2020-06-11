#ifndef HASH_JOIN_MT_AT_CHUNK_WORKER_H
#define HASH_JOIN_MT_AT_CHUNK_WORKER_H
namespace mt_atomic_chunk {

void init_ht();

void process_r_ht_cpu(worker_ctx_t *w_ctx, unsigned threads);
void process_s_ht_cpu(worker_ctx_t *w_ctx, unsigned threads);

void print_ht();
}
#endif  /* HASH_JOIN_MT_AT_CHUNK_WORKER_H */
