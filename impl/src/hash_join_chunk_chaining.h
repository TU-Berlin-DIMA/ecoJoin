#ifndef HASH_JOIN_MT_AT_CHUNK_WORKER_H
#define HASH_JOIN_MT_AT_CHUNK_WORKER_H
/*
static const long n_sec = 1000000000L;

static const int tpl_per_chunk  = 30;
static const int ht_size = 20000;
static const int max_try = ht_size;

using namespace std;

static const int cacheline_size = 64;
static const int chunk_size = tpl_per_chunk * sizeof(uint32_t) * 2;
*/
namespace mt_atomic_chunk {

void init_ht();

void process_r_ht_cpu(worker_ctx_t *w_ctx, unsigned threads);
void process_s_ht_cpu(worker_ctx_t *w_ctx, unsigned threads);

void print_ht(worker_ctx_t *w_ctx);
}
#endif  /* HASH_JOIN_MT_AT_CHUNK_WORKER_H */
